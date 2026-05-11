#!/usr/bin/env python3
"""
Amalgama Agent — the Vessel.
Runs on client hardware. Contains no proprietary logic.
All intelligence lives in the Platform API.

Usage:
    python amalgama_agent.py \
        --model_a /path/to/model_a \
        --model_b /path/to/model_b \
        --output  /path/to/merged \
        --api_key sk-amalgama-xxxx
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import re

import requests
import torch
import yaml
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.amalgama.ai/v1"  # overridden by --api_base
MAX_ATTEMPTS = 3
RETRY_COUNT = 3
BACKOFF_BASE = 2  # seconds

HUMANEVAL_PROBLEMS = 50
GSM8K_PROBLEMS = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[amalgama] {msg}", flush=True)


def api_call(method: str, url: str, api_key: str, **kwargs) -> dict:
    """Make an API call with retry logic and exponential backoff."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_exc = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
            if resp.status_code == 401:
                log("ERROR: Invalid API key. Obtain a valid key at https://amalgama.ai.")
                sys.exit(1)
            if resp.status_code == 422:
                body = resp.json()
                log(f"ERROR: Validation error from platform: {body}")
                sys.exit(1)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            log(f"HTTP error on attempt {attempt}/{RETRY_COUNT}: {e}")
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            log(f"Connection error on attempt {attempt}/{RETRY_COUNT}: {e}")
        except requests.exceptions.Timeout as e:
            last_exc = e
            log(f"Timeout on attempt {attempt}/{RETRY_COUNT}: {e}")
        if attempt < RETRY_COUNT:
            wait = BACKOFF_BASE ** attempt
            log(f"Retrying in {wait}s…")
            time.sleep(wait)
    log(f"ERROR: Platform unreachable after {RETRY_COUNT} attempts. Last error: {last_exc}")
    sys.exit(1)


def post_error(job_id: str, api_key: str, stage: str, message: str) -> None:
    """Best-effort error report to the platform."""
    try:
        url = f"{BASE_URL}/jobs/{job_id}/error"
        requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"stage": stage, "message": message},
            timeout=10,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Step 1 — Registration
# ---------------------------------------------------------------------------


def register_job(model_a: str, model_b: str, api_key: str) -> str:
    log("Registering job with Amalgama platform…")
    result = api_call(
        "POST",
        f"{BASE_URL}/jobs/create",
        api_key,
        json={"api_key": api_key, "model_a_path": model_a, "model_b_path": model_b},
    )
    job_id = result["job_id"]
    log(f"Job created: {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Step 2 — Architecture compatibility check
# ---------------------------------------------------------------------------

COMPAT_KEYS = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]


def check_compatibility(model_a: str, model_b: str, job_id: str, api_key: str) -> None:
    log("Loading model configs for compatibility check…")
    try:
        cfg_a = AutoConfig.from_pretrained(model_a)
        cfg_b = AutoConfig.from_pretrained(model_b)
    except Exception as e:
        msg = f"Failed to load model config: {e}"
        log(f"ERROR: {msg}")
        post_error(job_id, api_key, "compatibility", msg)
        sys.exit(1)

    comparison = {}
    compatible = True
    for key in COMPAT_KEYS:
        val_a = getattr(cfg_a, key, None)
        val_b = getattr(cfg_b, key, None)
        match = val_a == val_b
        comparison[key] = {"model_a": val_a, "model_b": val_b, "match": match}
        if not match:
            compatible = False

    log(f"Compatibility: {'PASS' if compatible else 'FAIL'}")
    for key, vals in comparison.items():
        mark = "OK" if vals["match"] else "MISMATCH"
        log(f"  {key}: {vals['model_a']} vs {vals['model_b']} [{mark}]")

    result = api_call(
        "POST",
        f"{BASE_URL}/jobs/{job_id}/compatibility",
        api_key,
        json={"compatible": compatible, "comparison": comparison},
    )

    if not compatible:
        log("ERROR: Models are architecturally incompatible. Cannot merge.")
        sys.exit(1)

    log("Architecture compatibility confirmed.")


# ---------------------------------------------------------------------------
# Step 3 — Baseline benchmarking
# ---------------------------------------------------------------------------


def run_humaneval(model_path: str, num_problems: int = HUMANEVAL_PROBLEMS) -> float:
    """
    Run HumanEval benchmark subset.
    Uses the `evaluate_functional_correctness` approach via datasets + transformers.
    Returns pass@1 score in [0, 1].
    """
    try:
        log(f"  Loading tokenizer and model from {model_path}…")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        model.eval()

        dataset = load_dataset("openai_humaneval", split="test")
        problems = list(dataset)[:num_problems]

        passed = 0
        for i, problem in enumerate(problems):
            prompt = problem["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            full_code = problem["prompt"] + completion + "\n" + problem["test"]
            try:
                exec_globals: dict = {}
                exec(full_code, exec_globals)  # noqa: S102
                exec_globals["check"](exec_globals[problem["entry_point"]])  # noqa: S102
                passed += 1
            except Exception:
                pass
            if (i + 1) % 10 == 0:
                log(f"  HumanEval: {i + 1}/{num_problems} ({passed} passed so far)…")

        del model
        torch.cuda.empty_cache()
        score = passed / num_problems
        log(f"  HumanEval score: {score:.3f} ({passed}/{num_problems})")
        return score

    except Exception as e:
        log(f"  WARNING: HumanEval failed ({e}). Returning 0.0.")
        return 0.0


def run_gsm8k(model_path: str, num_problems: int = GSM8K_PROBLEMS) -> float:
    """
    Run GSM8K benchmark subset.
    Uses few-shot chain-of-thought prompting, extracts final numeric answer.
    Returns exact-match accuracy in [0, 1].
    """
    try:
        FEW_SHOT = (
            "Q: Natalia sold clips to 48 of her friends in April and then sold half as many clips"
            " in May. How many clips did Natalia sell altogether in April and May?\n"
            "A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether. #### 72\n\n"
            "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of"
            " babysitting. How much did she earn?\n"
            "A: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10\n\n"
        )

        log(f"  Loading tokenizer and model from {model_path}…")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        model.eval()

        dataset = load_dataset("gsm8k", "main", split="test")
        problems = list(dataset)[:num_problems]

        passed = 0
        for i, problem in enumerate(problems):
            prompt = FEW_SHOT + f"Q: {problem['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Extract answer after ####
            pred_match = re.search(r"####\s*(-?\d[\d,]*)", completion)
            gold_match = re.search(r"####\s*(-?\d[\d,]*)", problem["answer"])
            if pred_match and gold_match:
                pred = pred_match.group(1).replace(",", "")
                gold = gold_match.group(1).replace(",", "")
                if pred == gold:
                    passed += 1
            if (i + 1) % 10 == 0:
                log(f"  GSM8K: {i + 1}/{num_problems} ({passed} passed so far)…")

        del model
        torch.cuda.empty_cache()
        score = passed / num_problems
        log(f"  GSM8K score: {score:.3f} ({passed}/{num_problems})")
        return score

    except Exception as e:
        log(f"  WARNING: GSM8K failed ({e}). Returning 0.0.")
        return 0.0


def run_baselines(model_a: str, model_b: str, job_id: str, api_key: str) -> dict:
    log("Running baseline benchmarks on Model A…")
    a_humaneval = run_humaneval(model_a)
    a_gsm8k = run_gsm8k(model_a)

    log("Running baseline benchmarks on Model B…")
    b_humaneval = run_humaneval(model_b)
    b_gsm8k = run_gsm8k(model_b)

    baselines = {
        "model_a": {"humaneval": a_humaneval, "gsm8k": a_gsm8k},
        "model_b": {"humaneval": b_humaneval, "gsm8k": b_gsm8k},
    }

    log(f"Baselines — A: HumanEval={a_humaneval:.3f}, GSM8K={a_gsm8k:.3f}")
    log(f"Baselines — B: HumanEval={b_humaneval:.3f}, GSM8K={b_gsm8k:.3f}")

    api_call(
        "POST",
        f"{BASE_URL}/jobs/{job_id}/baselines",
        api_key,
        json=baselines,
    )
    log("Baselines posted to platform.")
    return baselines


# ---------------------------------------------------------------------------
# Step 4 — Request merge parameters
# ---------------------------------------------------------------------------


def get_merge_parameters(job_id: str, api_key: str) -> dict:
    log("Requesting merge parameters from platform brain…")
    result = api_call("GET", f"{BASE_URL}/jobs/{job_id}/merge_parameters", api_key)
    log(f"  Attempt:   {result['attempt']}")
    log(f"  Algorithm: {result['algorithm']}")
    log(f"  Parameters: {result['parameters']}")
    log(f"  Rationale: {result['rationale']}")
    return result


# ---------------------------------------------------------------------------
# Step 5 — Execute merge
# ---------------------------------------------------------------------------

MERGEKIT_CONFIGS = {
    "ties": lambda params, model_a, model_b, dtype: {
        "models": [
            {"model": model_a, "parameters": {"density": params.get("density", 0.5), "weight": 1.0}},
            {"model": model_b, "parameters": {"density": params.get("density", 0.5), "weight": params.get("weight", 1.0)}},
        ],
        "merge_method": "ties",
        "base_model": model_a,
        "parameters": {"normalize": True},
        "dtype": dtype,
    },
    "dare_ties": lambda params, model_a, model_b, dtype: {
        "models": [
            {"model": model_a, "parameters": {"density": params.get("density", 0.5), "weight": 1.0}},
            {"model": model_b, "parameters": {"density": params.get("density", 0.5), "weight": params.get("weight", 1.0)}},
        ],
        "merge_method": "dare_ties",
        "base_model": model_a,
        "parameters": {"normalize": True},
        "dtype": dtype,
    },
    "linear": lambda params, model_a, model_b, dtype: {
        "models": [
            {"model": model_a, "parameters": {"weight": params.get("weight_a", 0.5)}},
            {"model": model_b, "parameters": {"weight": params.get("weight_b", 0.5)}},
        ],
        "merge_method": "linear",
        "dtype": dtype,
    },
    "slerp": lambda params, model_a, model_b, dtype: {
        "slices": [
            {
                "sources": [
                    {"model": model_a, "layer_range": [0, None]},
                    {"model": model_b, "layer_range": [0, None]},
                ],
                "parameters": {"t": params.get("t", 0.5)},
            }
        ],
        "merge_method": "slerp",
        "base_model": model_a,
        "dtype": dtype,
    },
}


def build_mergekit_config(merge_params: dict, model_a: str, model_b: str) -> dict:
    algorithm = merge_params.get("algorithm", "ties")
    parameters = merge_params.get("parameters", {})
    dtype = merge_params.get("dtype", "bfloat16")

    builder = MERGEKIT_CONFIGS.get(algorithm)
    if builder is None:
        log(f"WARNING: Unknown algorithm '{algorithm}', falling back to ties.")
        builder = MERGEKIT_CONFIGS["ties"]

    return builder(parameters, model_a, model_b, dtype)


def execute_merge(
    model_a: str,
    model_b: str,
    output_path: str,
    merge_params: dict,
    job_id: str,
    api_key: str,
) -> None:
    config = build_mergekit_config(merge_params, model_a, model_b)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="amalgama_merge_"
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    log(f"MergeKit config written to {config_path}")
    log(f"Running merge → {output_path}…")

    try:
        result = subprocess.run(
            [
                "mergekit-yaml",
                config_path,
                output_path,
                "--copy-tokenizer",
                "--allow-crimes",
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour timeout for large models
        )
        os.unlink(config_path)

        if result.returncode != 0:
            msg = f"mergekit-yaml exited {result.returncode}:\n{result.stderr}"
            log(f"ERROR: {msg}")
            post_error(job_id, api_key, "merge", msg)
            sys.exit(1)

        log("Merge complete.")
    except subprocess.TimeoutExpired:
        os.unlink(config_path)
        msg = "mergekit-yaml timed out after 2 hours."
        log(f"ERROR: {msg}")
        post_error(job_id, api_key, "merge", msg)
        sys.exit(1)
    except FileNotFoundError:
        os.unlink(config_path)
        msg = "mergekit-yaml not found. Install mergekit: pip install mergekit"
        log(f"ERROR: {msg}")
        post_error(job_id, api_key, "merge", msg)
        sys.exit(1)

    api_call(
        "POST",
        f"{BASE_URL}/jobs/{job_id}/merge_complete",
        api_key,
        json={"attempt": merge_params["attempt"], "output_path": output_path},
    )
    log("Merge completion posted to platform.")


# ---------------------------------------------------------------------------
# Step 6 — Post-merge benchmarking
# ---------------------------------------------------------------------------


def run_merged_benchmarks(
    merged_path: str,
    attempt: int,
    job_id: str,
    api_key: str,
) -> dict:
    log(f"Running post-merge benchmarks (attempt {attempt})…")
    humaneval = run_humaneval(merged_path)
    gsm8k = run_gsm8k(merged_path)

    scores = {"attempt": attempt, "humaneval": humaneval, "gsm8k": gsm8k}
    log(f"Merged scores — HumanEval={humaneval:.3f}, GSM8K={gsm8k:.3f}")

    api_call(
        "POST",
        f"{BASE_URL}/jobs/{job_id}/merged_scores",
        api_key,
        json=scores,
    )
    log("Merged scores posted to platform.")
    return scores


# ---------------------------------------------------------------------------
# Step 7 — Request verdict
# ---------------------------------------------------------------------------


def get_verdict(job_id: str, api_key: str) -> dict:
    log("Requesting verdict from platform…")
    result = api_call("GET", f"{BASE_URL}/jobs/{job_id}/verdict", api_key)
    log(f"Verdict: {result['verdict']}")
    log(f"Reason:  {result['reason']}")
    return result


# ---------------------------------------------------------------------------
# Step 8 — Retry loop managed externally (see main)
# ---------------------------------------------------------------------------


def delete_merged_output(output_path: str) -> None:
    if os.path.exists(output_path):
        log(f"Deleting failed merge output at {output_path}…")
        shutil.rmtree(output_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 9 — Download and save report
# ---------------------------------------------------------------------------


def download_report(output_path: str, job_id: str, api_key: str) -> None:
    log("Downloading final report…")
    result = api_call("GET", f"{BASE_URL}/jobs/{job_id}/report", api_key)

    report_path = os.path.join(output_path, "amalgama_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Report saved to {report_path}")

    print_certification_table(result)


def print_certification_table(report: dict) -> None:
    certified = report.get("certified", False)
    status = "CERTIFIED" if certified else "NOT CERTIFIED"
    border = "=" * 60

    print()
    print(border)
    print(f"  AMALGAMA MERGE REPORT — {status}")
    print(border)
    print(f"  Job ID:    {report.get('job_id', 'N/A')}")
    print(f"  Certified: {certified}")
    print()

    baselines = report.get("baselines", {})
    if baselines:
        print("  BASELINES")
        print(f"    Model A — HumanEval: {baselines.get('model_a', {}).get('humaneval', 'N/A'):.3f}  "
              f"GSM8K: {baselines.get('model_a', {}).get('gsm8k', 'N/A'):.3f}")
        print(f"    Model B — HumanEval: {baselines.get('model_b', {}).get('humaneval', 'N/A'):.3f}  "
              f"GSM8K: {baselines.get('model_b', {}).get('gsm8k', 'N/A'):.3f}")
        print()

    attempts = report.get("attempts", [])
    if attempts:
        print("  ATTEMPTS")
        for a in attempts:
            print(f"    #{a.get('attempt_number', '?')} {a.get('algorithm', '?')} "
                  f"— HumanEval: {a.get('humaneval', 'N/A'):.3f}  "
                  f"GSM8K: {a.get('gsm8k', 'N/A'):.3f}  "
                  f"[{a.get('verdict', '?').upper()}]")
        print()

    narrative = report.get("narrative", "")
    if narrative:
        print("  SUMMARY")
        for line in narrative.splitlines():
            print(f"    {line}")
        print()

    print(border)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amalgama Agent — intelligent model merger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_a", required=True, help="Path to Model A")
    parser.add_argument("--model_b", required=True, help="Path to Model B")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--api_key", required=True, help="Amalgama API key (sk-amalgama-…)")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help=f"Maximum merge retry attempts (default: {MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--api_base",
        default=BASE_URL,
        help=f"Platform API base URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--resume",
        metavar="JOB_ID",
        default=None,
        help="Resume an interrupted job by its ID (skips steps already completed)",
    )
    return parser.parse_args()


def fetch_resume_state(job_id: str, api_key: str) -> dict:
    return api_call("GET", f"{BASE_URL}/jobs/{job_id}/resume_state", api_key)


def main() -> None:
    args = parse_args()

    global BASE_URL
    BASE_URL = args.api_base.rstrip("/")

    model_a = str(Path(args.model_a).resolve())
    model_b = str(Path(args.model_b).resolve())
    output_path = str(Path(args.output).resolve())
    api_key = args.api_key
    max_attempts = args.max_attempts

    if not os.path.isdir(model_a):
        log(f"ERROR: model_a path does not exist: {model_a}")
        sys.exit(1)
    if not os.path.isdir(model_b):
        log(f"ERROR: model_b path does not exist: {model_b}")
        sys.exit(1)

    os.makedirs(output_path, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resume path
    # -----------------------------------------------------------------------
    if args.resume:
        job_id = args.resume
        log(f"Resuming job {job_id}…")
        state = fetch_resume_state(job_id, api_key)
        status = state["status"]
        attempts_completed = state["attempts_completed"]
        log(f"  Status: {status}  |  Attempts completed: {attempts_completed}")

        if status in ("complete", "failed", "unmergeable", "failed_incompatible"):
            log("Job already finished. Downloading report.")
            download_report(output_path, job_id, api_key)
            sys.exit(0)

        # Decide where to re-enter
        skip_compat = status not in ("created",)
        skip_baselines = state["baselines_recorded"]
        start_attempt = attempts_completed + 1

        # If the last attempt has a retry verdict it means scores were posted
        # but the next attempt's merge hasn't started — start_attempt is correct.
        # If the last attempt has no verdict yet (interrupted mid-merge or
        # mid-benchmark), re-run that attempt from the top.
        last_verdict = state.get("last_verdict") or {}
        if attempts_completed > 0 and last_verdict.get("verdict") not in ("retry", "certified", "unmergeable"):
            start_attempt = attempts_completed  # redo incomplete attempt
            log(f"  Last attempt {attempts_completed} was incomplete — re-running it.")
            delete_merged_output(output_path)
            os.makedirs(output_path, exist_ok=True)

        if not skip_compat:
            check_compatibility(model_a, model_b, job_id, api_key)
        else:
            log("  Skipping compatibility check (already done).")

        if not skip_baselines:
            run_baselines(model_a, model_b, job_id, api_key)
        else:
            log("  Skipping baselines (already recorded).")

    # -----------------------------------------------------------------------
    # Fresh start path
    # -----------------------------------------------------------------------
    else:
        # Step 1 — Register
        job_id = register_job(model_a, model_b, api_key)

        # Step 2 — Architecture compatibility
        check_compatibility(model_a, model_b, job_id, api_key)

        # Step 3 — Baselines
        run_baselines(model_a, model_b, job_id, api_key)

        start_attempt = 1

    # -----------------------------------------------------------------------
    # Merge loop
    # -----------------------------------------------------------------------
    certified = False
    for attempt in range(start_attempt, max_attempts + 1):
        log(f"\n--- Merge attempt {attempt}/{max_attempts} ---")

        # Step 4 — Get merge parameters
        merge_params = get_merge_parameters(job_id, api_key)

        # Step 5 — Execute merge
        execute_merge(model_a, model_b, output_path, merge_params, job_id, api_key)

        # Step 6 — Post-merge benchmarks
        run_merged_benchmarks(output_path, attempt, job_id, api_key)

        # Step 7 — Verdict
        verdict = get_verdict(job_id, api_key)

        if verdict["verdict"] == "certified":
            certified = True
            log("Merge certified by platform.")
            break
        elif verdict["verdict"] == "unmergeable":
            log("Platform determined models are unmergeable after all attempts.")
            break
        elif verdict["verdict"] == "retry":
            if attempt < max_attempts:
                log(f"Retrying with new parameters…")
                delete_merged_output(output_path)
                os.makedirs(output_path, exist_ok=True)
            else:
                log(f"Maximum attempts ({max_attempts}) reached without certification.")
        else:
            log(f"Unknown verdict '{verdict['verdict']}'. Stopping.")
            break

    # Step 9 — Download report (always, regardless of certification)
    download_report(output_path, job_id, api_key)

    if not certified:
        log("Job complete. Merge was NOT certified. See report for details.")
        sys.exit(2)

    log("Job complete. Merge certified.")


if __name__ == "__main__":
    main()
