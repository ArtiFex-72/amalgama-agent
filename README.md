# amalgama-agent

The Vessel — open source client for the [Amalgama](https://amalgama.ai) model merging platform.

Runs entirely on your own hardware. Contains no proprietary logic.
All merge intelligence lives in the Amalgama platform API.

## What it does

1. Registers a merge job with the Amalgama platform
2. Checks architectural compatibility between your two models
3. Benchmarks both models (HumanEval, GSM8K)
4. Receives merge parameters from the platform brain
5. Executes the merge locally via [mergekit](https://github.com/arcee-ai/mergekit)
6. Benchmarks the merged model
7. Requests a verdict — certified or retry with new parameters
8. Repeats up to 3 times if needed
9. Downloads and saves the certification report

## Installation

```bash
curl -O https://raw.githubusercontent.com/ArtiFex-72/amalgama-agent/main/amalgama_agent.py
pip install transformers mergekit datasets torch accelerate pyyaml requests
```

## Usage

```bash
python amalgama_agent.py \
  --model_a /path/to/model_a \
  --model_b /path/to/model_b \
  --output  /path/to/merged_output \
  --api_key sk-amalgama-xxxx
```

Get an API key at [amalgama.ai](https://amalgama.ai).

## Options

| Flag | Default | Description |
|---|---|---|
| `--model_a` | required | Path to first model directory |
| `--model_b` | required | Path to second model directory |
| `--output` | required | Output directory for merged model |
| `--api_key` | required | Amalgama API key (`sk-amalgama-…`) |
| `--max_attempts` | `3` | Maximum merge retry attempts |
| `--api_base` | `https://api.amalgama.ai/v1` | Platform API URL (for self-hosting or dev) |

## Docker

```bash
docker run --rm \
  -v /path/to/models:/models \
  ghcr.io/artifex00-00/amalgama-agent \
  --model_a /models/model_a \
  --model_b /models/model_b \
  --output  /models/merged \
  --api_key sk-amalgama-xxxx
```

## Requirements

- Python 3.10+
- CUDA GPU with enough VRAM to load both models simultaneously
- ~2× model size in free disk space for the merge output

## License

MIT
