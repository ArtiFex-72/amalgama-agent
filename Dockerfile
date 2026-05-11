FROM python:3.11-slim

# Install system deps for torch/mergekit
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY amalgama_agent.py .

ENTRYPOINT ["python", "amalgama_agent.py"]
