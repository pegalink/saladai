# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir unsloth trl peft accelerate bitsandbytes datasets huggingface_hub
RUN pip install --no-cache-dir --upgrade --no-deps git+https://github.com/huggingface/transformers.git

WORKDIR /app

COPY download_assets.py .

# BuildKit secret mount: HF_TOKEN is available as an env var ONLY during this
# RUN step, and is never written into any image layer or build cache.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python3 download_assets.py

COPY train.py .

CMD ["python3", "train.py"]
