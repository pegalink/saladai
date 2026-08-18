# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir unsloth transformers trl peft accelerate bitsandbytes datasets huggingface_hub

WORKDIR /app
COPY download_assets.py .
COPY train.py .

# Build-time download using a BuildKit secret - HF_TOKEN never gets baked
# into an image layer this way.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python3 download_assets.py

CMD ["python3", "train.py"]
