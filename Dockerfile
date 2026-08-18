FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir unsloth transformers trl peft accelerate bitsandbytes datasets huggingface_hub

WORKDIR /app
COPY train.py .

CMD ["python3", "train.py"]
