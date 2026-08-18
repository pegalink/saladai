FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install unsloth transformers trl peft accelerate bitsandbytes datasets huggingface_hub

WORKDIR /app
COPY train.py .
COPY dataset.jsonl .

CMD ["python3", "train.py"]
