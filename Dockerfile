FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir unsloth trl peft accelerate bitsandbytes datasets huggingface_hub
RUN pip install --no-cache-dir --upgrade --no-deps git+https://github.com/huggingface/transformers.git

WORKDIR /app
COPY train.py .

CMD ["python3", "train.py"]
