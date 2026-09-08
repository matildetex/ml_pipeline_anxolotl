FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.1.2 com CUDA 12.1 (driver 535 suporta CUDA 12.x)
RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
