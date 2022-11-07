FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
WORKDIR /work
COPY pyproject.toml .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir .
