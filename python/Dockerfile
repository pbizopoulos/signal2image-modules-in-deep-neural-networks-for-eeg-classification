FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
ENV PIP_NO_CACHE_DIR=1
WORKDIR /work
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .
