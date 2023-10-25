FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
