FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
