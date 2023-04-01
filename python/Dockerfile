FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ENV HOME=/usr/src/app/bin
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
