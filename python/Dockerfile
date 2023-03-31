FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
ENV HOME=/usr/src/app/bin
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]
