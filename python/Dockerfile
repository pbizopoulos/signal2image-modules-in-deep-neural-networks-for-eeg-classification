FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt
