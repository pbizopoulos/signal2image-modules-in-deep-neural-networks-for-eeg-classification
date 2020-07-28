FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
