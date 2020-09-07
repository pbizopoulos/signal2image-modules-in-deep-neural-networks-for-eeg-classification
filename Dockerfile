FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
