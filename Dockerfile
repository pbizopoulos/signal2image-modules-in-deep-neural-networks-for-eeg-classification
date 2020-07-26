FROM pytorch/pytorch
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
