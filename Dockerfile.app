FROM python
COPY app-requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel && python3 -m pip install --no-cache-dir -r app-requirements.txt
