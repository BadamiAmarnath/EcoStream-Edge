FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and PyTorch runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose web server port
EXPOSE 5000

# Ensure non-buffered stdout for real-time logging
ENV PYTHONUNBUFFERED=1 \
    ECOSTREAM_HOST=0.0.0.0 \
    ECOSTREAM_PORT=5000

CMD ["python", "app.py"]
