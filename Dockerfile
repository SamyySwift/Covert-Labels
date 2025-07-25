FROM python:3.12.2-slim

# Install system dependencies (remove CUDA libraries)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Remove CUDA installation section entirely

WORKDIR /app

# Create models directory for volume mounting
RUN mkdir -p /app/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Declare volume for model persistence
VOLUME ["/app/models"]

EXPOSE 8080

CMD ["python", "app.py"]
