FROM python:3.12.2-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository and install CUDA libraries (per quickstart guidance)
RUN wget -qO /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cuda-nvcc-12-2 \
      libcublas-12-2 \
      libcudnn8 \
    && rm -rf /var/lib/apt/lists/* && rm /cuda-keyring.deb

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Add entrypoint
RUN chmod +x /app/docker-entrypoint.sh
CMD ["/app/docker-entrypoint.sh"]

EXPOSE 8080

# Use a single worker to avoid TensorFlow model loading issues
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "flask_app:app"]
