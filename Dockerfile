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
    libgtk-3-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy needed app files; destination must be a directory
COPY flask_app.py train_autoencoder.py docker-entrypoint.sh ./

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "flask_app:app"]
