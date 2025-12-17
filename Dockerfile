FROM python:3.12.2-slim

# Install only the libs needed for headless OpenCV/TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

# Only copy needed app files; destination must be a directory
COPY flask_app.py train_autoencoder.py docker-entrypoint.sh autoencoder_genuine.keras   ./
COPY autoencoder_outputs/ ./autoencoder_outputs/
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "flask_app:app"]