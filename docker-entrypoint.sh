#!/usr/bin/env bash
set -euo pipefail

# Ensure volume structure
mkdir -p /data /data/uploads /data/autoencoder_outputs
mkdir -p /app/autoencoder_outputs

# Link model and thresholds from volume into app paths expected by flask_app.py
if [ -f /data/autoencoder_genuine.keras ]; then
  ln -sf /data/autoencoder_genuine.keras /app/autoencoder_genuine.keras
fi
if [ -f /data/autoencoder_outputs/anomaly_threshold.txt ]; then
  ln -sf /data/autoencoder_outputs/anomaly_threshold.txt /app/autoencoder_outputs/anomaly_threshold.txt
fi
if [ -f /data/autoencoder_outputs/anomaly_threshold_patch.txt ]; then
  ln -sf /data/autoencoder_outputs/anomaly_threshold_patch.txt /app/autoencoder_outputs/anomaly_threshold_patch.txt
fi

# Permissions for volume
chmod -R 755 /data

# Start the app
exec gunicorn -w 1 -t 300 -b 0.0.0.0:8080 flask_app:app