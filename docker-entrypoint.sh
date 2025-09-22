#!/usr/bin/env bash
set -euo pipefail

# Ensure volume structure
mkdir -p /data /data/uploads

# Seed model + maps into the volume if they don't exist yet
[ -f /data/batch_classifier_model.keras ] || cp -f /app/batch_classifier_model.keras /data/batch_classifier_model.keras
[ -f /data/batch_label_map.json ] || cp -f /app/batch_label_map.json /data/batch_label_map.json
[ -f /data/batch_metadata_map.json ] || cp -f /app/batch_metadata_map.json /data/batch_metadata_map.json

# Keep your current code working by linking files in /app to the /data copies
ln -sf /data/batch_classifier_model.keras /app/batch_classifier_model.keras
ln -sf /data/batch_label_map.json /app/batch_label_map.json
ln -sf /data/batch_metadata_map.json /app/batch_metadata_map.json

# The DB is created by the app on first use via init_db() at DB_PATH
# Ensure uploads dir exists (already created above)
chmod -R 755 /data

# Start the app
exec gunicorn -w 1 -b 0.0.0.0:8080 flask_app:app