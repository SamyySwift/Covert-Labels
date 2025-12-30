#!/usr/bin/env bash
set -euo pipefail

mkdir -p /data /data/reference_images /data/temp_uploads
ln -sf /data/reference_images /app/reference_images
ln -sf /data/temp_uploads /app/temp_uploads

# Persist SQLite DB on the volume
touch /data/product_auth.db || true
ln -sf /data/product_auth.db /app/product_auth.db

chmod -R 755 /data

exec gunicorn -w 2 -t 600 -b 0.0.0.0:8080 product_auth:app