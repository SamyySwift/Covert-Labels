BASE_URL = https://covert-labels-aged-dust-9234.fly.dev/

1.POST /api/upload_label

Request (JSON option):
{
"label_id": "string (required)",
"image": "data:image/png;base64,.... (required)",
"metadata": {
"product": "string",
"batch": "string",
"barcode": "string",
"manufacturer": "string",
"production_date": "YYYY-MM-DD",
"expiry_date": "YYYY-MM-DD",
}
}

- Success Response: 200 JSON
  {
  "label_id": "same as input",
  "processed_image_base64": "data:image/png;base64,..."
  }
- Error Responses:
- 400 if label_id/image missing or invalid JSON
- 415 if unsupported file type (multipart)
- 500 for server errors

2. POST /api/train

- Success: 202 JSON
  {
  "status": "started",
  "training_status": {
  "state": "running",
  "started_at": "ISO-8601",
  "finished_at": null,
  "returncode": null,
  "error": null,
  "log": ""
  }
  }
- Already running: 409 JSON { "status":"error", "error":"Training already in progress", "training_status": {...} }

3. GET /api/train-status

- Purpose: Poll training job status.
- Success: 200 JSON
  {
  "status": "ok",
  "training_status": {
  "state": "idle|running|finished|failed",
  "started_at": "ISO-8601|null",
  "finished_at": "ISO-8601|null",
  "returncode": int|null,
  "error": "string|null",
  "log": "stdout/stderr stream so far"
  }
  }

4. POST: /api/verify

Request (JSON option):
{
"image": "data:image/png;base64,.... (required)",
}

Successful Authentication (Status: 200)
{
"status": "authentic",
"predicted_batch": "07250103",
"confidence": 0.6663,
"label_id": "3",
"microdots_detected": true,
"ai_detection_summary": "YES",
"metadata": {
"product": "Example Product",
"batch": "07250103",
"barcode": "1234567890123",
"manufacturer": "Example Manufacturer",
"production_date": "2025-07-25",
"expiry_date": "2026-07-25",
"notes": "Any extra notes here"
}
}

Suspicious — No Microdots Detected (Status: 200)
{
"status": "suspicious",
"reason": "No microdots detected by AI",
"predicted_batch": null,
"confidence": 0.0,
"label_id": null,
"microdots_detected": false,
"ai_detection_summary": "Model did not detect microdot pattern"
}

Suspicious — Low Confidence (Status: 200)
{
"status": "suspicious",
"reason": "Low batch classification confidence",
"predicted_batch": "07250103",
"confidence": 0.42,
"label_id": "3",
"microdots_detected": true,
"ai_detection_summary": "YES",
"metadata": {
// present if a matching upload exists; otherwise null/omitted
}
}

Notes:

- You can retrieve full upload details (and verify what metadata is stored) by calling:
  GET /api/uploads/<label_id>
