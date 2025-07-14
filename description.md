## API Endpoints

BASE_URL = https://covert-labels-aged-dust-9234.fly.dev

### 1. Health Check

GET /api/health

Description: Check if the API server and ML model are running properly.

Request: No parameters required

Response:
{
"status": "healthy",
"model_loaded": true,
"timestamp": "2024-01-09T10:30:00.000Z"
}

### 2. Upload Labels (Add Authentication Patterns)

POST /api/upload-labels

Description: Upload product label images and embed invisible authentication patterns (watermarks, microdots) for anti-counterfeiting.

Request:

- Content-Type: multipart/form-data
- Form Fields:

  - images (files, required): Multiple image files to process
  - batch_id (string, required): Unique identifier for this batch of products
  - product_name (string, optional): Name of the product (default: "unknown")
  - sku (string, optional): SKU identifier (default: "unknown")
    Response:
    {
    "message": "Successfully uploaded 4 images to batch test_batch_123",
    "batch_id": "test_batch_123",
    "uploaded_files": 4,
    "batch_path": "dataset/authentic/test_batch_123",
    "modified_images": [
    {
    "filename": "product_label_1.jpg",
    "modified_image": "base64_encoded_image_data..."
    }
    ]
    }
    Error Responses:

- 400 : Missing images or batch_id
- 500 : Processing error

### 3. Trigger Model Training

POST /api/train

Description: Start training the authentication model with new data.

Request: No parameters required

Response:
{
"message": "Training started",
"status": "training",
"timestamp": "2024-01-09T10:30:00.000Z"
}
Error Responses:

- 400 : Training already in progress
- 500 : Failed to start training

### 4. Training Status

GET /api/training-status

Description: Get the current status of model training.

Request: No parameters required

Response:
{
"status": "training",
"progress": 45,
"message": "Training epoch 3/10",
"timestamp": "2024-01-09T10:30:00.000Z"
}
Status Values:

- "idle" : No training in progress
- "training" : Training is running
- "completed" : Training finished successfully
- "error" : Training failed

### 5. Verify Labels

POST /api/verify

Description: Verify if uploaded images are authentic by detecting embedded authentication patterns.

Request:

- Content-Type: multipart/form-data
- Form Fields:
  - images (files, required): Image files to verify

Response:
{
"overall_authenticity": "authentic",
"confidence_score": 0.85,
"batch_match": "test_batch_123",
"individual_results": [
{
"filename": "scanned_label.jpg",
"authenticity": "authentic",
"confidence": 0.87,
"batch_id": "test_batch_123",
"product_info": {
"product_name": "Sartor Face Cleanser",
"sku": "SFC500"
}
}
]
}

Error Responses:

- 400: No images provided
- 500: Verification failed
