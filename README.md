# layout-detector-cloud

A cloud-based document layout detection service that uses YOLOv10 to identify and extract layout elements from PDF documents and images. The service can be deployed as a FastAPI application or GCP cloud run.

## Features

- **Document Layout Detection**: Identifies various document elements like text blocks, tables, images, headers, etc.
- **Multi-format Support**: Processes PDF documents and image files (JPG, PNG)
- **Cloud Deployment**: Ready for deployment on cloud platforms
- **RESTful API**: FastAPI-based endpoint for easy integration
- **GCP cloud run**: deployment on GCP

  ## Project Structure

```
layout-detector-cloud/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

## Docker Deployment

### 1. Build Docker Image
```bash
# Build the image
docker build -t layout-detector:latest .

# Tag for your registry (replace with your registry)
docker tag layout-detector:latest your-registry/layout-detector:latest
```

### 2. Run Locally with Docker
```bash
docker run -p 8080:8080 layout-detector:latest
```

#### Google Cloud Run
```bash
# Tag for Google Container Registry
docker tag layout-detector:latest gcr.io/your-project-id/layout-detector:latest

# Push to GCR
docker push gcr.io/your-project-id/layout-detector:latest

# Deploy to Cloud Run
gcloud run deploy layout-detector \
  --image gcr.io/your-project-id/layout-detector:latest \
  --platform managed \
  --region your-region \
  --allow-unauthenticated
```

## API Usage

### Detect Layout from PDF/Image

**Endpoint**: `POST /detect`

**Request**: Upload a PDF or image file

**Response**: JSON array of page results with detected bounding boxes

```bash
# Example using curl
curl -X POST "http://localhost:8080/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response Format**:
```json
[
  {
    "page_number": 1,
    "width": 2048,
    "height": 1536,
    "boxes": [
      {
        "x1": 100,
        "y1": 200,
        "x2": 500,
        "y2": 300,
        "class_name": "text",
        "confidence": 0.95,
        "rel_x1": 0.0488,
        "rel_y1": 0.1302,
        "rel_x2": 0.2441,
        "rel_y2": 0.1953
      }
    ]
  }
]
```
