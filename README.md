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
