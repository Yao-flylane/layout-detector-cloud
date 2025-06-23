from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz  # PyMuPDF
import numpy as np
import os
from doclayout_yolo import YOLOv10
import tempfile
from typing import List
from pydantic import BaseModel
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model from local (already inside Docker image)
MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"
model = YOLOv10(MODEL_PATH)

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    rel_x1: float
    rel_y1: float
    rel_x2: float
    rel_y2: float

class PageResult(BaseModel):
    page_number: int
    width: int
    height: int
    boxes: List[BBox]

def extract_boxes(det_res, width: int, height: int) -> List[BBox]:
    """Helper to extract structured BBox from detection result"""
    boxes = []
    for box, conf, cls in zip(det_res.boxes.xyxy, det_res.boxes.conf, det_res.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = det_res.names[int(cls)]
        boxes.append(BBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            rel_x1=round(x1 / width, 4),
            rel_y1=round(y1 / height, 4),
            rel_x2=round(x2 / width, 4),
            rel_y2=round(y2 / height, 4),
            class_name=class_name,
            confidence=round(float(conf), 4)
        ))
    return boxes

@app.post("/detect", response_model=List[PageResult])
async def detect_layout(file: UploadFile = File(...)):
    filename = file.filename.lower()
    results = []
    logger.info(f"Received file: {filename}")

    if filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(await file.read())
            pdf_path = tmp_pdf.name

        doc = fitz.open(pdf_path)
        logger.info(f"Processing PDF with {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            height, width = img.shape[:2]

            logger.info(f"Running model on page {page_num + 1}")
            det_res = model.predict(img, imgsz=1024, conf=0.3, device="cpu")[0]

            boxes = extract_boxes(det_res, width, height)
            results.append(PageResult(page_number=page_num + 1, width=width, height=height, boxes=boxes))

        os.remove(pdf_path)

    elif filename.endswith((".jpg", ".jpeg", ".png")):
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = np.array(image)
        height, width = img.shape[:2]

        logger.info("Running model on image")
        det_res = model.predict(img, imgsz=1024, conf=0.3, device="cpu")[0]

        boxes = extract_boxes(det_res, width, height)
        results.append(PageResult(page_number=1, width=width, height=height, boxes=boxes))

    else:
        logger.error("Unsupported file format")
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, JPG, PNG are accepted.")

    logger.info("Detection completed")
    return results
