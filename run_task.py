import boto3
import fitz  # PyMuPDF
import numpy as np
import os
import io
import tempfile
import json
import time
import asyncio
import aiohttp
from PIL import Image
import uuid

from doclayout_yolo import YOLOv10

APPSYNC_URL = os.environ['APPSYNC_API_URL']
APPSYNC_API_KEY = os.environ['APPSYNC_API_KEY']

MUTATION = """
mutation CreateDocumentVisionOutput($input: CreateDocumentVisionOutputInput!) {
  createDocumentVisionOutput(input: $input) {
    documentUuid
    totalPages
    page_number
    width
    height
    box_id
    class_name
    confidence
    bbox
    rel_bbox
    s3_url
  }
}
"""

s3 = boto3.client('s3')
MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"
model = YOLOv10(MODEL_PATH)


def extract_boxes(det_res, width: int, height: int):
    boxes = []
    for box, conf, cls in zip(det_res.boxes.xyxy, det_res.boxes.conf, det_res.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = det_res.names[int(cls)]
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "rel_x1": round(x1 / width, 4),
            "rel_y1": round(y1 / height, 4),
            "rel_x2": round(x2 / width, 4),
            "rel_y2": round(y2 / height, 4),
            "class_name": class_name,
            "confidence": round(float(conf), 4)
        })
    return boxes


def crop_and_save(pil_image, box, output_bucket):
    crop_id = str(uuid.uuid4())
    if box['class_name'] in ('table', 'figure', 'isolate_formula'):
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        cropped = pil_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format='JPEG')
        buf.seek(0)
        s3_key = f"{crop_id}.jpg"
        s3.put_object(Bucket=output_bucket, Key=s3_key, Body=buf, ContentType="image/jpeg")
        s3_url = f"https://{output_bucket}.s3.amazonaws.com/{s3_key}"
    else:
        s3_url = ""
    return s3_url, crop_id


async def call_appsync_mutation(session, input_data):
    # Check if AppSync is configured
    if not APPSYNC_URL or not APPSYNC_API_KEY:
        print("Warning: AppSync not configured, skipping mutation")
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': APPSYNC_API_KEY,
    }
    payload = {
        'query': MUTATION,
        'variables': {'input': input_data}
    }
    
    try:
        async with session.post(APPSYNC_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"AppSync HTTP error {resp.status}: {text}")
            result = await resp.json()
            if 'errors' in result:
                raise Exception(f"AppSync GraphQL errors: {result['errors']}")
            return result['data']['createDocumentVisionOutput']
    except Exception as e:
        print(f"AppSync mutation failed: {e}")
        return None


def process_detections(img, boxes, output_bucket):
    detection_boxes = []
    pil_image = Image.fromarray(img)
    for box in boxes:
        s3_url, crop_id = crop_and_save(pil_image, box, output_bucket)
        bbox_array = [int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])]
        rel_bbox_array = [float(box["rel_x1"]), float(box["rel_y1"]), float(box["rel_x2"]), float(box["rel_y2"])]
        
        detection_box = {
            "box_id": crop_id,
            "class_name": box["class_name"],
            "confidence": float(box["confidence"]),
            "bbox": json.dumps(bbox_array),
            "rel_bbox": json.dumps(rel_bbox_array),
            "s3_url": s3_url
        }
        detection_boxes.append(detection_box)
    return detection_boxes


def create_mutation_input_data(document_uuid, total_pages, page_num, width, height, box):
    return {
        "documentUuid": document_uuid,
        "totalPages": int(total_pages),
        "page_number": int(page_num),
        "width": int(width),
        "height": int(height),
        "box_id": box["box_id"],
        "class_name": box["class_name"],
        "confidence": float(box["confidence"]),
        "bbox": box["bbox"],
        "rel_bbox": box["rel_bbox"],
        "s3_url": box["s3_url"],
    }


async def process_page(session, img, page_num, total_pages, input_key, output_bucket, document_uuid):
    height, width = img.shape[:2]
    det_res = model.predict(img, imgsz=1024, conf=0.3, device="cpu")[0]
    boxes = extract_boxes(det_res, width, height)
    detection_boxes = process_detections(img, boxes, output_bucket)
    
    result = {
        "page_number": page_num,
        "width": width,
        "height": height,
        "detection_boxes": detection_boxes
    }

    tasks = [
        call_appsync_mutation(session, create_mutation_input_data(
            document_uuid, total_pages, page_num, width, height, box
        ))
        for box in detection_boxes
    ]
    
    results_mutations = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result_mutation in enumerate(results_mutations):
        if isinstance(result_mutation, Exception):
            print(f"Error saving box {detection_boxes[i]['box_id']}: {result_mutation}")
        elif result_mutation is None:
            print(f"Skipped saving box {detection_boxes[i]['box_id']} (AppSync not configured)")
        else:
            print(f"Saved box {detection_boxes[i]['box_id']} to AppSync/DynamoDB")
    
    return result


async def process_file_async(input_bucket, input_key, output_bucket, document_uuid):
    obj = s3.get_object(Bucket=input_bucket, Key=input_key)
    file_ext = input_key.lower().split('.')[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(obj['Body'].read())
        tmp_file_path = tmp_file.name

    results = []
    session_timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        if file_ext == "pdf":
            doc = fitz.open(tmp_file_path)
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                result = await process_page(session, img, page_num + 1, total_pages, input_key, output_bucket, document_uuid)
                results.append(result)
        elif file_ext in ("jpg", "jpeg", "png"):
            image = Image.open(tmp_file_path).convert("RGB")
            img = np.array(image)
            result = await process_page(session, img, 1, 1, input_key, output_bucket, document_uuid)
            results.append(result)
        else:
            raise ValueError("Unsupported file type")

    return {"file_name": input_key, "total_pages": len(doc) if file_ext == "pdf" else 1, "results": results}


if __name__ == "__main__":
    input_bucket = os.environ["INPUT_BUCKET"]
    input_key = os.environ["INPUT_KEY"]
    output_bucket = os.environ["OUTPUT_BUCKET"]
    document_uuid = os.environ["DOCUMENT_UUID"]

    start = time.time()
    results = asyncio.run(process_file_async(input_bucket, input_key, output_bucket, document_uuid))

    result_key = input_key.rsplit(".", 1)[0] + "_bbox_result.json"

    print(f"Upload complete. Took {round(time.time() - start, 2)}s")
    print("🎉 Task completed, exiting container.")
