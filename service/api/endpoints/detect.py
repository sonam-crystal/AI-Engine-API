from fastapi import APIRouter, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import numpy as np
from service.core.logic.tiny_yolov4.detect2 import object_detector
yolo_router = APIRouter()

@yolo_router.post("/detect")
def detect(file: UploadFile):

    if file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    image = Image.open(BytesIO(file.file.read()))
    image=np.array(image)

    return object_detector(image)
