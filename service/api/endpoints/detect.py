from fastapi import APIRouter, UploadFile, HTTPException
from io import BytesIO
from typing import Optional
from PIL import Image
import numpy as np
from service.core.logic.tiny_yolov4.detect2 import object_detector
import service.core.config.db as _db
from service.core.models.detail import Detail
# from service.core.schemas.detail import Detail 
yolo_router = APIRouter()

_db.create_database()

session = _db.Session()

@yolo_router.post("/detect")
def detect(file: UploadFile, expected_reading: Optional[str] = None):

    if file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    image = Image.open(BytesIO(file.file.read()))
    image=np.array(image)

    output = object_detector(image)
    actual_reading = str(output["reading"]) + str(output["annotation"])

    # create a new record and add it to the session
    new_detection = Detail(image_url=file.filename,expectedReading=expected_reading,actualReading=actual_reading)
    session.add(new_detection)

    # commit the changes to the database
    session.commit()
    
    return output
