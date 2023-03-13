from pydantic import BaseModel

class Detail(BaseModel):
    id: int
    image_url: str
    expectedReading: str
    actualReading: str
