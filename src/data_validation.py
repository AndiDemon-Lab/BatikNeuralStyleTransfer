from pydantic import BaseModel

class TrainRequest(BaseModel):
    content_image_path: str
    style_image_path: str