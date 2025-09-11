from pydantic import BaseModel

class ExtractRequest(BaseModel):
    text: str