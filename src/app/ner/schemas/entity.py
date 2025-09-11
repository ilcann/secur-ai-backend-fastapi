from pydantic import BaseModel

class NerEntity(BaseModel):
    text: str
    label: str
    start: int | None = None
    end: int | None = None
    score: float | None = None