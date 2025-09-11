from pydantic import BaseModel
from typing import List
from .entity import NerEntity

class ExtractResponse(BaseModel):
    entities: List[NerEntity]
