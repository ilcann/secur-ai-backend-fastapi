from typing import Optional
from pydantic import BaseModel
class EntityLabelDto(BaseModel):
    key:str
    name:str
    description: Optional[str] = None