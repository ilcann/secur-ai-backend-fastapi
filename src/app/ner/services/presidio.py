from app.ner.schemas.entity import NerEntity
from presidio_analyzer import AnalyzerEngine
from typing import List
from app.ner.services import NerBaseService
from typing import List, Optional
from fastapi import HTTPException
from app.ner.schemas.entityLabel import EntityLabelDto
from app.ner.services import NerBaseService
import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=env_path)
NESTJS_HOST = os.getenv("NESTJS_HOST", "localhost")
NESTJS_PORT = os.getenv("NESTJS_PORT", "3002")

class NerPresidioService(NerBaseService):
    def __init__(self):
        self.model = AnalyzerEngine()
        self.labels: List[str] = []
    def update_labels(self, labels: List[str]):
        self.labels = labels
    
    def get_labels(self) -> List[str]:
        return self.labels

    async def fetch_labels(self) -> List[EntityLabelDto]:
        try:
            async with httpx.AsyncClient() as client:
                url = f"http://{NESTJS_HOST}:{NESTJS_PORT}/labels"
                res = await client.get(url)
                res.raise_for_status()
                response_json = res.json()
                labels_data = response_json.get("data", {}).get("labels", [])
                return [EntityLabelDto(**l) for l in labels_data]
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"Failed to fetch labels: {err}")
    
    def extract_keys(self, labels: Optional[List[EntityLabelDto]]) -> List[str]:
        if labels is None:
            return []
        return [label.key for label in labels]
    def extract_entities(self, text: str) -> List[NerEntity]:
        raw_entities = self.model.analyze(text, entities=self.labels)
        entities = [NerEntity(**ent) for ent in raw_entities]
        return entities