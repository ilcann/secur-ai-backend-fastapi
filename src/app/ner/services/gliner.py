from typing import List, Optional
from fastapi import HTTPException
from app.ner.schemas.entityLabel import EntityLabelDto
from gliner import GLiNER
from app.ner.schemas import NerEntity
from app.ner.services import NerBaseService
import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=env_path)
NESTJS_HOST = os.getenv("NESTJS_HOST", "localhost")
NESTJS_PORT = os.getenv("NESTJS_PORT", "3002")

MODEL_NAME = "urchade/gliner_large-v2.1"

LABELS = [
    # 👤 Kişisel PII
    "person", "organization", "phone number", "address", "passport number",
    "email", "credit card number", "social security number", "health insurance id number",
    "date of birth", "mobile phone number", "bank account number", "medication",
    "cpf", "driver's license number", "tax identification number", "medical condition",
    "identity card number", "national id number", "ip address", "email address",
    "iban", "credit card expiration date", "username", "health insurance number",
    "registration number", "student id number", "insurance number", "flight number",
    "landline phone number", "blood type", "cvv", "reservation number", "digital signature",
    "social media handle", "license plate number", "cnpj", "postal code", "passport_number",
    "serial number", "vehicle registration number", "credit card brand", "fax number",
    "visa number", "insurance company", "identity document number", "transaction number",
    "national health insurance number", "cvc", "birth certificate number", "train ticket number",
    "passport expiration date", "social_security_number", "company"
]


class NerGlinerService(NerBaseService):
    def __init__(self):
        self.model = GLiNER.from_pretrained(MODEL_NAME)
        self.labels: List[str] = []

    def extract_entities(self, text: str) -> List[NerEntity]:
        raw_entities = self.model.predict_entities(text, threshold=0.5, labels=self.labels)
        entities = [NerEntity(**ent) for ent in raw_entities]
        return entities
    
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
    