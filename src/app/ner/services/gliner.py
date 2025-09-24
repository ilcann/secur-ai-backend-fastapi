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
import re

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

REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CREDIT_CARD_NUMBER": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "DATE_OF_BIRTH": re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"),  # YYYY-MM-DD
    "PASSPORT_NUMBER": re.compile(r"\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b"),  # örnek pattern
}


class NerGlinerService(NerBaseService):
    def __init__(self):
        self.model = GLiNER.from_pretrained(MODEL_NAME)
        self.labels: List[str] = []

    def extract_entities(self, text: str) -> List[NerEntity]:
        entities: List[NerEntity] = []

        # 1. Regex tabanlı extraction
        for label, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    NerEntity(
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        text=match.group()
                    )
                )

        # 2. Model tabanlı extraction
        raw_entities = self.model.predict_entities(text, threshold=0.5, labels=self.labels)
        model_entities = [NerEntity(**ent) for ent in raw_entities]

        # 3. Birleştir (gerekirse overlap kontrolü yapabilirsin)
        entities = self.merge_entities(entities, model_entities)

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
    
    def merge_entities(self, regex_entities: List[NerEntity], model_entities: List[NerEntity]) -> List[NerEntity]:
        final = regex_entities[:]
        for ent in model_entities:
            if not any(e.start <= ent.start < e.end or ent.start <= e.start < ent.end for e in regex_entities):
                final.append(ent)
        return final
