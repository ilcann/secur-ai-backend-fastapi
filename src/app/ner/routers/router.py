from fastapi import APIRouter
from app.ner.schemas import ExtractRequest, ExtractResponse
from app.ner.services import NerBaseService, NerGlinerService

ner_router = APIRouter()

ner_service: NerBaseService = NerGlinerService()

@ner_router.post("/extract", response_model=ExtractResponse)
def extract_entities(request: ExtractRequest):
    ner_entities = ner_service.extract_entities(request.text)
    print("Ner Entities:", ner_entities)
    return ExtractResponse(entities=ner_entities)

@ner_router.post("/labels/sync")
async def sync_labels():
    labels = await ner_service.fetch_labels()
    keys = ner_service.extract_keys(labels)
    ner_service.update_labels(keys)
    return {"status": "ok", "synced_labels": keys}

@ner_router.get('/labels')
def get_labels():
    return ner_service.get_labels()