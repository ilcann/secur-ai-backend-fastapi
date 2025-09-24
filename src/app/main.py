from fastapi import APIRouter, FastAPI
from app.ner.routers import ner_router

app = FastAPI()

app.include_router(ner_router, prefix="/ner")