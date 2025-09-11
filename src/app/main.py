from fastapi import APIRouter, FastAPI
from app.ner.routers import ner_router

app = FastAPI(
    docs_url="/fastapi/docs",
    redoc_url="/fastapi/redoc",
    openapi_url="/fastapi/openapi.json",
)

fastapi_router = APIRouter(prefix="/fastapi")
fastapi_router.include_router(ner_router, prefix="/ner")

app.include_router(fastapi_router)