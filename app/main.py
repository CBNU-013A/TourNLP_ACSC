# /app/main.py

from fastapi import FastAPI
from app.routes.generate import router as api_router

app = FastAPI()

app.include_router(api_router, prefix="/api/v1/generate", tags=["generate"])

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Template!"}