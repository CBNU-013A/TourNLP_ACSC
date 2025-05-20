# /app/main.py

from fastapi import FastAPI
# from app.routes import router as api_router

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Template!"}