# /app/routes/generate.py

from fastapi import APIRouter, HTTPException, Query

from app.services.llm_generator import category_manager as cm
from app.schemas.generate_schema import CategorySetRequest, CategorySetResponse
from app.core.logger import logger

router = APIRouter()

# GET get all
@router.get("/categories", response_model=CategorySetResponse)
def get_categories():
    return {"categories": cm.get_all()}

# POST set_all
@router.post("/categories", response_model=CategorySetResponse)
def set_categories(req: CategorySetRequest):
    if req.method != "manual" or not req.categories:
        raise HTTPException(status_code=400, detail="Only manual category setting is supported here")
    cm.set_all(req.categories)
    logger.info(f"💾 카테고리 수동 설정 완료: {req.categories}")
    return {"categories": cm.get_all()}

# PATCH add
@router.patch("/categories", response_model=CategorySetResponse)
def add_categoriy(category: str = Query(...)):
    cm.add(category)
    logger.info(f"💾 카테고리 수동 추가 완료: {category}")
    return {"categories": cm.get_all()}

# DELETE remove
@router.delete("/categories/{name}", response_model=CategorySetResponse)
def remove_category(name: str):
    cm.remove(name)
    logger.info(f"🗑️ {name} 카테고리 삭제 완료")
    return {"categories": cm.get_all()}

# DELETE clear
@router.delete("/categories", response_model=CategorySetResponse)
def clear_categories():
    cm.clear()
    logger.info("🚀 카테고리 삭제 완료")
    return {"categories": cm.get_all()}