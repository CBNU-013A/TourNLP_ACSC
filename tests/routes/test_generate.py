# /tests/routes/test_generate.py
# integration test for /categories endpoints

import pytest

# TC01: GET /categories
def test_get_categories_initially_empty(client):
    response = client.get("/api/v1/generate/categories")
    assert response.status_code == 200
    assert response.json() == {"categories": []}


# TC02: POST /categories
def test_post_set_categories_manual(client):
    data = {
        "method": "manual",
        "categories": ["숙소", "음식"]
    }
    response = client.post("/api/v1/generate/categories", json=data)
    assert response.status_code == 200
    assert sorted(response.json()["categories"]) == ["숙소", "음식"]

def test_post_set_categories_invalid_method(client):
    data = {
        "method": "llm",  # 현재는 manual만 허용
        "categories": ["숙소"]
    }
    response = client.post("/api/v1/generate/categories", json=data)
    assert response.status_code == 400


# TC03: PATCH /categories
def test_patch_add_category(client):
    response = client.patch("/api/v1/generate/categories", params={"category": "교통"})
    assert response.status_code == 200
    assert "교통" in response.json()["categories"]


# TC04: DELETE /categories/{name}
def test_delete_specific_category(client):
    # 먼저 추가
    client.patch("/api/v1/generate/categories", params={"category": "쇼핑"})
    
    # 삭제
    response = client.delete("/api/v1/generate/categories/쇼핑")
    assert response.status_code == 200
    assert "쇼핑" not in response.json()["categories"]


# TC05: DELETE /categories (전체 삭제)
def test_clear_all_categories(client):
    # 일부 세팅
    client.post("/api/v1/generate/categories", json={"method": "manual", "categories": ["숙소", "음식", "교통"]})
    
    # 전체 삭제
    response = client.delete("/api/v1/generate/categories")
    assert response.status_code == 200
    assert response.json() == {"categories": []}