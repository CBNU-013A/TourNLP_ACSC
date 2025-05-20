# /tests/conftest.py

import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
async def async_client():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

@pytest.fixture(scope="module")
def client():
    return TestClient(app)