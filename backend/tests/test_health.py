"""Phase 3 acceptance test: the health endpoint responds with its exact contract."""
from fastapi.testclient import TestClient

from app.main import APP_NAME, APP_VERSION, app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
    }
