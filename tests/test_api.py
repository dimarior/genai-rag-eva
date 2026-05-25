"""
tests/test_api.py
-----------------
Tests unitarios para la API FastAPI del RAG EVA.
Equivalente a tests/test_api.py del artículo de Medium.

Correr con:
  pytest
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    """El endpoint raíz debe responder 200 con mensaje."""
    response = client.get("/")
    assert response.status_code == 200
    assert "api" in response.json()
    assert "status" in response.json()


def test_health():
    """El health check debe responder ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint():
    """El endpoint de Prometheus debe responder con texto plano."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_eva_query_count" in response.text


def test_ask_sin_pregunta():
    """Una pregunta vacía debe retornar error de validación."""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Unprocessable Entity
