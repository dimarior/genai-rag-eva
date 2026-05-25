"""
api/main.py
-----------
API REST con soporte para session_id y memoria SQLite.
"""

import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import API_TITLE
from src.rag_chain import ask

app = FastAPI(
    title=API_TITLE,
    description="Sistema RAG para consultas sobre tickets de soporte de la app móvil EVA",
    version="1.0.0",
)

QUERY_COUNT = Counter("rag_eva_query_count",
                      "Total de consultas al RAG")
QUERY_LATENCY = Histogram("rag_eva_query_latency_seconds",
                           "Latencia de respuesta en segundos",
                           buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])
QUERY_ERRORS = Counter("rag_eva_query_errors_total",
                       "Total de errores en consultas")


class QueryRequest(BaseModel):
    pregunta: str
    session_id: str = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "pregunta": "¿Qué errores comunes tiene la app EVA?",
                "session_id": "usuario_123"
            }
        }


@app.get("/", tags=["Estado"])
def root():
    return {"api": API_TITLE, "status": "activa", "version": "1.0.0"}


@app.get("/health", tags=["Estado"])
def health():
    return {
        "status": "ok",
        "api": API_TITLE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/ask", tags=["RAG"])
def query_rag(request: QueryRequest):
    QUERY_COUNT.inc()
    inicio = time.time()
    try:
        respuesta = ask(request.pregunta, session_id=request.session_id)
        latencia = round(time.time() - inicio, 3)
        QUERY_LATENCY.observe(latencia)
        return {
            "pregunta": request.pregunta,
            "respuesta": respuesta,
            "session_id": request.session_id,
            "latencia_segundos": latencia,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        QUERY_ERRORS.inc()
        return {
            "error": str(e),
            "mensaje": "Error procesando la consulta.",
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/metrics", tags=["Monitoreo"])
def metrics():
    return Response(content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST)