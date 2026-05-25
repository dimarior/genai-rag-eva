"""
main.py
-------
Orquestador principal del pipeline RAG para tickets EVA.

Sigue el mismo patrón del artículo de Medium:
cada paso es independiente y se puede correr solo
o como pipeline completo desde aquí.

Orden de ejecución:
  Paso 1 → ingest.py   : CSV → texto procesado
  Paso 2-5 → rag_chain : chunks → embeddings → vectorstore
  Paso 6-8 → rag_chain : retriever → LLM → respuesta
  Paso 9 → evaluate.py : métricas + MLflow

IMPORTANTE: Antes de correr este archivo,
asegúrate de tener MLflow corriendo en otra terminal:
  mlflow server --host 127.0.0.1 --port 5000
"""

from src.ingest import csv_to_text
from src.rag_chain import build_vectorstore, ask
from src.evaluate import evaluar


def run_pipeline():
    print("\n" + "=" * 60)
    print(" GENAI-RAG-MLOPS-LAB — PIPELINE COMPLETO")
    print("   App de Soporte EVA — Tickets HubSpot")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PASO 1: Preparar documento
    # ------------------------------------------------------------------
    print("\n PASO 1 — Preparando documento desde CSV...")
    csv_to_text()

    # ------------------------------------------------------------------
    # PASOS 2-5: Chunks + Embeddings + Vectorstore
    # ------------------------------------------------------------------
    print("\n PASOS 2-5 — Chunking, embeddings y vectorstore...")
    print("   (Este paso puede tardar 5-15 minutos la primera vez)")
    build_vectorstore()

    # ------------------------------------------------------------------
    # PASOS 6-8: Prueba de consulta completa
    # ------------------------------------------------------------------
    print("\n PASOS 6-8 — Prueba de consulta RAG...")
    print("   Vectorstore creado.")

    # ------------------------------------------------------------------
    # PASO 9: Evaluación
    # ------------------------------------------------------------------
    print("\n PASO 9 — Evaluación disponible desde: python -m src.evaluate")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETO")
    print("")
    print("  Abre los servicios en tu navegador:")
    print("  → MLflow:    http://127.0.0.1:5000")
    print("  → FastAPI:   http://127.0.0.1:8000/docs  (uvicorn)")
    print("  → Streamlit: http://localhost:8501        (streamlit run)")
    print("  → Prometheus:http://localhost:9090        (docker)")
    print("  → Grafana:   http://localhost:3000        (docker)")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
