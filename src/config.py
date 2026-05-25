from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
REPORTS_DIR = ROOT_DIR / "reports"
EVALUATION_DIR = REPORTS_DIR / "evaluation"

OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
API_TITLE = "GenAI RAG - Soporte EVA Recamier"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "rag-eva-tickets"

# Parámetros LLM 
LLM_TEMPERATURE = 0.1      # Precisión
LLM_TOP_K = 10             # Solo considera los 10 tokens más probables
LLM_TOP_P = 0.9            # Probabilidad acumulada máxima
RETRIEVER_K = 4            # Tickets similares a recuperar

# Crear carpetas si no existen al importar config
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR, EVALUATION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
