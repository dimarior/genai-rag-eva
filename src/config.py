import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Carga variables desde .env (nunca se sube a git)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
REPORTS_DIR = ROOT_DIR / "reports"
EVALUATION_DIR = REPORTS_DIR / "evaluation"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
API_TITLE = "GenAI RAG - Soporte EVA Recamier"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "rag-eva-tickets"

# Parámetros LLM
LLM_TEMPERATURE = 0.1
LLM_TOP_K = 10
LLM_TOP_P = 0.9
RETRIEVER_K = 4

# --- Mistral (generación) ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# --- Embeddings: sentence-transformers---
# Corre embebido en el proceso Python (sin API key, sin servidor externo tipo
# Ollama), lo que permite desplegar en Streamlit Community Cloud sin fricción.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
)

# --- Multimodal ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Validación temprana: falla rápido y claro si falta la key, en vez de fallar
# a mitad de una consulta con un error críptico de la librería.
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "Falta MISTRAL_API_KEY. Crea un archivo .env en la raíz del proyecto "
        "con MISTRAL_API_KEY=tu_key."
    )

# Crear carpetas si no existen al importar config
for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR, EVALUATION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)