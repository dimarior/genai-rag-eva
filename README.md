# 🤖 GENAI-RAG-MLOPS-LAB

Sistema RAG (Retrieval-Augmented Generation) para consultas sobre tickets de soporte de la app móvil **EVA** de Recamier, construido con prácticas MLOps.

## Stack tecnológico

| Componente | Herramienta | Puerto |
|---|---|---|
| LLM local | Ollama + llama3.2 | — |
| Embeddings | nomic-embed-text | — |
| Vector DB | ChromaDB | local |
| Pipeline RAG | LangChain | — |
| Tracking | MLflow | :5000 |
| API REST | FastAPI | :8000 |
| Frontend | Streamlit | :8501 |
| Métricas | Prometheus | :9090 |
| Dashboards | Grafana | :3000 |

## Estructura del proyecto

```
GENAI-RAG-MLOPS-LAB/
├── api/main.py              ← FastAPI: /ask /health /metrics
├── app/streamlit_app.py     ← UI para hacer preguntas
├── data/
│   ├── raw/                 ← CSV de tickets HubSpot
│   └── processed/           ← Texto generado
├── docker/prometheus.yml    ← Configuración Prometheus
├── reports/evaluation/      ← Resultados de evaluación
├── src/
│   ├── config.py            ← Configuración central
│   ├── ingest.py            ← CSV → texto (Paso 1)
│   ├── rag_chain.py         ← Pipeline RAG (Pasos 2-8)
│   └── evaluate.py          ← Evaluación (Paso 9)
├── tests/test_api.py        ← Tests unitarios
├── vectorstore/             ← ChromaDB en disco
├── docker-compose.yml
├── main.py                  ← Orquestador principal
└── requirements.txt
```

## Instalación

### 1. Instalar Ollama y modelos

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Copiar el CSV

Coloca el archivo CSV de tickets en `data/raw/`

## Ejecución

### Terminal 1 — MLflow (siempre primero)
```powershell
mlflow server --host 127.0.0.1 --port 5000
```

### Terminal 2 — Pipeline principal
```powershell
.venv\Scripts\activate
python main.py
```

### Terminal 2 — Levantar API (después de main.py)
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Terminal 3 — Frontend
```powershell
streamlit run app/streamlit_app.py
```

### Terminal 4 — Monitoreo
```powershell
docker compose up -d
```

## URLs

- MLflow: http://127.0.0.1:5000
- FastAPI Docs: http://127.0.0.1:8000/docs
- Streamlit: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Tests

```powershell
pytest
```
