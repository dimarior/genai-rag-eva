# GenAI RAG EVA - Recamier

Sistema RAG (Retrieval-Augmented Generation) para consultas sobre tickets de soporte de la app movil EVA de Recamier, construido con practicas MLOps.

## Stack tecnologico

| Componente | Herramienta | Puerto |
|---|---|---|
| LLM API | Mistral Small Latest | nube |
| LLM local (respaldo) | Ollama + llama3.2 | local |
| Embeddings | nomic-embed-text | local |
| Vector DB | ChromaDB | local |
| Memoria persistente | SQLite | local |
| Pipeline RAG | LangChain | — |
| Tracking | MLflow Genai | :5000 |
| API REST | FastAPI | :8080 |
| Frontend | Streamlit | :8501 |
| Metricas | Prometheus | :9090 |
| Dashboards | Grafana | :3000 |

## Estructura del proyecto

```
genai-rag-eva/
├── api/main.py              <- FastAPI: /ask /health /metrics
├── app/
│   ├── streamlit_app.py     <- UI con sidebar de conversaciones
│   └── assets/              <- Logos e imagenes corporativas
├── data/
│   ├── raw/                 <- CSV de tickets HubSpot
│   └── processed/           <- Texto generado
├── docker/prometheus.yml    <- Configuracion Prometheus
├── reports/evaluation/      <- Resultados de evaluacion RAG
├── src/
│   ├── config.py            <- Configuracion central
│   ├── ingest.py            <- CSV a texto (Paso 1)
│   ├── rag_chain.py         <- Pipeline RAG (Pasos 2-8)
│   ├── memory.py            <- Memoria persistente SQLite
│   └── evaluate.py          <- Evaluacion (Paso 9)
├── tests/test_api.py        <- Tests unitarios
├── vectorstore/             <- ChromaDB en disco
├── eva_memory.db            <- Base de datos SQLite (memoria)
├── docker-compose.yml
├── main.py                  <- Orquestador principal
└── requirements.txt
```

## Requisitos previos

- Python 3.11
- uv (gestor de entornos virtuales)
- Git
- Ollama
- Docker Desktop (opcional, para Prometheus y Grafana)

## Instalacion

### 1. Instalar Ollama y modelos

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Clonar el repositorio

```powershell
git clone https://github.com/dimarior/genai-rag-eva.git
cd genai-rag-eva
```

### 3. Crear carpetas necesarias

```powershell
mkdir data\raw
mkdir data\processed
mkdir vectorstore
mkdir reports\evaluation
```

### 4. Crear entorno virtual e instalar dependencias

```powershell
uv python install 3.11
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 5. Configurar API keys

Abre `src/config.py` y configura:

```python
MISTRAL_API_KEY = "tu_api_key_de_mistral"
MISTRAL_MODEL   = "mistral-small-latest"
```

Obtener API key gratuita en: https://console.mistral.ai
El tier gratuito incluye 1 billon de tokens por mes sin tarjeta de credito.

### 6. Copiar el CSV

Coloca el archivo CSV de tickets en:
```
data/raw/hubspot-crm-exports-r-tickets-eva_eva4-2026-05-07.csv
```

## Ejecucion

### Terminal 1 — MLflow (siempre primero)
```powershell
.venv\Scripts\activate
mlflow server --host 127.0.0.1 --port 5000
```

### Terminal 2 — Ollama
```powershell
ollama serve
```

### Terminal 3 — Pipeline principal
```powershell
.venv\Scripts\activate
python main.py
```

Esperar el mensaje: PIPELINE COMPLETO

### Terminal 3 — Levantar API (despues de main.py)
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
```

### Terminal 4 — Frontend
```powershell
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```

### Terminal 5 — Monitoreo (opcional, requiere Docker)
```powershell
docker compose up -d
```

## URLs

- MLflow: http://127.0.0.1:5000
- FastAPI Docs: http://127.0.0.1:8080/docs
- Streamlit: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Caracteristicas principales

- Pipeline RAG completo con trazabilidad MLflow Genai
- Memoria persistente de conversaciones con SQLite
- Sidebar con historial de conversaciones por sesion
- Soporte para Mistral API (nube) y Ollama (local)
- Diseno corporativo Recamier con logos y colores oficiales
- Anti-alucinaciones configuradas (temperature, top_k, top_p)
- Prompt especializado en soporte tecnico EVA

## Tests

```powershell
pytest
```

## Notas importantes

- El vectorstore se genera automaticamente al correr main.py
- No cerrar procesos con Task Manager — usar Ctrl+C para evitar corrupcion de ChromaDB
- La memoria SQLite se guarda en eva_memory.db en la raiz del proyecto
- MLflow debe estar corriendo antes de ejecutar cualquier script
