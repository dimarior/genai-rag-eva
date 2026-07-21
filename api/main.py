"""
api/main.py
-----------
API REST con soporte para session_id, memoria SQLite y entrada multimodal
(audio, imagen, PDF).
"""

import time
from datetime import datetime
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import API_TITLE
from src.rag_chain import ask

app = FastAPI(
    title=API_TITLE,
    description="Sistema RAG para consultas sobre tickets de soporte de Recamier",
    version="1.1.0",
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


class MultimodalResponse(BaseModel):
    respuesta: str
    session_id: str
    latencia_segundos: float
    modalidad: str
    texto_extraido: str = ""
    timestamp: str


@app.get("/", tags=["Estado"])
def root():
    return {"api": API_TITLE, "status": "activa", "version": "1.1.0"}


@app.get("/health", tags=["Estado"])
def health():
    return {
        "status": "ok",
        "api": API_TITLE,
        "multimodal": {"audio": True, "imagen": True, "documento": True},
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


# ── Endpoint de audio ─────────────────────────────────────────────────────────

@app.post("/ask/audio", response_model=MultimodalResponse, tags=["Multimodal"])
async def ask_audio_endpoint(
    audio: UploadFile = File(..., description="Archivo de audio (wav, mp3, ogg, m4a)"),
    session_id: str = Form(default="default"),
):
    """Transcribe el audio con faster-whisper y lo procesa como una consulta normal."""
    from src.multimodal.audio import transcribe_audio

    QUERY_COUNT.inc()
    t0 = time.time()
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")

    transcripcion = transcribe_audio(audio_bytes, filename=audio.filename or "audio.wav")

    if not transcripcion["success"]:
        QUERY_ERRORS.inc()
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo transcribir el audio: {transcripcion['error']}"
        )

    texto = transcripcion["text"]

    try:
        respuesta = ask(texto, session_id=session_id)
    except Exception as e:
        QUERY_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latencia = round(time.time() - t0, 3)
    QUERY_LATENCY.observe(latencia)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        modalidad="audio",
        texto_extraido=texto,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Endpoint de imagen ────────────────────────────────────────────────────────

@app.post("/ask/image", response_model=MultimodalResponse, tags=["Multimodal"])
async def ask_image_endpoint(
    imagen: UploadFile = File(..., description="Imagen (jpg, png, bmp, webp) - ej. captura de error, factura"),
    session_id: str = Form(default="default"),
    pregunta_adicional: str = Form(default=""),
):
    """Extrae texto de la imagen con EasyOCR y lo procesa como una consulta normal."""
    from src.multimodal.image import extract_text_from_image, describe_image_context

    QUERY_COUNT.inc()
    t0 = time.time()
    imagen_bytes = await imagen.read()

    if not imagen_bytes:
        raise HTTPException(status_code=400, detail="Archivo de imagen vacío.")

    if len(imagen_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el límite de 5 MB. Usa una imagen más pequeña."
        )

    ocr_result = extract_text_from_image(imagen_bytes, filename=imagen.filename or "image.jpg")
    texto_ocr = ocr_result["text"] if ocr_result["success"] else ""
    contexto_imagen = describe_image_context(texto_ocr, filename=imagen.filename or "")

    pregunta_final = (
        f"{contexto_imagen}\n\nPregunta del usuario: {pregunta_adicional}"
        if pregunta_adicional.strip() else contexto_imagen
    )

    try:
        respuesta = ask(pregunta_final, session_id=session_id)
    except Exception as e:
        QUERY_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latencia = round(time.time() - t0, 3)
    QUERY_LATENCY.observe(latencia)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        modalidad="imagen",
        texto_extraido=texto_ocr,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Endpoint de documento PDF ─────────────────────────────────────────────────

@app.post("/ask/document", response_model=MultimodalResponse, tags=["Multimodal"])
async def ask_document_endpoint(
    documento: UploadFile = File(..., description="Documento PDF (factura, contrato, comprobante)"),
    session_id: str = Form(default="default"),
    pregunta_adicional: str = Form(default=""),
):
    """Extrae texto del PDF (PyMuPDF + fallback OCR) y lo procesa como una consulta normal."""
    from src.multimodal.document import extract_text_from_pdf, describe_document_context

    QUERY_COUNT.inc()
    t0 = time.time()
    pdf_bytes = await documento.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo PDF vacío.")

    if len(pdf_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="El PDF supera el límite de 5 MB. Comprime el documento e intenta de nuevo."
        )

    pdf_result = extract_text_from_pdf(pdf_bytes, filename=documento.filename or "document.pdf")

    if not pdf_result["success"]:
        QUERY_ERRORS.inc()
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo procesar el PDF: {pdf_result['error']}"
        )

    texto_pdf = pdf_result["text"]
    contexto_doc = describe_document_context(
        texto_pdf,
        filename=documento.filename or "",
        ocr_used=pdf_result.get("ocr_used", False),
    )

    pregunta_final = (
        f"{contexto_doc}\n\nPregunta del usuario: {pregunta_adicional}"
        if pregunta_adicional.strip() else contexto_doc
    )

    try:
        respuesta = ask(pregunta_final, session_id=session_id)
    except Exception as e:
        QUERY_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

    latencia = round(time.time() - t0, 3)
    QUERY_LATENCY.observe(latencia)
    return MultimodalResponse(
        respuesta=respuesta,
        session_id=session_id,
        latencia_segundos=latencia,
        modalidad="documento",
        texto_extraido=texto_pdf[:500],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", tags=["Monitoreo"])
def metrics():
    return Response(content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST)