"""
rag_chain.py
------------
Pipeline RAG completo con MLflow Genai Tracing y memoria SQLite.
Cubre TODAS las categorías de soporte de Recamier (Aplicaciones, Conectividad,
Admin Usuarios, Backup, Hardware, etc.) y todas las filiales (Recamier,
Dermodis, Lansey, Keramer, Arte Francés, Fondelar) — no solo EVA.

Pasos:
  1. load_vectorstore   — carga ChromaDB desde disco
  2. format_documents   — formatea chunks recuperados (con su metadata)
  3. retrieve_documents — busca tickets similares
  4. generate_answer    — genera respuesta con historial (Mistral)
  5. ask                — función principal con memoria

Embeddings: sentence-transformers (local, sin API key ni servidor externo,
corre embebido en el proceso Python, sin API key ni servidor externo).
"""

import json
import mlflow
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    PROCESSED_DATA_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    EMBEDDING_MODEL_NAME,
)
from src.memory import (
    save_message,
    get_history,
    format_history_for_prompt,
)

# ---------------------------------------------------------------------------
# Configurar MLflow
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

PROMPT_TEMPLATE = [
    ("system", """Eres EVA, asistente virtual de soporte técnico interno
de Recamier S.A. y sus filiales (Recamier, Dermodis, Lansey, Keramer,
Arte Francés, Fondelar).

COMPORTAMIENTO CONVERSACIONAL:
- Si el usuario saluda sin dar su nombre, responde cordialmente
  y preséntate brevemente:
  "Buenos días/tardes, soy EVA, asistente virtual de
  soporte de Recamier. ¿En qué puedo ayudarle?"
- NO preguntes el nombre proactivamente — espera a que el usuario
  lo comparta voluntariamente.
- Si el usuario comparte su nombre, acúsalo de recibo cordialmente
  y úsalo en las respuestas siguientes.
- Si preguntan "¿cómo me llamo?" y conoces el nombre del historial,
  responde directamente sin buscar en tickets:
  "Usted se llama [nombre]."
- Para preguntas personales o sociales que no son de soporte,
  responde brevemente y redirige al tema de soporte técnico.
- Si el usuario envía una imagen, audio o documento PDF, el contenido
  ya viene transcrito/extraído en el mensaje. Trátalo como una
  consulta normal, usando ese texto como la pregunta o el contexto.

ROL:
Experto en soporte que conoce todos los tickets históricos de Recamier,
cubriendo aplicaciones (IBES, EVA, ONBASE, OUTLOOK, SNAP, BI4WEB, TEAMS,
E-commerce, etc.), conectividad y VPN, administración de usuarios,
backups, hardware y software de PC, equipos móviles, impresión y
telefonía. Ayudas a agentes de soporte y analistas a encontrar
soluciones basadas en casos reales documentados en el sistema.

CÓMO USAR EL CONTEXTO:
- Cada ticket recuperado incluye su Categoría, Subcategoría y Filial.
  Usa SIEMPRE esos campos para confirmar que la solución corresponde
  exactamente al sistema/aplicación que el usuario está consultando.
- Si hay tickets de varias categorías distintas en el contexto y solo
  una corresponde a la consulta del usuario, ignora las demás y usa
  solo la relevante — no mezcles soluciones de sistemas diferentes.
- Si la filial del ticket recuperado es distinta a la que menciona el
  usuario, acláralo antes de dar la solución (los procesos pueden
  variar entre Recamier, Dermodis, Lansey, etc.).

TONO Y ESTILO:
- Usa un tono profesional, cordial y corporativo
- Antes de dar la solución valida con frases como:
  "De acuerdo con los registros de soporte..."
  "Con base en la información disponible en el sistema..."
  "Según los casos documentados..."
- Trata al usuario de "usted" siempre
- Si conoces el nombre del usuario úsalo naturalmente
- Cierra siempre con una acción concreta o recomendación

INSTRUCCIONES:
1. Responde SIEMPRE en español
2. Si el contexto tiene información relevante y de la categoría correcta,
   úsala para responder
3. Si NO hay información suficiente o de la categoría correcta en los
   tickets responde exactamente:
   "De acuerdo con nuestra base de conocimiento, no encontré registros
   relacionados con su consulta. Le sugiero contactar directamente al
   equipo de soporte técnico para una atención personalizada."
4. Máximo 3 párrafos — sé conciso y directo
5. Si hay solución clara en los tickets ponla primero

PLANTILLA DE RESPUESTA cuando hay información:
**Situación identificada:**
[Describe el problema basado en los tickets, mencionando la categoría/aplicación]

**Solución documentada:**
[Pasos exactos según los tickets — numerados]

**Recomendación:**
[Acción concreta para prevenir o escalar]

LO QUE NO DEBES HACER:
- No inventes soluciones que no estén en los tickets
- No respondas en inglés
- No mezcles soluciones de categorías/aplicaciones distintas
- No ignores el nombre del usuario si lo conoces"""),

    ("human", """{history}

Contexto de tickets similares de Recamier:
{context}

Consulta: {question}

Respuesta:"""),
]


def _get_embeddings():
    """Embeddings locales con sentence-transformers.
    Corre embebido en el proceso Python, sin API key ni servidor externo,
    lo que permite desplegar en Streamlit Community Cloud sin fricción."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Build vectorstore
# ---------------------------------------------------------------------------

def _cargar_documentos_jsonl() -> list[Document]:
    """Lee tickets_recamier.jsonl (generado por src/ingest.py) y arma
    objetos Document de LangChain con su metadata (categoría, subcategoría,
    filial, etc.) para poder filtrar/inspeccionar después."""
    jsonl_path = PROCESSED_DATA_DIR / "tickets_recamier.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            "No se encontró tickets_recamier.jsonl. "
            "Ejecuta primero: python -m src.ingest"
        )

    documentos = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for linea in f:
            registro = json.loads(linea)
            documentos.append(
                Document(page_content=registro["text"], metadata=registro["metadata"])
            )
    return documentos


def build_vectorstore():
    """Crea chunks (preservando metadata), embeddings y guarda en ChromaDB."""
    with mlflow.start_run(run_name="build_vectorstore"):
        documentos = _cargar_documentos_jsonl()
        print(f"Tickets cargados: {len(documentos)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "],
        )
        # split_documents conserva la metadata de cada Document en sus chunks
        chunks = splitter.split_documents(documentos)
        print(f"Chunks generados: {len(chunks)}")

        mlflow.log_param("modelo_embeddings", EMBEDDING_MODEL_NAME)
        mlflow.log_param("modelo_llm", MISTRAL_MODEL)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_metric("total_tickets", len(documentos))
        mlflow.log_metric("total_chunks", len(chunks))

        print(f"Creando embeddings con sentence-transformers ({EMBEDDING_MODEL_NAME})... (puede tardar varios minutos con 7000+ tickets)")
        embeddings = _get_embeddings()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

        mlflow.log_metric("vectorstore_size", vectorstore._collection.count())
        print(f"Vectorstore guardado en: {VECTORSTORE_DIR}")
        print(f"Total vectores en ChromaDB: {vectorstore._collection.count()}")

    return None


def update_vectorstore():
    """
    Actualización INCREMENTAL: agrega al vectorstore existente solo los
    tickets que todavía no están indexados (por ticket_id), sin borrar ni
    reconstruir lo que ya había.

    Úsalo cuando llega un archivo nuevo (otro año, u otra área) que
    quieres sumar al conocimiento del RAG sin repetir el proceso completo.
    Requiere que ya exista un vectorstore (corre build_vectorstore() la
    primera vez).
    """
    with mlflow.start_run(run_name="update_vectorstore"):
        if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
            raise FileNotFoundError(
                "No existe un vectorstore todavía. Corre build_vectorstore() "
                "(python -m src.rag_chain) la primera vez."
            )

        embeddings = _get_embeddings()
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )

        # ticket_id de todo lo que YA está indexado, para no duplicar
        existentes = vectorstore.get(include=["metadatas"])
        ids_existentes = {
            m.get("ticket_id") for m in existentes["metadatas"] if m.get("ticket_id")
        }
        print(f"Tickets ya indexados en el vectorstore: {len(ids_existentes)}")

        documentos = _cargar_documentos_jsonl()
        nuevos = [
            d for d in documentos
            if d.metadata.get("ticket_id") not in ids_existentes
        ]
        print(f"Tickets nuevos por agregar: {len(nuevos)}")

        if not nuevos:
            print("No hay tickets nuevos — el vectorstore ya está al día.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "],
        )
        chunks_nuevos = splitter.split_documents(nuevos)
        print(f"Chunks nuevos generados: {len(chunks_nuevos)}")

        vectorstore.add_documents(chunks_nuevos)

        mlflow.log_metric("tickets_nuevos_agregados", len(nuevos))
        mlflow.log_metric("vectorstore_size", vectorstore._collection.count())
        print(f"Vectorstore actualizado. Total vectores: {vectorstore._collection.count()}")

    return None


# ---------------------------------------------------------------------------
# Pipeline RAG con @mlflow.trace y memoria
# ---------------------------------------------------------------------------

@mlflow.trace(name="load_vectorstore")
def load_vectorstore():
    """Carga ChromaDB desde disco usando el mismo método que build_vectorstore."""
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
    return vectorstore


@mlflow.trace(name="format_documents")
def format_docs(docs) -> str:
    """Une los chunks recuperados en un bloque de texto, incluyendo su
    metadata visible para que el LLM sepa a qué categoría/filial pertenece."""
    bloques = []
    for doc in docs:
        meta = doc.metadata
        encabezado = (
            f"[Categoría: {meta.get('categoria', '')} | "
            f"Subcategoría: {meta.get('subcategoria', '')} | "
            f"Filial: {meta.get('filial', '')}]"
        )
        bloques.append(f"{encabezado}\n{doc.page_content}")
    return "\n\n".join(bloques)


@mlflow.trace(name="retrieve_documents")
def retrieve_documents(question: str):
    """Recupera los k tickets más similares a la pregunta."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
    docs = retriever.invoke(question)
    return docs


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str, history: str = "") -> str:
    """Genera respuesta usando Mistral API."""
    from langchain_mistralai import ChatMistralAI

    prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question,
        "history": history,
    })


@mlflow.trace(name="rag_pipeline_recamier")
def ask(question: str, session_id: str = "default") -> str:
    # Detectar si es conversación social — no buscar en tickets
    saludos = ["me llamo", "mi nombre es", "hola", "buenos", "gracias", "chao"]
    es_social = any(s in question.lower() for s in saludos)

    history_list = get_history(session_id, limit=6)
    history_text = format_history_for_prompt(history_list)

    if es_social:
        context = f"El usuario dice: {question}"
    else:
        docs = retrieve_documents(question)
        context = format_docs(docs)

    respuesta = generate_answer(question, context, history_text)
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", respuesta)
    return respuesta


if __name__ == "__main__":
    pregunta = "¿Cómo se resuelve un problema de conexión VPN?"
    print(f"\nPregunta: {pregunta}")
    print("\nRespuesta:")
    print(ask(pregunta, session_id="test"))