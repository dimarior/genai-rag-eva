"""
rag_chain.py
------------
Pipeline RAG completo con MLflow Genai Tracing y memoria SQLite.

Pasos:
  1. load_vectorstore  — carga ChromaDB desde disco
  2. format_documents  — formatea chunks recuperados
  3. retrieve_documents — busca tickets similares
  4. generate_answer   — genera respuesta con historial
  5. ask               — función principal con memoria

Requiere: pip install "mlflow[genai]" google-generativeai
"""

import mlflow
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    PROCESSED_DATA_DIR,
    VECTORSTORE_DIR,
    OLLAMA_LLM_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    RETRIEVER_K,
    #GEMINI_API_KEY,
    #GEMINI_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
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
    ("system", """Eres EVA Assistant, asistente virtual especializado
en soporte técnico de la aplicación móvil EVA de Recamier Colombia.

COMPORTAMIENTO CONVERSACIONAL:
- Si el usuario saluda sin dar su nombre, responde cordialmente
  y preséntate brevemente:
  "Buenos días/tardes, soy EVA Assistant, asistente virtual de 
  soporte de Recamier. ¿En qué puedo ayudarle?"
- NO preguntes el nombre proactivamente — espera a que el usuario
  lo comparta voluntariamente.
- Si el usuario comparte su nombre, acúsalo de recibo cordialmente
  y úsalo en las respuestas siguientes.
- Si preguntan "¿cómo me llamo?" y conoces el nombre del historial,
  responde directamente sin buscar en tickets:
  "Usted se llama [nombre]."
- Para preguntas personales o sociales que no son de soporte,
  responde brevemente y redirige al tema de EVA.

ROL:
Experto en soporte que conoce todos los tickets históricos de EVA.
Ayudas a agentes de soporte y analistas a encontrar soluciones
basadas en casos reales documentados en el sistema.

CONTEXTO DEL SISTEMA:
- Los tickets son de tipo: Incidente o problema técnico / Nuevo requerimiento
- Las subcategorías son: Aplicaciones/EVA y Aplicaciones/EVA 4.0
- Las ciudades principales: Cali, Bogotá, Lima, Medellín, Bucaramanga
- Los estados posibles: Cierre Final, Pausado x Respuesta del Usuario,
  Pausado x Derivación a Aplicaciones, Pausado x 3er Nivel

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
2. Si el contexto tiene información relevante úsala para responder
3. Si NO hay información suficiente en los tickets responde exactamente:
   "De acuerdo con nuestra base de conocimiento, no encontré registros 
   relacionados con su consulta. Le sugiero contactar directamente al 
   equipo de soporte técnico para una atención personalizada."
4. Máximo 3 párrafos — sé conciso y directo
5. Si hay solución clara en los tickets ponla primero

PLANTILLA DE RESPUESTA cuando hay información:
**Situación identificada:**
[Describe el problema basado en los tickets]

**Solución documentada:**
[Pasos exactos según los tickets — numerados]

**Recomendación:**
[Acción concreta para prevenir o escalar]

LO QUE NO DEBES HACER:
- No inventes soluciones que no estén en los tickets
- No respondas en inglés
- No ignores el nombre del usuario si lo conoces"""),

    ("human", """{history}

Contexto de tickets similares de EVA:
{context}

Consulta: {question}

Respuesta:"""),
]


# ---------------------------------------------------------------------------
# Build vectorstore
# ---------------------------------------------------------------------------

def build_vectorstore():
    """Crea chunks, embeddings y guarda en ChromaDB."""
    with mlflow.start_run(run_name="build_vectorstore"):
        txt_path = PROCESSED_DATA_DIR / "tickets_eva.txt"
        if not txt_path.exists():
            raise FileNotFoundError(
                "No se encontró tickets_eva.txt. "
                "Ejecuta primero: python -m src.ingest"
            )

        texto_completo = txt_path.read_text(encoding="utf-8")
        print(f"Documento cargado: {len(texto_completo):,} caracteres")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["---", "\n\n", "\n", " "],
        )
        chunks = splitter.create_documents([texto_completo])
        print(f"Chunks generados: {len(chunks)}")

        mlflow.log_param("modelo_embeddings", OLLAMA_EMBEDDING_MODEL)
        mlflow.log_param("modelo_llm", OLLAMA_LLM_MODEL)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_metric("total_chunks", len(chunks))

        print(f"Creando embeddings con {OLLAMA_EMBEDDING_MODEL}... (puede tardar)")
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

        mlflow.log_metric("vectorstore_size", vectorstore._collection.count())
        print(f"Vectorstore guardado en: {VECTORSTORE_DIR}")
        print(f"Total vectores en ChromaDB: {vectorstore._collection.count()}")

    return None


# ---------------------------------------------------------------------------
# Pipeline RAG con @mlflow.trace y memoria
# ---------------------------------------------------------------------------

@mlflow.trace(name="load_vectorstore")
def load_vectorstore():
    """Carga ChromaDB desde disco usando el mismo método que build_vectorstore."""
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
    return vectorstore

@mlflow.trace(name="format_documents")
def format_docs(docs):
    """Une los chunks recuperados en un bloque de texto."""
    return "\n\n".join(doc.page_content for doc in docs)


@mlflow.trace(name="retrieve_documents")
def retrieve_documents(question: str):
    """Recupera los k tickets más similares a la pregunta."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
    docs = retriever.invoke(question)
    return docs

# Modelo Local------
# @mlflow.trace(name="generate_answer")
# def generate_answer(question: str, context: str, history: str = "") -> str:
#     """Genera respuesta usando contexto + historial."""
#     prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
#     llm = OllamaLLM(
#         model=OLLAMA_LLM_MODEL,
#         temperature=LLM_TEMPERATURE,
#         top_k=LLM_TOP_K,
#         top_p=LLM_TOP_P,
#     )
#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({
#         "context": context,
#         "question": question,
#         "history": history,
#     })

# Modelo Gemini------
# @mlflow.trace(name="generate_answer")
# def generate_answer(question: str, context: str, history: str = "") -> str:
#     """Genera respuesta usando Gemini API + contexto + historial."""
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     import os
#     os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

#     prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
#     llm = ChatGoogleGenerativeAI(
#         model=GEMINI_MODEL,
#         temperature=LLM_TEMPERATURE,
#     )
#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({
#         "context": context,
#         "question": question,
#         "history": history,
#     })


# Modelo Mistral------
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


@mlflow.trace(name="rag_pipeline_eva")
def ask(question: str, session_id: str = "default") -> str:
    # Detectar si es conversación social — no buscar en tickets
    saludos = ["me llamo", "mi nombre es", "hola", "buenos", "gracias", "chao"]
    es_social = any(s in question.lower() for s in saludos)
    
    history_list = get_history(session_id, limit=6)
    history_text = format_history_for_prompt(history_list)

    if es_social:
        # Responder directamente sin buscar en tickets
        context = f"El usuario dice: {question}"
    else:
        docs = retrieve_documents(question)
        context = format_docs(docs)

    respuesta = generate_answer(question, context, history_text)
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", respuesta)
    return respuesta


if __name__ == "__main__":
    pregunta = "¿Qué errores comunes tiene la aplicación EVA?"
    print(f"\nPregunta: {pregunta}")
    print("\nRespuesta:")
    print(ask(pregunta, session_id="test"))