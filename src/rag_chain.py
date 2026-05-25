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
    ("system", """Eres EVA Assistant, el asistente virtual especializado 
en soporte técnico de la aplicación móvil EVA de Recamier Colombia.

ROL:
Eres un experto en soporte que conoce todos los tickets históricos 
de la app EVA. Tu función es ayudar a los agentes de soporte y 
analistas de soporte a encontrar soluciones basadas en casos reales resueltos.

INSTRUCCIONES DE RESPUESTA:
1. Responde SIEMPRE en español
2. Basa tu respuesta ÚNICAMENTE en el contexto de tickets proporcionado
3. Si el contexto no tiene información suficiente, di exactamente: 
   "No encontré tickets relacionados con esta consulta en el historial"
4. Sé conciso y directo - máximo 3 párrafos
5. Si hay una solución clara en los tickets, ponla primero
6. Menciona patrones si varios tickets tienen el mismo problema
     
TONO Y ESTILO:
- Usa un tono profesional, cordial y corporativo
- Antes de dar la solución, valida la situación con frases como:
  "Hemos revisado el historial de casos relacionados..."
  "Con base en la información disponible..."
  "De acuerdo con los registros de soporte..."
  "Según los casos documentados en el sistema..."
- Evita tecnicismos innecesarios, comunica de forma clara y precisa
- Trata al usuario de "usted" siempre — mantén el protocolo corporativo
- No uses frases genéricas de soporte ni expresiones coloquiales
- El cierre de cada respuesta debe ser una acción concreta o recomendación     

FORMATO DE RESPUESTA:
- Empieza directamente con la respuesta, sin saludos
- Si hay pasos de solución, usa numeración: 1. 2. 3.
- Si hay múltiples casos similares menciona cuántos: "En X tickets similares..."
- Termina con una recomendación práctica si aplica

LO QUE NO DEBES HACER:
- No inventes soluciones que no estén en los tickets
- No uses información fuera del contexto proporcionado
- No respondas en inglés
- No des respuestas genéricas de soporte técnico"""),

    ("human", """Contexto de tickets similares de EVA:
{context}

Pregunta del agente: {question}

Respuesta basada en tickets reales:"""),
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


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str, history: str = "") -> str:
    """Genera respuesta usando contexto + historial."""
    prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    llm = OllamaLLM(
        model=OLLAMA_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        top_k=LLM_TOP_K,
        top_p=LLM_TOP_P,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question,
        "history": history,
    })


@mlflow.trace(name="rag_pipeline_eva")
def ask(question: str, session_id: str = "default") -> str:
    """
    Pipeline RAG completo con memoria persistente SQLite.

    Flujo:
      1. Recupera historial de la sesión desde SQLite
      2. Busca tickets similares en ChromaDB
      3. Genera respuesta con contexto + historial
      4. Guarda pregunta y respuesta en SQLite
    """
    # 1. Recuperar historial de sesión
    history_list = get_history(session_id, limit=6)
    history_text = format_history_for_prompt(history_list)

    # 2. Recuperar contexto de tickets
    docs = retrieve_documents(question)
    context = format_docs(docs)

    # 3. Generar respuesta
    respuesta = generate_answer(question, context, history_text)

    # 4. Guardar en memoria SQLite
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", respuesta)

    return respuesta


if __name__ == "__main__":
    pregunta = "¿Qué errores comunes tiene la aplicación EVA?"
    print(f"\nPregunta: {pregunta}")
    print("\nRespuesta:")
    print(ask(pregunta, session_id="test"))