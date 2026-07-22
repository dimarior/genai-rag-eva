"""
rag_chain.py
------------
Pipeline RAG completo con MLflow Genai Tracing y memoria persistente en
Supabase. Cubre TODAS las categorias de soporte de Recamier (Aplicaciones,
Conectividad, Admin Usuarios, Backup, Hardware, etc.) y todas las filiales.

Vectorstore: Supabase + pgvector (reemplaza a ChromaDB local, necesario
para poder desplegar en Streamlit Community Cloud sin depender de un
filesystem persistente).

Embeddings: sentence-transformers (local, sin API key ni servidor externo).
"""

import json
import mlflow
from supabase import create_client
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    EMBEDDING_MODEL_NAME,
    SUPABASE_URL,
    SUPABASE_KEY,
)
from src.memory import (
    save_message,
    get_history,
    format_history_for_prompt,
)

# MLflow es opcional: si no hay un servidor de tracking disponible (por
# ejemplo, en Streamlit Community Cloud, donde no existe 127.0.0.1:5000),
# se cambia a un registro local en archivo (sin red) para que el resto del
# codigo (los decoradores @mlflow.trace usados mas abajo) sigan funcionando
# sin intentar conectarse a un servidor que no existe.
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as _e:
    print(f"MLflow remoto no disponible ({_e}); usando registro local en archivo.")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

TABLE_NAME = "documents"
QUERY_NAME = "match_documents"

# Umbral minimo de similitud (0 a 1) para considerar que un ticket es
# realmente relevante. Si el mejor resultado no supera este valor, se
# asume que la pregunta esta fuera del dominio de soporte tecnico (por
# ejemplo, temas ajenos como historia, ciencia, cultura general, etc.)
# y no se llama a Mistral con contexto irrelevante.
UMBRAL_SIMILITUD = 0.45

PROMPT_TEMPLATE = [
    ("system", """Eres el asistente virtual de soporte Recamier interno
de Recamier S.A. y sus filiales (Recamier, Dermodis, Lansey, Keramer,
Arte Frances, Fondelar).

COMPORTAMIENTO CONVERSACIONAL:
- Si el usuario saluda sin dar su nombre, responde cordialmente
  y presentate brevemente:
  "Hola, soy tu asistente virtual de soporte Recamier. ¿En que puedo ayudarte?"
- NO preguntes el nombre proactivamente — espera a que el usuario
  lo comparta voluntariamente.
- Si el usuario comparte su nombre, acusalo de recibo cordialmente
  y usalo en las respuestas siguientes.
- Si preguntan "¿como me llamo?" y conoces el nombre del historial,
  responde directamente sin buscar en tickets:
  "Tu te llamas [nombre]."
- Para preguntas personales o sociales que no son de soporte,
  responde brevemente y redirige al tema de soporte tecnico.
- Si el usuario envia una imagen, audio o documento PDF, el contenido
  ya viene transcrito/extraido en el mensaje. Tratalo como una
  consulta normal, usando ese texto como la pregunta o el contexto.

ROL:
Experto en soporte que conoce todos los tickets historicos de Recamier,
cubriendo aplicaciones (IBES, EVA, ONBASE, OUTLOOK, SNAP, BI4WEB, TEAMS,
E-commerce, etc.), conectividad y VPN, administracion de usuarios,
backups, hardware y software de PC, equipos moviles, impresion y
telefonia. Ayudas a agentes de soporte y analistas a encontrar
soluciones basadas en casos reales documentados en el sistema.

COMO USAR EL CONTEXTO:
- Cada ticket recuperado incluye su Categoria, Subcategoria y Filial.
  Usa SIEMPRE esos campos para confirmar que la solucion corresponde
  exactamente al sistema/aplicacion que el usuario esta consultando.
- Si hay tickets de varias categorias distintas en el contexto y solo
  una corresponde a la consulta del usuario, ignora las demas y usa
  solo la relevante — no mezcles soluciones de sistemas diferentes.
- Si la filial del ticket recuperado es distinta a la que menciona el
  usuario, aclaralo antes de dar la solucion (los procesos pueden
  variar entre Recamier, Dermodis, Lansey, etc.).

TONO Y ESTILO:
- Usa un tono profesional, cordial y corporativo
- Antes de dar la solucion valida con frases como:
  "De acuerdo con los registros de soporte..."
  "Con base en la informacion disponible en el sistema..."
  "Segun los casos documentados..."
- Trata al usuario de "tu" siempre, nunca de "usted"
- Si conoces el nombre del usuario usalo naturalmente
- Cierra siempre con una accion concreta o recomendacion

INSTRUCCIONES:
1. Responde SIEMPRE en español
2. Si el contexto tiene informacion relevante y de la categoria correcta,
   usala para responder
3. Si NO hay informacion suficiente o de la categoria correcta en los
   tickets responde exactamente:
   "De acuerdo con nuestra base de conocimiento, no encontre registros
   relacionados con su consulta. Le sugiero contactar directamente al
   equipo de soporte tecnico para una atencion personalizada."
4. Maximo 3 parrafos — se conciso y directo
5. Si hay solucion clara en los tickets ponla primero

PLANTILLA DE RESPUESTA cuando hay informacion:
**Situacion identificada:**
[Describe el problema basado en los tickets, mencionando la categoria/aplicacion]

**Solucion documentada:**
[Pasos exactos segun los tickets — numerados]

**Recomendacion:**
[Accion concreta para prevenir o escalar]

LO QUE NO DEBES HACER:
- No inventes soluciones que no esten en los tickets
- No respondas en ingles
- No mezcles soluciones de categorias/aplicaciones distintas
- No ignores el nombre del usuario si lo conoces"""),

    ("human", """{history}

Contexto de tickets similares de Recamier:
{context}

Consulta: {question}

Respuesta:"""),
]


# El modelo de embeddings y el cliente de Supabase se crean UNA SOLA VEZ
# por proceso y se reutilizan — crearlos de nuevo en cada pregunta era
# el principal cuello de botella de rendimiento.
_embeddings_instance = None
_supabase_client_instance = None


def _get_embeddings():
    """Embeddings locales con sentence-transformers, sin API key.
    Se crea una sola vez y se reutiliza (singleton por proceso)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance


def _get_supabase_client():
    """Cliente de Supabase, tambien reutilizado en vez de recrearse
    en cada consulta."""
    global _supabase_client_instance
    if _supabase_client_instance is None:
        _supabase_client_instance = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client_instance


def _get_vectorstore():
    return SupabaseVectorStore(
        client=_get_supabase_client(),
        embedding=_get_embeddings(),
        table_name=TABLE_NAME,
        query_name=QUERY_NAME,
    )


def _cargar_documentos_jsonl() -> list[Document]:
    """Lee tickets_recamier.jsonl (generado por src/ingest.py)."""
    jsonl_path = PROCESSED_DATA_DIR / "tickets_recamier.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            "No se encontro tickets_recamier.jsonl. "
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


def build_vectorstore(tamano_lote: int = 100):
    """Sube los tickets a Supabase por lotes, con progreso visible.
    Requiere que la tabla 'documents' y la funcion 'match_documents' ya
    existan en Supabase (ver SQL de configuracion inicial)."""
    with mlflow.start_run(run_name="build_vectorstore_supabase"):
        documentos = _cargar_documentos_jsonl()
        print(f"Tickets cargados: {len(documentos)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(documentos)
        total_chunks = len(chunks)
        print(f"Chunks generados: {total_chunks}")

        mlflow.log_param("modelo_embeddings", EMBEDDING_MODEL_NAME)
        mlflow.log_param("modelo_llm", MISTRAL_MODEL)
        mlflow.log_param("tamano_lote", tamano_lote)
        mlflow.log_metric("total_chunks", total_chunks)

        embeddings = _get_embeddings()
        client = _get_supabase_client()

        import time
        inicio = time.time()
        procesados = 0

        for i in range(0, total_chunks, tamano_lote):
            lote = chunks[i:i + tamano_lote]
            SupabaseVectorStore.from_documents(
                lote,
                embeddings,
                client=client,
                table_name=TABLE_NAME,
                query_name=QUERY_NAME,
            )
            procesados += len(lote)
            transcurrido = time.time() - inicio
            velocidad = procesados / transcurrido if transcurrido > 0 else 0
            restantes = total_chunks - procesados
            eta_seg = restantes / velocidad if velocidad > 0 else 0
            print(
                f"  {procesados}/{total_chunks} chunks "
                f"({100*procesados/total_chunks:.1f}%) — "
                f"{velocidad:.1f} chunks/seg — ETA: {eta_seg/60:.1f} min"
            )

        print(f"Vectorstore en Supabase actualizado. Total procesados: {procesados}")
    return None


@mlflow.trace(name="format_documents")
def format_docs(docs) -> str:
    """Une los chunks recuperados en un bloque de texto, con su metadata visible."""
    bloques = []
    for doc in docs:
        meta = doc.metadata
        encabezado = (
            f"[Categoria: {meta.get('categoria', '')} | "
            f"Subcategoria: {meta.get('subcategoria', '')} | "
            f"Filial: {meta.get('filial', '')}]"
        )
        bloques.append(f"{encabezado}\n{doc.page_content}")
    return "\n\n".join(bloques)


@mlflow.trace(name="retrieve_documents")
def retrieve_documents(question: str, filial: str = None, categoria: str = None):
    """Busca los k tickets mas similares en Supabase, con filtro real por
    filial/categoria vía la funcion match_documents (filtro jsonb).
    Si el filtro no encuentra nada, amplia la busqueda sin filtro.

    Devuelve una tupla (documentos, mejor_puntaje_similitud) — el puntaje
    del resultado mas parecido, usado para detectar preguntas fuera del
    dominio de soporte tecnico (ver UMBRAL_SIMILITUD)."""
    vectorstore = _get_vectorstore()

    filtro = {}
    if filial and filial != "todas":
        filtro["filial"] = filial
    if categoria and categoria != "todas":
        filtro["categoria"] = categoria

    resultados = []
    if filtro:
        resultados = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVER_K, filter=filtro
        )

    if not resultados:
        resultados = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVER_K
        )

    if not resultados:
        return [], 0.0

    docs = [doc for doc, score in resultados]
    mejor_puntaje = max(score for doc, score in resultados)
    return docs, mejor_puntaje


@mlflow.trace(name="generate_answer")
def generate_answer(question: str, context: str, history: str = "") -> str:
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
def ask(question: str, session_id: str = "default", filial: str = None, categoria: str = None) -> str:
    saludos = ["me llamo", "mi nombre es", "hola", "buenos", "gracias", "chao"]
    es_social = any(s in question.lower() for s in saludos)

    history_list = get_history(session_id, limit=6)
    history_text = format_history_for_prompt(history_list)

    if es_social:
        context = f"El usuario dice: {question}"
        respuesta = generate_answer(question, context, history_text)
    else:
        docs, mejor_puntaje = retrieve_documents(question, filial=filial, categoria=categoria)

        if mejor_puntaje < UMBRAL_SIMILITUD:
            # La pregunta no se parece lo suficiente a ningun ticket real:
            # se asume fuera del dominio de soporte tecnico de Recamier.
            # No se llama a Mistral, para no arriesgar una respuesta
            # inventada ni gastar tokens en un contexto irrelevante.
            respuesta = (
                "Tu consulta parece estar fuera del alcance del soporte tecnico "
                "de Recamier, por lo que no cuento con informacion en los tickets "
                "historicos para responderte. Si tu pregunta si esta relacionada "
                "con soporte tecnico, intenta reformularla con mas detalle."
            )
        else:
            context = format_docs(docs)
            respuesta = generate_answer(question, context, history_text)
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", respuesta)
    return respuesta


if __name__ == "__main__":
    pregunta = "¿Como se resuelve un problema de conexion VPN?"
    print(f"\nPregunta: {pregunta}")
    print("\nRespuesta:")
    print(ask(pregunta, session_id="test"))