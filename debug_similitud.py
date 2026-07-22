from src.rag_chain import _get_vectorstore, RETRIEVER_K

preguntas = [
    "¿Cual es la distancia de la Tierra a la Luna?",
    "¿Como se resuelve un problema de conexion VPN?",
    "¿Que me recomiendas para el desayuno?",
]

vs = _get_vectorstore()
for p in preguntas:
    resultados = vs.similarity_search_with_relevance_scores(p, k=RETRIEVER_K)
    print(f"\nPregunta: {p}")
    for doc, score in resultados[:3]:
        print(f"  score={score:.4f}  |  {doc.page_content[:60]}...")