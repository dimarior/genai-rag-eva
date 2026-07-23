from src.rag_chain import _get_vectorstore, RETRIEVER_K

preguntas = [
    "quien es el presidente de estados unidos",
    "¿Cuál es la distancia de la Tierra a la Luna?",
    "¿Cómo se resuelve un problema de conexión VPN?",
]

vs = _get_vectorstore()
for p in preguntas:
    resultados = vs.similarity_search_with_relevance_scores(p, k=RETRIEVER_K)
    print(f"\nPregunta: {p}")
    for doc, score in resultados[:3]:
        print(f"  score={score:.4f}  |  {doc.page_content[:60]}...")