from src.rag_chain import _get_vectorstore, RETRIEVER_K, UMBRAL_SIMILITUD

preguntas = [
    "[Compania: Recamier] quien es el presidente de estados unidos",
    "quien es el presidente de estados unidos",
]

vs = _get_vectorstore()
print(f"UMBRAL_SIMILITUD actual: {UMBRAL_SIMILITUD}\n")
for p in preguntas:
    resultados = vs.similarity_search_with_relevance_scores(
        p, k=RETRIEVER_K, filter={"filial": "Recamier"}
    )
    mejor = max((s for _, s in resultados), default=0)
    print(f"Pregunta: {p}")
    print(f"  Mejor puntaje CON filtro Recamier: {mejor:.4f}")
    for doc, score in resultados[:3]:
        print(f"    score={score:.4f}  |  {doc.page_content[:60]}...")
    print()