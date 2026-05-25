"""
evaluate.py
-----------
Paso 9 del pipeline RAG:
Evalúa la calidad de las respuestas del sistema RAG
usando un conjunto de preguntas de prueba basadas
en los tickets reales de EVA.

Registra scores en MLflow y guarda reporte JSON.

Equivalente al proceso de evaluación de modelos
del artículo de Medium (accuracy, f1, etc.)
pero adaptado para RAG con métricas de relevancia.
"""

import json
import time
import mlflow

from src.rag_chain import ask
from src.config import EVALUATION_DIR, MLFLOW_TRACKING_URI, EXPERIMENT_NAME


# ---------------------------------------------------------------------------
# Preguntas de evaluación basadas en los tickets reales del CSV EVA
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "pregunta": "¿Qué errores comunes reportan los usuarios de la app EVA?",
        "keywords": ["aplicación", "error", "problema", "sistema", "EVA"],
        "descripcion": "Pregunta sobre errores frecuentes",
    },
    {
        "pregunta": "¿Cómo se resuelven los tickets de EVA relacionados con RC?",
        "keywords": ["RC", "validar", "generar", "remoto", "soporte", "solución"],
        "descripcion": "Pregunta sobre resolución de RC",
    },
    {
        "pregunta": "¿Qué tipo de soporte se da en la ciudad de Cali para EVA?",
        "keywords": ["Cali", "soporte", "agente", "ticket", "área"],
        "descripcion": "Pregunta filtrada por ciudad",
    },
    {
        "pregunta": "¿Cuáles son los problemas reportados en EVA 4.0?",
        "keywords": ["EVA 4.0", "requerimiento", "versión", "actualización"],
        "descripcion": "Pregunta sobre versión específica",
    },
    {
        "pregunta": "¿Qué hace el soporte cuando un ticket queda pausado?",
        "keywords": ["pausado", "respuesta", "usuario", "espera", "estado"],
        "descripcion": "Pregunta sobre estados de ticket",
    },
]


def calcular_score(respuesta: str, keywords: list) -> float:
    """
    Métrica simple: porcentaje de keywords que aparecen
    en la respuesta (no distingue mayúsculas).
    Escala de 0.0 a 1.0
    """
    hits = sum(
        1 for kw in keywords
        if kw.lower() in respuesta.lower()
    )
    return hits / len(keywords)


def evaluar():
    """
    Corre todas las preguntas del EVAL_SET, calcula scores,
    los loguea en MLflow y guarda un reporte JSON en reports/evaluation/
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    resultados = []
    scores_totales = []

    print("=" * 60)
    print("📊 EVALUACIÓN DEL SISTEMA RAG — TICKETS EVA")
    print("=" * 60)

    with mlflow.start_run(run_name="evaluacion_rag_eva"):

        mlflow.log_param("total_preguntas_eval", len(EVAL_SET))

        for i, item in enumerate(EVAL_SET, start=1):
            print(f"\n🔍 Q{i}: {item['pregunta']}")

            inicio = time.time()
            respuesta = ask(item["pregunta"])
            latencia = round(time.time() - inicio, 3)

            score = calcular_score(respuesta, item["keywords"])
            scores_totales.append(score)

            # Loguear en MLflow
            mlflow.log_metric(f"score_q{i}", score)
            mlflow.log_metric(f"latencia_q{i}_seg", latencia)

            print(f"   ✅ Score: {score:.2f} | Latencia: {latencia}s")
            print(f"   💬 Respuesta (preview): {respuesta[:150]}...")

            resultados.append({
                "pregunta_num": i,
                "pregunta": item["pregunta"],
                "descripcion": item["descripcion"],
                "keywords_buscadas": item["keywords"],
                "respuesta_completa": respuesta,
                "score": score,
                "latencia_segundos": latencia,
            })

        # Score promedio global
        avg_score = sum(scores_totales) / len(scores_totales)
        mlflow.log_metric("avg_score_global", avg_score)

        print("\n" + "=" * 60)
        print(f"📈 Score promedio global: {avg_score:.2f} / 1.00")
        print("=" * 60)

        # Guardar reporte completo en JSON
        reporte_path = EVALUATION_DIR / "resultados_evaluacion.json"
        reporte_path.write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        mlflow.log_artifact(str(reporte_path), artifact_path="evaluation")

        print(f"\n✅ Reporte guardado en: {reporte_path}")
        print("✅ Resultados logueados en MLflow → http://127.0.0.1:5000")

    return resultados


if __name__ == "__main__":
    evaluar()
