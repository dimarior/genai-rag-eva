"""
ingest.py
---------
Paso 1 del pipeline RAG:
Lee el CSV de tickets EVA exportado desde HubSpot,
combina los campos relevantes y guarda un documento
de texto plano en data/processed/tickets_eva.txt

Equivalente al src/data.py del artículo de Medium,
adaptado para RAG en lugar de clasificación.
"""

import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Nombre exacto del archivo CSV exportado de HubSpot
CSV_NAME = "hubspot-crm-exports-r-tickets-eva_eva4-2026-05-07.csv"


def csv_to_text() -> str:
    """
    Lee el CSV de tickets EVA, combina los campos más
    relevantes por fila y guarda un único archivo .txt
    con todos los tickets separados por '---'.

    Returns:
        str: Ruta del archivo generado.
    """
    csv_path = RAW_DATA_DIR / CSV_NAME

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV en {csv_path}\n"
            f"Copia el archivo dentro de: data/raw/"
        )

    df = pd.read_csv(csv_path)

    # Reemplazar valores nulos para no perder contexto
    df = df.fillna("Sin información")

    print(f"📋 Total tickets cargados: {len(df)}")
    print(f"📋 Columnas: {df.columns.tolist()}")

    textos = []
    for _, row in df.iterrows():
        texto = f"""Ticket ID: {row['Ticket ID']}
Título: {row['Nombre del Ticket de Servicio']}
Tipo de solicitud: {row['Tipo de Solicitud Recamier']}
Categoría: {row['Categoría Ticket Recamier']}
Subcategoría: {row['Subcategoría Ticket Recamier']}
Estado: {row['Estado del Ticket de Servicio']}
Fuente: {row['Fuente']}
Fecha creación: {row['Fecha de creación']}
Fecha cierre: {row['Fecha de cierre']}
Agente mesa: {row['Agente Mesa Recamier']}
Ciudad: {row['Ciudad Recamier ']}
Filial: {row['Filial Recamier (T)']}
Solución: {row['Solución entregada al cliente RECAMIER que justifica el cierre del Ticket']}"""
        textos.append(texto)

    # Guardar como un único documento con separadores
    output_path = PROCESSED_DATA_DIR / "tickets_eva.txt"
    output_path.write_text(
        "\n\n---\n\n".join(textos),
        encoding="utf-8"
    )

    print(f"✅ {len(textos)} tickets procesados")
    print(f"✅ Documento guardado en: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    csv_to_text()
