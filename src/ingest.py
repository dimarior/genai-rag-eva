"""
ingest.py
---------
Paso 1 del pipeline RAG:
Lee TODOS los archivos de tickets que encuentre en data/raw/ — tanto .xlsx
como .csv, de cualquier año o exportación (2024, 2025, 2026, o lo que se
agregue después) — los combina, limpia y guarda como un único JSONL (un
ticket por línea) con texto + metadata, listo para indexar en ChromaDB.

No hay que hardcodear nombres de archivo: cualquier .xlsx o .csv que pongas
en data/raw/ se procesa automáticamente. Si un archivo tiene una estructura
de columnas distinta a la esperada (por ejemplo, un export viejo con otros
nombres de campo), se salta con una advertencia en vez de romper todo el
proceso — así un archivo con formato distinto no bloquea a los demás.
"""

import json
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Cada campo puede tener MÁS DE UN nombre posible según el export
# (HubSpot no siempre nombra las columnas igual entre exports distintos).
# Si llega un archivo nuevo con un nombre de columna que no está aquí,
# solo agrega el nombre exacto a la lista correspondiente — no hay que
# tocar el resto del código.
COLUMNAS_ALIASES = {
    "titulo": ["Nombre del Ticket de Servicio", "Título del Ticket", "Asunto"],
    "descripcion": ["Descripción del Ticket de Servicio", "Descripción"],
    "tipo_solicitud": ["Tipo de Solicitud Recamier", "Tipo de Solicitud"],
    "categoria": ["Categoría Ticket Recamier", "Categoría"],
    "subcategoria": ["Subcategoría Ticket Recamier", "Subcategoría"],
    "estado": ["Estado del Ticket de Servicio", "Estado"],
    "filial": ["Filial Recamier (T)", "Filial"],
    "ciudad": ["Ciudad Recamier ", "Ciudad Recamier", "Ciudad"],
    "area_solicitante": ["Area Solicitante Recamier (T)", "Área Solicitante Recamier (T)", "Área Solicitante"],
    "prioridad": ["Prioridad"],
    "solucion": [
        "Associated Note",
        "Solución entregada al cliente RECAMIER que justifica el cierre del Ticket",
        "Solución",
    ],
    "fecha_creacion": ["Fecha de creación", "Fecha Creación"],
    "ticket_id": ["Ticket ID", "ID de Ticket de Servicio"],
    "nivel_escalamiento": ["Nivel 2 - Línea Responsable - MdS", "Línea Responsable"],
    "es_vip": ["¿Ticket VIP Recamier?", "Ticket VIP"],
}

# Campos sin los cuales no vale la pena procesar el archivo
COLUMNAS_REQUERIDAS = ["titulo", "categoria", "subcategoria", "solucion"]

EXTENSIONES_SOPORTADAS = ("*.xlsx", "*.csv")


def _mapear_columnas(df: pd.DataFrame) -> dict:
    """Para cada campo canónico, busca cuál de sus alias existe en el
    archivo y devuelve {campo_canonico: nombre_de_columna_real_o_None}."""
    mapeo = {}
    for campo, alias in COLUMNAS_ALIASES.items():
        columna_real = next((a for a in alias if a in df.columns), None)
        mapeo[campo] = columna_real
    return mapeo


def _limpiar_texto(valor) -> str:
    """Convierte NaN/None a string vacío y recorta espacios."""
    if pd.isna(valor):
        return ""
    return str(valor).strip()


def _leer_archivo(path) -> pd.DataFrame | None:
    """Lee un .xlsx o .csv según su extensión. Devuelve None si falla."""
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception as e:
        print(f"  ⚠ No se pudo leer {path.name}: {e}")
        return None


def _procesar_archivo(path) -> list[dict]:
    """Lee un archivo de tickets y devuelve la lista de registros {text, metadata}.
    Detecta las columnas por alias; si le faltan campos requeridos, lo
    salta (devuelve lista vacía) con aviso explicando qué falta."""
    df = _leer_archivo(path)
    if df is None:
        return []

    col = _mapear_columnas(df)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if col[c] is None]
    if faltantes:
        print(
            f"  ⚠ {path.name}: se salta — no se encontró columna para: {faltantes}\n"
            f"    Columnas disponibles en el archivo: {list(df.columns)}\n"
            f"    Si el archivo SÍ trae esa información con otro nombre de columna,\n"
            f"    agrega ese nombre a COLUMNAS_ALIASES en este archivo."
        )
        return []

    total_original = len(df)

    # Solo tickets con solución/gestión real documentada por el agente.
    df = df[df[col["solucion"]].notna()].copy()

    print(f"  {path.name}: {len(df)}/{total_original} tickets usables")

    registros = []
    for _, row in df.iterrows():
        def obtener(campo):
            columna = col[campo]
            return _limpiar_texto(row[columna]) if columna else ""

        categoria = obtener("categoria")
        subcategoria = obtener("subcategoria")
        filial = obtener("filial")
        area_solicitante = obtener("area_solicitante")
        tipo_solicitud = obtener("tipo_solicitud")
        descripcion = obtener("descripcion")
        # Associated Note puede traer varias notas separadas por ";" — cada
        # una suele ser un paso distinto de gestion/escalamiento del ticket
        # (por ejemplo: nota del agente nivel 1, luego nota de nivel 2/3).
        # Se muestran como pasos numerados para que quede claro que es un
        # historial, no una sola solucion.
        solucion_raw = obtener("solucion")
        pasos_solucion = [p.strip() for p in solucion_raw.split(";") if p.strip()]
        if pasos_solucion:
            solucion = "\n".join(
                f"  {i}. {paso}" for i, paso in enumerate(pasos_solucion, start=1)
            )
        else:
            solucion = "  (sin detalle documentado)"
        titulo = obtener("titulo")
        ticket_id = obtener("ticket_id")
        fecha = obtener("fecha_creacion")
        nivel_escalamiento = obtener("nivel_escalamiento")
        es_vip = obtener("es_vip")

        texto = (
            f"Ticket: {titulo}\n"
            f"Categoría: {categoria}\n"
            f"Subcategoría: {subcategoria}\n"
            f"Filial: {filial}\n"
            f"Área solicitante: {area_solicitante}\n"
            f"Tipo de solicitud: {tipo_solicitud}\n"
        )
        if nivel_escalamiento:
            texto += f"Nivel de escalamiento / Línea responsable: {nivel_escalamiento}\n"
        if es_vip:
            texto += f"Ticket VIP: {es_vip}\n"
        texto += (
            f"\nDescripción del problema:\n{descripcion}\n\n"
            f"Historial de gestión / escalamiento (en orden):\n{solucion}"
        )

        metadata = {
            "ticket_id": ticket_id,
            "categoria": categoria,
            "subcategoria": subcategoria,
            "filial": filial,
            "nivel_escalamiento": nivel_escalamiento,
            "es_vip": es_vip,
            "area_solicitante": area_solicitante,
            "tipo_solicitud": tipo_solicitud,
            "fecha_creacion": fecha,
            "archivo_origen": path.name,
        }

        registros.append({"text": texto, "metadata": metadata})

    return registros


def xlsx_to_jsonl() -> str:
    """
    Busca TODOS los .xlsx y .csv en data/raw/, los procesa y los combina
    en un único JSONL, eliminando duplicados por Ticket ID.

    Returns:
        str: Ruta del archivo generado.
    """
    archivos = sorted(
        p for patron in EXTENSIONES_SOPORTADAS for p in RAW_DATA_DIR.glob(patron)
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .xlsx o .csv en {RAW_DATA_DIR}\n"
            f"Copia ahí los consolidados (2024, 2025, 2026, etc.)"
        )

    print(f"Archivos encontrados: {len(archivos)}")
    for a in archivos:
        print(f"  - {a.name}")
    print()

    todos_los_registros = []
    for path in archivos:
        todos_los_registros.extend(_procesar_archivo(path))

    if not todos_los_registros:
        raise ValueError(
            "Ningún archivo pudo procesarse (todos fueron saltados). "
            "Revisa los avisos de arriba."
        )

    # Deduplicar por ticket_id, en caso de que el mismo ticket aparezca
    # en mas de un archivo (ej. un export mas reciente con notas de
    # seguimiento/escalamiento agregadas despues). En vez de quedarse con
    # "el primero que aparece", se compara el LARGO del texto (titulo +
    # descripcion + solucion) y se conserva la version MAS DETALLADA —
    # asi, si un archivo nuevo trae notas de solucion mas completas para
    # un ticket que ya existia, esa version gana sobre la mas corta.
    mejor_por_id = {}
    sin_ticket_id = []

    for r in todos_los_registros:
        tid = r["metadata"]["ticket_id"]
        if not tid:
            sin_ticket_id.append(r)
            continue
        actual = mejor_por_id.get(tid)
        if actual is None or len(r["text"]) > len(actual["text"]):
            mejor_por_id[tid] = r

    # Los registros sin ticket_id se deduplican por texto exacto (caso raro)
    vistos_sin_id = set()
    registros_sin_id_unicos = []
    for r in sin_ticket_id:
        if r["text"] not in vistos_sin_id:
            vistos_sin_id.add(r["text"])
            registros_sin_id_unicos.append(r)

    registros_unicos = list(mejor_por_id.values()) + registros_sin_id_unicos

    duplicados = len(todos_los_registros) - len(registros_unicos)
    if duplicados:
        print(f"\n{duplicados} tickets duplicados detectados — se conservo "
              f"en cada caso la version con mas detalle documentado")

    print(f"\nTotal tickets únicos combinados: {len(registros_unicos)}")

    output_path = PROCESSED_DATA_DIR / "tickets_recamier.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in registros_unicos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Documento guardado en: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    xlsx_to_jsonl()