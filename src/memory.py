"""
src/memory.py
--------------
Memoria persistente de conversaciones usando Supabase (Postgres en la nube).
Reemplaza la version anterior con SQLite local, que no servia para
desplegar en Streamlit Community Cloud (filesystem efimero: cada vez que
la app se reinicia o "duerme", se perdia el historial).

Requiere SUPABASE_URL y SUPABASE_KEY en el .env, y la tabla "conversaciones"
ya creada en Supabase (ver el SQL de configuracion inicial).
"""

from datetime import datetime, timezone
from supabase import create_client, Client

from src.config import SUPABASE_URL, SUPABASE_KEY

_client: Client = None


def _get_client() -> Client:
    """Crea el cliente de Supabase una sola vez y lo reutiliza."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def save_message(session_id: str, role: str, content: str):
    """Guarda un mensaje en el historial de la sesion."""
    client = _get_client()
    client.table("conversaciones").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Recupera los ultimos N mensajes de una sesion, en orden cronologico.
    Retorna lista de dicts con 'role' y 'content'.
    """
    client = _get_client()
    respuesta = (
        client.table("conversaciones")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    filas = respuesta.data or []
    # Supabase las devuelve mas reciente primero; invertimos para orden cronologico
    return [{"role": f["role"], "content": f["content"]} for f in reversed(filas)]


def get_all_sessions() -> list[str]:
    """Retorna todos los session_id unicos que tienen historial."""
    client = _get_client()
    respuesta = client.table("conversaciones").select("session_id").execute()
    filas = respuesta.data or []
    return sorted(set(f["session_id"] for f in filas))


def format_history_for_prompt(history: list[dict]) -> str:
    """Convierte el historial en texto plano legible para inyectar en el prompt.
    Se mantiene completo (sin truncar) para conservar memoria real de la
    conversacion (incluyendo detalles de PDFs/imagenes/audio ya analizados
    en turnos anteriores) — el control sobre cuando usarlo o ignorarlo
    depende de la instruccion del prompt en rag_chain.py, no de cortar
    texto aqui."""
    if not history:
        return ""
    lineas = []
    for msg in history:
        etiqueta = "Usuario" if msg["role"] == "user" else "Asistente"
        lineas.append(f"{etiqueta}: {msg['content']}")
    return "\n".join(lineas)