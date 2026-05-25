"""
src/memory.py
-------------
Memoria persistente con SQLite para el chatbot EVA.
Guarda el historial de conversaciones entre sesiones.

"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

from src.config import ROOT_DIR

# Ruta del archivo SQLite
DB_PATH = ROOT_DIR / "eva_memory.db"


def init_db():
    """Crea las tablas si no existen."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session
        ON conversations(session_id, created_at)
    """)

    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    """Guarda un mensaje en el historial."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Recupera los últimos N mensajes de una sesión.
    Retorna lista de dicts con 'role' y 'content'.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM conversations "
        "WHERE session_id = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    # Invertir para orden cronológico
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def get_all_sessions() -> list[str]:
    """Retorna todas las sesiones únicas."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT session_id FROM conversations "
        "ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def clear_session(session_id: str):
    """Borra el historial de una sesión."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM conversations WHERE session_id = ?",
        (session_id,)
    )
    conn.commit()
    conn.close()


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Convierte el historial en texto para incluir en el prompt.
    El LLM usa esto como contexto de conversación previa.
    """
    if not history:
        return ""

    lines = ["Historial de conversación previa:"]
    for msg in history:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        lines.append(f"{role}: {msg['content'][:300]}")

    return "\n".join(lines)


# Inicializar DB al importar
init_db()