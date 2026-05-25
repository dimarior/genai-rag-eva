"""
app/streamlit_app.py
"""

import base64
import uuid
import requests
from pathlib import Path
from PIL import Image
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"
HEALTH_URL = "http://127.0.0.1:8000/health"

def img_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def get_img_tag(path: str, width: str = "auto", extra_style: str = "") -> str:
    b64 = img_to_base64(path)
    if not b64:
        return ""
    ext = "jpg" if path.endswith(".jpg") else "png"
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};{extra_style}" />'

ASSETS   = Path("app/assets")
LOGO_EVA = str(ASSETS / "eva-logo.png")
BANNER   = str(ASSETS / "banner-recamier.jpg")

favicon = Image.open("app/assets/recamierco-favicon.ico")
st.set_page_config(
    page_title="Asistente de Soporte EVA — Recamier",
    page_icon=favicon,
    layout="centered",
)

# ---------------------------------------------------------------------------
# Inicializar session_id y historial de chat
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section[data-testid="stMain"] > div {
        background-color: #FFFFFF !important;
    }

    .banner-wrap {
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .banner-wrap img {
        width: 100%;
        display: block;
        max-height: 200px;
        object-fit: cover;
    }

    .hero-section {
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0.5rem 0 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero-section h1 {
        color: #1A1A2E;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 1.2rem;
        letter-spacing: -0.5px;
    }
    .hero-eva-logo { display: flex; justify-content: center; }

    .status-ok {
        background: #F0FFF4;
        border: 1px solid #9AE6B4;
        border-left: 4px solid #38A169;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        color: #276749;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }
    .status-err {
        background: #FFF5F5;
        border: 1px solid #FEB2B2;
        border-left: 4px solid #E53E3E;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        color: #C53030;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1A1A2E;
        margin: 1.4rem 0 0.7rem;
    }

    /* Preguntas frecuentes — rectángulo */
    .stButton > button {
        background: white !important;
        color: #4A5568 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        transition: all 0.2s !important;
        text-align: left !important;
        line-height: 1.4 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background: #F7FAFC !important;
        border-color: #87CEEB !important;
        color: #1A6B8A !important;
    }

    /* Botón Consultar — azul pill */
    .stButton > button[kind="primary"] {
        background: #87CEEB !important;
        color: #1A3A5C !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 99px !important;
        box-shadow: 0 3px 12px rgba(135,206,235,0.4) !important;
        transition: background 0.2s, box-shadow 0.2s, transform 0.2s !important;
        padding: 0.55rem 1rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #7C4DFF !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(124,77,255,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Botón Limpiar HTML */
    .btn-limpiar {
        background: #7DBE7E;
        color: white;
        border: none;
        border-radius: 99px;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.55rem 1rem;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 3px 12px rgba(125,190,126,0.4);
        transition: background 0.2s, box-shadow 0.2s, transform 0.2s;
        font-family: 'Inter', sans-serif;
        height: 42px;
    }
    .btn-limpiar:hover {
        background: #7C4DFF !important;
        color: white;
        box-shadow: 0 6px 20px rgba(124,77,255,0.4);
        transform: translateY(-1px);
    }

    .stTextArea textarea {
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background: #FAFAFA !important;
        font-size: 0.9rem !important;
        color: #2D3748 !important;
        box-shadow: none !important;
    }
    .stTextArea textarea:focus {
        border-color: #87CEEB !important;
        box-shadow: 0 0 0 3px rgba(135,206,235,0.15) !important;
        background: white !important;
    }
    [data-testid="stTextArea"] > div > div {
        border: none !important;
        box-shadow: none !important;
    }

    /* Chat history */
    .chat-user {
        background: #E8F4FD;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1A3A5C;
        text-align: right;
    }
    .chat-bot {
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #87CEEB;
        border-radius: 2px 12px 12px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2D3748;
        line-height: 1.7;
    }
    .chat-meta {
        font-size: 0.7rem;
        color: #A0AEC0;
        margin-top: 0.3rem;
    }
    .session-info {
        font-size: 0.72rem;
        color: #A0AEC0;
        text-align: right;
        margin-bottom: 0.5rem;
    }

    .respuesta-box {
        background: white;
        border: 1.5px solid #E2E8F0;
        border-top: 4px solid #87CEEB;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .respuesta-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #87CEEB;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.75rem;
    }
    .respuesta-text { color: #2D3748; line-height: 1.8; font-size: 0.95rem; }

    .metrics-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin-top: 1rem;
    }
    .metric-card {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-val { font-size:1rem; font-weight:700; color:#1A6B8A; display:block; }
    .metric-lbl { font-size:0.68rem; color:#A0AEC0; margin-top:2px; display:block; }

    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #E2E8F0, transparent);
        margin: 1.2rem 0;
    }

    .footer {
        text-align: center;
        background: #1A1A1A;
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 2.5rem;
        padding: 1.4rem;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
banner_tag = get_img_tag(BANNER, width="100%",
                         extra_style="max-height:200px;object-fit:cover;display:block;")
if banner_tag:
    st.markdown(f'<div class="banner-wrap">{banner_tag}</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
logo_eva_tag = get_img_tag(LOGO_EVA, width="300px",
                           extra_style="filter:brightness(0);")
st.markdown(f"""
<div class="hero-section">
    <h1>Asistente de Soporte</h1>
    <div class="hero-eva-logo">{logo_eva_tag}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estado API
# ---------------------------------------------------------------------------
try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.markdown(
            '<div class="status-ok">Sistema conectado y funcionando correctamente</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-err">El sistema respondió con un estado inesperado</div>',
            unsafe_allow_html=True)
except requests.exceptions.ConnectionError:
    st.markdown("""
    <div class="status-err">
        No se puede conectar con la API —
        Ejecuta: <code>uvicorn api.main:app --reload --host 127.0.0.1 --port 8000</code>
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Historial de conversación (memoria visual de sesión)
# ---------------------------------------------------------------------------
if st.session_state["chat_history"]:
    st.markdown('<div class="section-title">Conversación</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="session-info">Sesión: {st.session_state["session_id"]}</div>',
        unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">⏱️ {msg.get("latencia", "")}s · llama3.2</div>'
                f'</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Preguntas frecuentes
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Preguntas frecuentes</div>',
            unsafe_allow_html=True)

ejemplos = [
    "¿Cómo se resuelve cuando EVA no sincroniza?",
    "¿Qué hacer cuando un usuario no puede ingresar a EVA?",
    "¿Cómo se instala correctamente la app EVA?",
    "¿Cómo se resuelven problemas con los recibos de caja en EVA?",
    "¿Qué diferencia hay en soporte entre EVA y EVA 4.0?",
    "¿Cuándo se escala un ticket de EVA a tercer nivel?",
]

cols = st.columns(3)
for i, ejemplo in enumerate(ejemplos):
    if cols[i % 3].button(ejemplo, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_actual"] = ejemplo

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Nueva consulta
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Nueva consulta</div>',
            unsafe_allow_html=True)

if st.session_state.get("limpiar_clicked"):
    st.session_state["pregunta_actual"] = ""
    st.session_state["limpiar_clicked"] = False

pregunta_default = st.session_state.get("pregunta_actual", "")

pregunta = st.text_area(
    label="Pregunta",
    value=pregunta_default,
    placeholder="Escribe tu pregunta sobre los tickets de soporte EVA...",
    height=110,
    label_visibility="collapsed",
)

col1, col2 = st.columns(2)
with col1:
    consultar = st.button("Consultar", type="primary", use_container_width=True)
with col2:
    st.markdown("""
        <button class="btn-limpiar"
            onclick="
                var inputs = window.parent.document.querySelectorAll('textarea');
                inputs.forEach(function(el){
                    el.value = '';
                    el.dispatchEvent(new Event('input', {bubbles:true}));
                });
            ">
            Limpiar
        </button>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Procesar consulta
# ---------------------------------------------------------------------------
if consultar and pregunta.strip():
    with st.spinner("Analizando tickets y generando respuesta..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "pregunta": pregunta.strip(),
                    "session_id": st.session_state["session_id"]
                },
                timeout=120,
            )
            result = response.json()
            if "error" in result:
                st.error(f"{result['error']}")
            else:
                # Agregar al historial visual
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": pregunta.strip()
                })
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": result["respuesta"],
                    "latencia": result["latencia_segundos"]
                })
                st.session_state["pregunta_actual"] = ""

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">Respuesta del Asistente EVA</div>
                    <div class="respuesta-text">{result['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{result['latencia_segundos']}s</span>
                        <span class="metric-lbl">Tiempo de respuesta</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">1.153</span>
                        <span class="metric-lbl">Tickets analizados</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">llama3.2</span>
                        <span class="metric-lbl">Modelo activo</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API en :8000")
        except requests.exceptions.Timeout:
            st.warning("La consulta tardó demasiado. Intenta de nuevo.")

elif consultar and not pregunta.strip():
    st.warning("Escribe una pregunta antes de consultar.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Recamier · Transformación Digital · 2026
</div>
""", unsafe_allow_html=True)