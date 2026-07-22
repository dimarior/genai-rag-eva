"""
app/streamlit_app.py - EVA Recamier
Version integrada, llama directo a src.rag_chain (sin FastAPI), mismo
patron que GAIA, con tema claro para la identidad de Recamier.
"""

import base64
import time
import uuid
from pathlib import Path
from PIL import Image
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELO_ACTIVO = "mistral-small-latest"
ASSETS = Path("app/assets")


@st.cache_resource(show_spinner="Iniciando...")
def get_rag():
    from src import rag_chain
    return rag_chain


def responder(pregunta: str, session_id: str, filial: str = None, categoria: str = None) -> dict:
    rag = get_rag()
    t0 = time.time()
    respuesta = rag.ask(pregunta, session_id=session_id, filial=filial, categoria=categoria)
    latencia = round(time.time() - t0, 3)
    return {"respuesta": respuesta, "latencia_segundos": latencia}


FILIALES = {
    "todas":        {"nombre": "Todas las compañías"},
    "recamier":     {"nombre": "Recamier"},
    "keramer":      {"nombre": "Keramer"},
    "lansey":       {"nombre": "Lansey"},
    "dermodis":     {"nombre": "Dermodis"},
    "fondelar":     {"nombre": "Fondelar"},
    "arte_frances": {"nombre": "Arte Francés"},
}

CATEGORIAS = {
    "todas": {"nombre": "Todas las categorías",
              "placeholder": "Pregunta sobre cualquier tema de soporte técnico..."},
    "aplicaciones": {"nombre": "Aplicaciones",
                      "placeholder": "Pregunta sobre IBES, EVA, ONBASE, Outlook, SNAP..."},
    "admin_usuarios": {"nombre": "Admin Usuarios",
                        "placeholder": "Pregunta sobre creación/modificación de usuarios, bloqueos..."},
    "conectividad": {"nombre": "Conectividad",
                      "placeholder": "Pregunta sobre VPN, red, unidades compartidas..."},
    "software_pc": {"nombre": "Software PC",
                     "placeholder": "Pregunta sobre Office 365, sistema operativo, instalación..."},
    "microinformatica": {"nombre": "Microinformática",
                          "placeholder": "Pregunta sobre contraseñas, alistamiento de equipos..."},
    "hardware_pc": {"nombre": "Hardware PC",
                     "placeholder": "Pregunta sobre mantenimiento, mouse, teclado, pantalla..."},
    "equipos_moviles": {"nombre": "Equipos Móviles",
                         "placeholder": "Pregunta sobre celulares, terminales móviles..."},
    "impresion": {"nombre": "Impresión",
                  "placeholder": "Pregunta sobre impresoras, escáneres..."},
    "backup": {"nombre": "Backup",
               "placeholder": "Pregunta sobre restauración o traslado de información..."},
    "telefonia_fija": {"nombre": "Telefonía Fija",
                        "placeholder": "Pregunta sobre teléfonos, llamadas, extensiones..."},
    "videovigilancia": {"nombre": "Videovigilancia",
                         "placeholder": "Pregunta sobre cámaras de seguridad..."},
}

FILIAL_CATEGORIAS = {
    "todas": list(CATEGORIAS.keys()),
    "recamier": list(CATEGORIAS.keys()),
    "keramer": ["todas", "aplicaciones", "admin_usuarios", "conectividad",
                "equipos_moviles", "hardware_pc", "microinformatica",
                "software_pc", "backup"],
    "lansey": ["todas", "aplicaciones", "admin_usuarios", "conectividad",
               "hardware_pc", "microinformatica", "software_pc",
               "telefonia_fija", "backup"],
    "dermodis": ["todas", "aplicaciones", "admin_usuarios", "conectividad",
                 "equipos_moviles", "hardware_pc", "microinformatica",
                 "software_pc", "backup"],
    "fondelar": ["todas", "aplicaciones", "admin_usuarios", "conectividad",
                 "hardware_pc", "impresion", "microinformatica",
                 "software_pc", "backup"],
    "arte_frances": ["todas", "aplicaciones", "admin_usuarios", "conectividad",
                      "hardware_pc", "impresion", "microinformatica",
                      "software_pc", "telefonia_fija", "videovigilancia", "backup"],
}

PREGUNTAS = {
    "todas": [
        "Cómo se resuelve cuando EVA no sincroniza?",
        "Cómo se restaura una copia de backup?",
        "Cómo reporto una falla de conexión VPN?",
        "Cómo se restablece la contraseña de un usuario?",
        "Qué hacer si la impresora no imprime?",
        "Cómo se soluciona una falla en IBES?",
    ],
    "aplicaciones": [
        "Cómo se resuelve cuando EVA no sincroniza?",
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se soluciona un error en ONBASE?",
        "Cómo se resuelve una falla de Outlook?",
        "Qué hacer si SNAP no genera notificaciones correctamente?",
        "Cuándo se escala un ticket de aplicaciones a tercer nivel?",
    ],
    "admin_usuarios": [
        "Cómo se crea o modifica un usuario en IBES?",
        "Cómo se desbloquea un usuario?",
        "Cómo se cambia el dominio o correo de un usuario?",
        "Qué se necesita para dar de baja un usuario?",
    ],
    "conectividad": [
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si no hay acceso a una unidad de red?",
        "Cómo se revisa un punto de red que no funciona?",
    ],
    "software_pc": [
        "Cómo se soluciona un error de Office 365?",
        "Cómo se reinstala el sistema operativo de un equipo?",
        "Cómo se instala un software nuevo en un PC?",
    ],
    "microinformatica": [
        "Cómo se restablece la contraseña de un usuario?",
        "Cómo se alista o perfila un equipo nuevo?",
        "Cómo se solicitan herramientas de microinformatica?",
    ],
    "hardware_pc": [
        "Qué hacer si el mouse o teclado fallan?",
        "Cómo se soluciona una falla de pantalla?",
        "Cuándo se programa mantenimiento de un equipo?",
    ],
    "equipos_moviles": [
        "Qué hacer si un celular corporativo tiene fallas?",
        "Cómo se reporta dano en una terminal movil?",
        "Cómo se soluciona una falla en la impresora de etiquetas?",
    ],
    "impresion": [
        "Qué hacer si la impresora no imprime?",
        "Cómo se instala una impresora nueva?",
        "Cómo se configura un escáner?",
    ],
    "backup": [
        "Cómo se restaura una copia de backup?",
        "Cómo se traslada información entre equipos o servidores?",
    ],
    "telefonia_fija": [
        "Qué hacer si el teléfono fijo no funciona?",
        "Cómo se soluciona una falla de llamada?",
        "Cómo se cambia el nombre de una extension?",
    ],
    "videovigilancia": [
        "Cómo se configura una cámara de seguridad?",
    ],
}

PREGUNTAS_TODAS_POR_FILIAL = {
    "recamier": [
        "Cómo se resuelve cuando EVA no sincroniza?",
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si la impresora no imprime?",
        "Qué hacer si un celular corporativo tiene fallas?",
        "Cómo se configura una cámara de seguridad?",
    ],
    "keramer": [
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se restablece la contraseña de un usuario?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si un celular corporativo tiene fallas?",
        "Cómo se desbloquea un usuario?",
    ],
    "lansey": [
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se restablece la contraseña de un usuario?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si el teléfono fijo no funciona?",
        "Cómo se restaura una copia de backup?",
    ],
    "dermodis": [
        "Cómo se resuelve cuando EVA no sincroniza?",
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si un celular corporativo tiene fallas?",
        "Cómo se restablece la contraseña de un usuario?",
    ],
    "fondelar": [
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si la impresora no imprime?",
        "Cómo se restablece la contraseña de un usuario?",
    ],
    "arte_frances": [
        "Qué hacer cuando un usuario no puede ingresar a IBES?",
        "Cómo se resuelve un problema de conexión VPN?",
        "Qué hacer si la impresora no imprime?",
        "Qué hacer si el teléfono fijo no funciona?",
        "Cómo se configura una cámara de seguridad?",
    ],
}


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
    ext = "jpg" if path.endswith((".jpg", ".jpeg")) else "png"
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width};{extra_style}" />'


try:
    favicon = Image.open("app/assets/recamierco-favicon.ico")
    st.set_page_config(page_title="Asistente de Soporte EVA - Recamier",
                        page_icon=favicon, layout="wide", initial_sidebar_state="expanded")
except Exception:
    st.set_page_config(page_title="Asistente de Soporte EVA - Recamier",
                        page_icon="EVA", layout="wide", initial_sidebar_state="expanded")

LOGO_EVA = str(ASSETS / "eva-logo.png")
BANNER = str(ASSETS / "banner-recamier.png")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "sesiones" not in st.session_state:
    sid = st.session_state["session_id"]
    st.session_state["sesiones"] = {sid: {"nombre": "Nueva conversacion", "historial": []}}
    st.session_state["sesion_actual"] = sid
if "pregunta_actual" not in st.session_state:
    st.session_state["pregunta_actual"] = ""
if "filial_actual" not in st.session_state:
    st.session_state["filial_actual"] = "todas"
if "categoria_actual" not in st.session_state:
    st.session_state["categoria_actual"] = "todas"
if "mostrar_uploader" not in st.session_state:
    st.session_state["mostrar_uploader"] = None
if "archivo_adjunto" not in st.session_state:
    st.session_state["archivo_adjunto"] = None
if "tipo_adjunto" not in st.session_state:
    st.session_state["tipo_adjunto"] = None
if "texto_transcrito" not in st.session_state:
    st.session_state["texto_transcrito"] = ""
if "textarea_key" not in st.session_state:
    st.session_state["textarea_key"] = 0
if "limpiar_textarea" not in st.session_state:
    st.session_state["limpiar_textarea"] = False

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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    /* El header se deja visible (no display:none) porque ahi vive el boton
       para reabrir la sidebar si se colapsa. Solo lo hacemos transparente. */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
    }

    /* Sidebar responsive: ancho comodo en pantallas grandes, pero se
       adapta en pantallas angostas (tablet/celular) sin tapar el contenido */
    section[data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 340px !important;
        width: 24vw !important;
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    @media (max-width: 900px) {
        section[data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 85vw !important;
            width: 70vw !important;
        }
    }

    /* El desplegable de Categoria no debe salirse de la pantalla:
       se limita el alto y se agrega scroll INTERNO en varios niveles
       posibles del contenedor (Streamlit/BaseWeb varian segun version) */
    [data-baseweb="popover"] {
        max-height: 340px !important;
    }
    [data-baseweb="menu"],
    [role="listbox"] {
        max-height: 320px !important;
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
    }
    [data-testid="stSidebar"] * {
        color: #2D3748 !important;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 600;
        color: #A0AEC0 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1rem 0 0.5rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #F7FAFC !important;
        color: #2D3748 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 99px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s !important;
        box-shadow: none !important;
        padding: 0.5rem 0.8rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #EBF4FF !important;
        border-color: #87CEEB !important;
        color: #1A6B8A !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover * {
        color: #1A3A5C !important;
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
        max-height: 260px;
        object-fit: contain;
    }

    .hero-section {
        background: transparent;
        padding: 0.5rem 0 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .hero-section h1 {
        color: #1A1A2E;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 1rem;
    }
    .hero-eva-logo { display: flex; justify-content: center; }

    .status-ok {
        background: #F0FFF4;
        border: 1px solid #9AE6B4;
        border-left: 4px solid #38A169;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        color: #276749;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-err {
        background: #FFF5F5;
        border: 1px solid #FEB2B2;
        border-left: 4px solid #E53E3E;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        color: #C53030;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1A1A2E;
        margin: 1.2rem 0 0.6rem;
    }

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

    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: #7C4DFF !important;
        color: white !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 99px !important;
        box-shadow: 0 3px 12px rgba(124,77,255,0.35) !important;
        transition: background 0.2s, transform 0.2s !important;
        padding: 0.55rem 1rem !important;
        text-align: center !important;
    }
    .stButton > button[kind="primary"] *,
    [data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: #87CEEB !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"]:hover *,
    [data-testid="stBaseButton-primary"]:hover * {
        color: #1A3A5C !important;
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
        margin-top: 2rem;
        padding: 1.2rem;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }

    [data-baseweb="select"] > div {
        background-color: #F7FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        color: #2D3748 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
    }
    [data-baseweb="select"] * { cursor: pointer !important; }
    [data-baseweb="select"] span { color: #2D3748 !important; }
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        cursor: pointer !important;
    }
    [role="option"]:hover,
    [aria-selected="true"] {
        background-color: #F0EBFF !important;
        color: #7C4DFF !important;
    }

    /* Iconos compactos de adjuntos (Imagen / PDF / Voz) */
    .modal-btn-icon {
        text-align: center;
        margin-bottom: 2px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .modal-btn-icon img {
        height: 22px;
        object-fit: contain;
    }
    .adjunto-preview {
        display: flex; align-items: center; gap: 10px;
        background: #EBF4FF;
        border: 1px solid #87CEEB;
        border-radius: 10px; padding: 0.6rem 1rem;
        margin-bottom: 0.5rem; font-size: 0.82rem; color: #1A6B8A;
    }
    .adjunto-preview span { flex: 1; }

    /* File uploader / audio input en tema claro (por defecto es oscuro) */
    [data-testid="stFileUploaderDropzone"] {
        background: #FAFAFA !important;
        border: 1.5px dashed #E2E8F0 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #4A5568 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: #87CEEB !important;
    }
    [data-testid="stAudioInput"] {
        background: #FAFAFA !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
    }

    .adjunto-preview {
        display: flex; align-items: center; gap: 10px;
        background: #F0FAFF;
        border: 1px solid #87CEEB;
        border-radius: 10px; padding: 0.6rem 1rem;
        margin-bottom: 0.5rem; font-size: 0.82rem; color: #1A6B8A;
    }
    .adjunto-preview span { flex: 1; }

    .modal-btn-icon {
        text-align: center;
        margin-bottom: 4px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .modal-btn-icon img {
        height: 30px;
        object-fit: contain;
    }

    /* Botones Imagen/PDF/Voz: celeste normal, azul marino cuando esta activo
       (aislados del esquema morado de Consultar/Companias via st.container key).
       Rectangulo redondeado, NUNCA circulo — por eso el radio es fijo en px
       y no en 99px (que en un boton casi cuadrado se ve como circulo). */
    [data-testid="column"]:has(.st-key-cont_btn_imagen),
    [data-testid="column"]:has(.st-key-cont_btn_pdf),
    [data-testid="column"]:has(.st-key-cont_btn_voz) {
        min-width: 130px !important;
        flex: 0 0 130px !important;
    }
    .st-key-cont_btn_imagen .stButton > button,
    .st-key-cont_btn_pdf .stButton > button,
    .st-key-cont_btn_voz .stButton > button {
        background: #87CEEB !important;
        color: #1A3A5C !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        font-size: 0.8rem !important;
        padding: 0.45rem 0.6rem !important;
    }
    .st-key-cont_btn_imagen .stButton > button:hover,
    .st-key-cont_btn_pdf .stButton > button:hover,
    .st-key-cont_btn_voz .stButton > button:hover {
        background: #6FBEDE !important;
        color: #1A3A5C !important;
    }
    .st-key-cont_btn_imagen .stButton > button[kind="primary"],
    .st-key-cont_btn_pdf .stButton > button[kind="primary"],
    .st-key-cont_btn_voz .stButton > button[kind="primary"] {
        background: #1A3A5C !important;
        border-radius: 10px !important;
    }
    .st-key-cont_btn_imagen .stButton > button[kind="primary"] *,
    .st-key-cont_btn_pdf .stButton > button[kind="primary"] *,
    .st-key-cont_btn_voz .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }

    /* Evitar que los botones en fila (compañías, FAQs, Imagen/PDF/Voz) se
       apilen verticalmente en pantallas angostas; se encogen en vez de
       apilarse, para que el layout se vea consistente en cualquier tamaño.
       El texto del boton NUNCA se parte en varias lineas. */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }
    [data-testid="stHorizontalBlock"] [data-testid="column"] {
        min-width: 70px !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    @media (max-width: 600px) {
        [data-testid="stHorizontalBlock"] .stButton > button {
            font-size: 0.72rem !important;
            padding: 0.4rem 0.4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_rec_tag = get_img_tag(
        str(ASSETS / "recamier-logo.png"),
        width="140px",
        extra_style="filter:brightness(0); margin-bottom:1.5rem; display:block;"
    )
    if logo_rec_tag:
        st.markdown(logo_rec_tag, unsafe_allow_html=True)
    else:
        st.markdown("**Recamier**")

    st.markdown('<div class="sidebar-section">Conversaciones</div>', unsafe_allow_html=True)

    if st.button("+ Nueva conversacion", use_container_width=True, key="nueva_sidebar"):
        nuevo_sid = str(uuid.uuid4())[:8]
        st.session_state["sesiones"][nuevo_sid] = {"nombre": "Nueva conversacion", "historial": []}
        st.session_state["sesion_actual"] = nuevo_sid
        st.session_state["session_id"] = nuevo_sid
        st.session_state["pregunta_actual"] = ""
        st.rerun()

    st.markdown("---")

    for sid, datos in list(st.session_state["sesiones"].items()):
        if not datos["historial"]:
            continue
        if st.button(datos["nombre"], key=f"ses_{sid}", use_container_width=True):
            st.session_state["sesion_actual"] = sid
            st.session_state["session_id"] = sid
            st.rerun()

    st.markdown('<div class="sidebar-section">Compañías</div>', unsafe_allow_html=True)
    for key, fl in FILIALES.items():
        activo = st.session_state["filial_actual"] == key
        if st.button(fl["nombre"], key=f"fil_{key}", use_container_width=True,
                     type="primary" if activo else "secondary"):
            st.session_state["filial_actual"] = key
            st.rerun()

    st.markdown('<div class="sidebar-section">Categoría</div>', unsafe_allow_html=True)
    opciones_categoria = FILIAL_CATEGORIAS.get(st.session_state["filial_actual"], list(CATEGORIAS.keys()))
    if st.session_state["categoria_actual"] not in opciones_categoria:
        st.session_state["categoria_actual"] = "todas"

    categoria_sel = st.selectbox(
        "Categoria",
        options=opciones_categoria,
        format_func=lambda k: f"▸  {CATEGORIAS[k]['nombre']}",
        index=opciones_categoria.index(st.session_state["categoria_actual"]),
        label_visibility="collapsed",
    )
    if categoria_sel != st.session_state["categoria_actual"]:
        st.session_state["categoria_actual"] = categoria_sel
        st.rerun()

# ---------------------------------------------------------------------------
# CONTENIDO PRINCIPAL
# ---------------------------------------------------------------------------
banner_tag = get_img_tag(BANNER, width="100%", extra_style="display:block;")
if banner_tag:
    st.markdown(banner_tag, unsafe_allow_html=True)

# Status del sistema: ya NO se verifica cargando todo el motor RAG en cada
# vista de la pagina (eso obligaba a importar torch/embeddings de entrada,
# haciendo lenta la carga inicial incluso antes de hacer una pregunta).
# Si algo falla de verdad, se muestra al momento de consultar (ver el
# try/except del boton "Consultar" mas abajo).
st.markdown('<div class="status-ok">Sistema listo</div>', unsafe_allow_html=True)

# Historial conversacion actual
historial_actual = st.session_state["sesiones"].get(
    st.session_state["sesion_actual"], {}).get("historial", [])

if historial_actual:
    st.markdown('<div class="section-title">Conversacion</div>', unsafe_allow_html=True)
    for msg in historial_actual:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">{msg["content"]}'
                f'<div class="chat-meta">{msg.get("latencia", "")}s . {MODELO_ACTIVO}</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Preguntas frecuentes (dinamicas segun compania + categoria activas)
filial_actual = st.session_state["filial_actual"]
categoria_actual = st.session_state["categoria_actual"]
cat_info = CATEGORIAS[categoria_actual]

if categoria_actual == "todas" and filial_actual != "todas":
    ejemplos = PREGUNTAS_TODAS_POR_FILIAL.get(filial_actual, PREGUNTAS["todas"])
    titulo_faq = f"Preguntas frecuentes - {FILIALES[filial_actual]['nombre']}"
else:
    ejemplos = PREGUNTAS[categoria_actual]
    titulo_faq = f"Preguntas frecuentes - {cat_info['nombre']}"

st.markdown(f'<div class="section-title">{titulo_faq}</div>', unsafe_allow_html=True)

cols = st.columns(3)
for i, ejemplo in enumerate(ejemplos):
    if cols[i % 3].button(ejemplo, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_actual"] = ejemplo
        st.session_state["texto_transcrito"] = ejemplo
        st.session_state["textarea_key"] += 1
        st.rerun()

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tu consulta</div>', unsafe_allow_html=True)

if st.session_state.get("limpiar_textarea", False):
    st.session_state["pregunta_actual"] = ""
    st.session_state["texto_transcrito"] = ""
    st.session_state["textarea_key"] += 1
    st.session_state["limpiar_textarea"] = False

_texto_inicial = st.session_state.get("texto_transcrito", "") or st.session_state.get("pregunta_actual", "")

pregunta = st.text_area(
    label="Pregunta",
    placeholder=cat_info["placeholder"],
    height=110,
    label_visibility="collapsed",
    key=f"textarea_principal_{st.session_state['textarea_key']}",
    value=_texto_inicial,
)

# Botones multimodal compactos: icono + boton, en una fila corta
icon_img_b64 = img_to_base64(str(ASSETS / "icon-imagen.png"))
icon_pdf_b64 = img_to_base64(str(ASSETS / "icon-pdf.png"))
icon_aud_b64 = img_to_base64(str(ASSETS / "icon-audio.png"))

col_b1, col_b2, col_b3, col_spacer = st.columns([1, 1, 1, 9])

with col_b1:
    if icon_img_b64:
        st.markdown(f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_img_b64}"/></div>',
                     unsafe_allow_html=True)
    with st.container(key="cont_btn_imagen"):
        activo_img = st.session_state["mostrar_uploader"] == "imagen"
        if st.button("Imagen", key="btn_imagen", use_container_width=True, help="Adjuntar imagen",
                     type="primary" if activo_img else "secondary"):
            st.session_state["mostrar_uploader"] = None if activo_img else "imagen"
            st.session_state["archivo_adjunto"] = None
            st.session_state["tipo_adjunto"] = None

with col_b2:
    if icon_pdf_b64:
        st.markdown(f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_pdf_b64}"/></div>',
                     unsafe_allow_html=True)
    with st.container(key="cont_btn_pdf"):
        activo_pdf = st.session_state["mostrar_uploader"] == "documento"
        if st.button("PDF", key="btn_doc", use_container_width=True, help="Adjuntar documento PDF",
                     type="primary" if activo_pdf else "secondary"):
            st.session_state["mostrar_uploader"] = None if activo_pdf else "documento"
            st.session_state["archivo_adjunto"] = None
            st.session_state["tipo_adjunto"] = None

with col_b3:
    if icon_aud_b64:
        st.markdown(f'<div class="modal-btn-icon"><img src="data:image/png;base64,{icon_aud_b64}"/></div>',
                     unsafe_allow_html=True)
    with st.container(key="cont_btn_voz"):
        activo_voz = st.session_state["mostrar_uploader"] == "audio"
        if st.button("Voz", key="btn_audio", use_container_width=True, help="Grabar mensaje de voz",
                     type="primary" if activo_voz else "secondary"):
            st.session_state["mostrar_uploader"] = None if activo_voz else "audio"
            st.session_state["archivo_adjunto"] = None
            st.session_state["tipo_adjunto"] = None

if st.session_state["mostrar_uploader"] == "imagen":
    archivo = st.file_uploader("Selecciona una imagen (maximo 2 MB)",
                                type=["jpg", "jpeg", "png", "webp", "bmp"],
                                key="uploader_imagen", label_visibility="collapsed")
    if archivo:
        if len(archivo.getvalue()) > 2 * 1024 * 1024:
            st.error("La imagen supera el limite de 2 MB. Usa una imagen mas pequena.")
        else:
            st.session_state["archivo_adjunto"] = archivo
            st.session_state["tipo_adjunto"] = "imagen"

elif st.session_state["mostrar_uploader"] == "documento":
    archivo = st.file_uploader("Selecciona un PDF (maximo 2 MB)",
                                type=["pdf"], key="uploader_doc", label_visibility="collapsed")
    if archivo:
        if len(archivo.getvalue()) > 2 * 1024 * 1024:
            st.error("El archivo supera el limite de 2 MB. Usa un PDF mas pequeno.")
        else:
            st.session_state["archivo_adjunto"] = archivo
            st.session_state["tipo_adjunto"] = "documento"

elif st.session_state["mostrar_uploader"] == "audio":
    audio_grabado = st.audio_input("Graba tu mensaje de voz", key="grabador_audio")
    if audio_grabado:
        with st.spinner("Transcribiendo tu mensaje de voz..."):
            try:
                from src.multimodal.audio import transcribe_audio
                audio_grabado.seek(0)
                trans_result = transcribe_audio(audio_grabado.read(), filename="audio.wav")
                if trans_result.get("success") and trans_result.get("text"):
                    st.session_state["texto_transcrito"] = trans_result["text"]
                    st.session_state["textarea_key"] += 1
                    st.session_state["mostrar_uploader"] = None
                    st.rerun()
                else:
                    st.error(f"No se pudo transcribir: {trans_result.get('error', 'Audio no reconocido')}")
            except Exception as e:
                st.error(f"Error al transcribir: {str(e)}")

if st.session_state["archivo_adjunto"]:
    archivo_adj_prev = st.session_state["archivo_adjunto"]
    tipo_prev = st.session_state["tipo_adjunto"]
    col_prev, col_rm = st.columns([10, 1])
    with col_prev:
        st.markdown(f"""
        <div class="adjunto-preview">
            <span>{archivo_adj_prev.name}</span>
            <span style="color:#87CEEB;font-size:0.7rem;">
                {round(len(archivo_adj_prev.getvalue()) / 1024, 1)} KB
            </span>
        </div>
        """, unsafe_allow_html=True)
        if tipo_prev == "imagen":
            try:
                img_preview = Image.open(archivo_adj_prev)
                st.image(img_preview, width=200)
                archivo_adj_prev.seek(0)
            except Exception:
                pass
    with col_rm:
        if st.button("X", key="rm_adj", help="Quitar archivo"):
            st.session_state["archivo_adjunto"] = None
            st.session_state["tipo_adjunto"] = None
            st.session_state["mostrar_uploader"] = None
            st.rerun()

consultar = st.button("Consultar", type="primary", use_container_width=True)

archivo_adj = st.session_state.get("archivo_adjunto")
tipo_adj = st.session_state.get("tipo_adjunto")
hay_texto = bool(pregunta and pregunta.strip())
hay_archivo = archivo_adj is not None

if consultar and not hay_texto and not hay_archivo:
    st.warning("Escribe una pregunta o adjunta un archivo antes de consultar.")

elif consultar and (hay_texto or hay_archivo):
    filial_sel = st.session_state["filial_actual"]
    categoria_sel_actual = st.session_state["categoria_actual"]
    filial_nombre = FILIALES[filial_sel]["nombre"] if filial_sel != "todas" else None
    categoria_nombre = CATEGORIAS[categoria_sel_actual]["nombre"] if categoria_sel_actual != "todas" else None

    with st.spinner("Buscando en la base de conocimiento..."):
        try:
            resultado = None
            contenido_usuario = ""

            if hay_archivo and tipo_adj == "audio":
                from src.multimodal.audio import transcribe_audio
                archivo_adj.seek(0)
                transcripcion = transcribe_audio(archivo_adj.read(), filename=archivo_adj.name)
                if not transcripcion["success"]:
                    st.error(f"No se pudo transcribir el audio: {transcripcion['error']}")
                    st.stop()
                texto_audio = transcripcion["text"]
                resultado = responder(f"[Voz transcrita] {texto_audio}", st.session_state["session_id"],
                                       filial=filial_nombre, categoria=categoria_nombre)
                contenido_usuario = f"[Audio: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            elif hay_archivo and tipo_adj == "imagen":
                from src.multimodal.image import extract_text_from_image, describe_image_context
                archivo_adj.seek(0)
                ocr_result = extract_text_from_image(archivo_adj.read(), filename=archivo_adj.name)
                texto_ocr = ocr_result["text"] if ocr_result["success"] else ""
                contexto = describe_image_context(texto_ocr, filename=archivo_adj.name)
                if pregunta.strip():
                    contexto += f"\n\nPregunta del usuario: {pregunta.strip()}"
                resultado = responder(contexto, st.session_state["session_id"],
                                       filial=filial_nombre, categoria=categoria_nombre)
                contenido_usuario = f"[Imagen: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            elif hay_archivo and tipo_adj == "documento":
                from src.multimodal.document import extract_text_from_pdf, describe_document_context
                archivo_adj.seek(0)
                pdf_result = extract_text_from_pdf(archivo_adj.read(), filename=archivo_adj.name)
                if not pdf_result["success"]:
                    st.error(f"No se pudo procesar el PDF: {pdf_result['error']}")
                    st.stop()
                contexto = describe_document_context(pdf_result["text"], filename=archivo_adj.name,
                                                       ocr_used=pdf_result.get("ocr_used", False))
                if pregunta.strip():
                    contexto += f"\n\nPregunta del usuario: {pregunta.strip()}"
                resultado = responder(contexto, st.session_state["session_id"],
                                       filial=filial_nombre, categoria=categoria_nombre)
                contenido_usuario = f"[PDF: {archivo_adj.name}]"
                if hay_texto:
                    contenido_usuario += f" {pregunta.strip()}"

            else:
                prefijo = ""
                if filial_sel != "todas":
                    prefijo += f"[Compania: {FILIALES[filial_sel]['nombre']}] "
                if categoria_sel_actual != "todas":
                    prefijo += f"[Categoria: {CATEGORIAS[categoria_sel_actual]['nombre']}] "
                pregunta_enriquecida = f"{prefijo}{pregunta.strip()}" if prefijo else pregunta.strip()
                resultado = responder(pregunta_enriquecida, st.session_state["session_id"],
                                       filial=filial_nombre, categoria=categoria_nombre)
                contenido_usuario = pregunta.strip()

            if resultado and "respuesta" in resultado:
                sid = st.session_state["sesion_actual"]
                if sid not in st.session_state["sesiones"]:
                    st.session_state["sesiones"][sid] = {"nombre": "Nueva conversacion", "historial": []}

                st.session_state["sesiones"][sid]["historial"].append({"role": "user", "content": contenido_usuario})
                st.session_state["sesiones"][sid]["historial"].append({
                    "role": "assistant", "content": resultado["respuesta"],
                    "latencia": resultado["latencia_segundos"]})

                if len(st.session_state["sesiones"][sid]["historial"]) == 2:
                    st.session_state["sesiones"][sid]["nombre"] = contenido_usuario[:30]

                st.session_state["limpiar_textarea"] = True
                st.session_state["archivo_adjunto"] = None
                st.session_state["tipo_adjunto"] = None
                st.session_state["mostrar_uploader"] = None

                st.markdown(f"""
                <div class="respuesta-box">
                    <div class="respuesta-label">Respuesta del Asistente EVA</div>
                    <div class="respuesta-text">{resultado['respuesta']}</div>
                </div>
                <div class="metrics-row">
                    <div class="metric-card">
                        <span class="metric-val">{resultado['latencia_segundos']}s</span>
                        <span class="metric-lbl">Tiempo de respuesta</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">13.254</span>
                        <span class="metric-lbl">Tickets analizados</span>
                    </div>
                    <div class="metric-card">
                        <span class="metric-val">{MODELO_ACTIVO}</span>
                        <span class="metric-lbl">Modelo activo</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()

        except Exception as e:
            st.error(f"Ocurrio un error generando la respuesta: {e}")

st.markdown("""
<div class="footer">
    Recamier . Transformacion Digital . 2026
</div>
""", unsafe_allow_html=True)