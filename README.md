# Asistente de Soporte Recamier

Sistema de Recuperacion Aumentada por Generacion (RAG) que responde preguntas
de soporte tecnico interno usando el historial real de tickets de HubSpot de
Recamier S.A. y sus filiales (Recamier, Dermodis, Lansey, Keramer, Arte
Frances, Fondelar). Cubre todas las categorias de soporte registradas:
aplicaciones corporativas (IBES, ONBASE, Outlook, SNAP, BI4WEB, Teams,
e-commerce, entre otras), conectividad y VPN, administracion de usuarios,
backups, hardware y software de equipos de computo, equipos moviles,
impresion, telefonia fija y videovigilancia.

Este documento explica, paso a paso y sin dar nada por sabido, como instalar,
configurar y ejecutar el proyecto de manera local, y como desplegarlo en
Streamlit Community Cloud.

## Tabla de contenido

1. Que hace este proyecto
2. Arquitectura general
3. Estructura de carpetas
4. Requisitos previos
5. Instalacion paso a paso
6. Configuracion de Supabase
7. Procesamiento de los datos de tickets
8. Ejecucion local
9. Despliegue en Streamlit Community Cloud
10. Mantenimiento: agregar nuevos tickets
11. Preguntas frecuentes de configuracion

---

## 1. Que hace este proyecto

Un agente de soporte, o cualquier persona autorizada de Recamier, escribe una
pregunta en lenguaje natural (por ejemplo, "como se resuelve un problema de
conexion VPN"), opcionalmente adjunta una imagen, un audio o un documento PDF,
y opcionalmente filtra la busqueda por compania y por categoria. El sistema
busca los tickets historicos mas parecidos a la pregunta dentro de una base
vectorial, y con esa informacion como contexto genera una respuesta redactada
en tono profesional, citando la solucion documentada en casos reales, en
lugar de inventar una respuesta generica.

El sistema no reemplaza al equipo de soporte: cuando no encuentra informacion
suficiente en los tickets historicos, lo indica explicitamente y recomienda
contactar al equipo tecnico, en vez de arriesgarse a dar una respuesta
incorrecta.

## 2. Arquitectura general

El proyecto se compone de las siguientes piezas, todas coordinadas desde la
aplicacion de Streamlit:

**Interfaz de usuario.** Streamlit es la unica interfaz necesaria para usar
el sistema. Llama directamente a la logica de Python del proyecto, sin pasar
por una API intermedia, lo que simplifica el despliegue: basta con un solo
proceso corriendo.

**Generacion de texto.** Mistral AI, a traves de su API en la nube (modelo
`mistral-small-latest`), redacta la respuesta final a partir del contexto
recuperado y del historial de la conversacion.

**Embeddings (representacion vectorial del texto).** El modelo
`paraphrase-multilingual-mpnet-base-v2`, de la libreria `sentence-transformers`,
corre de forma local dentro del mismo proceso de Python. No requiere una
clave de API ni un servidor externo, lo que permite desplegar el proyecto en
un entorno con recursos limitados, como el nivel gratuito de Streamlit
Community Cloud.

**Base de datos vectorial y memoria de conversaciones.** Supabase, un
servicio de base de datos Postgres alojado en la nube con la extension
`pgvector`, cumple dos funciones:

- Almacena el contenido de cada fragmento de ticket junto con su vector de
  embedding, permitiendo busquedas de similitud semantica filtradas por
  compania y por categoria.
- Almacena el historial de cada conversacion (tabla `conversaciones`), lo que
  da memoria persistente al asistente sin depender de un archivo local. Esto
  es indispensable para el despliegue en la nube, ya que Streamlit Community
  Cloud no garantiza almacenamiento en disco persistente entre reinicios de
  la aplicacion.

**Orquestacion del flujo de preguntas y respuestas.** LangChain conecta los
componentes anteriores: recupera los fragmentos de ticket relevantes,
construye el mensaje enviado al modelo de lenguaje, y aplica el filtro real
por compania y categoria en la consulta a Supabase.

**Trazabilidad.** MLflow registra cada consulta y cada construccion del
indice vectorial, permitiendo auditar tiempos de respuesta, parametros
usados y volumen de datos procesados.

**Procesamiento de contenido multimodal.** Cuando el usuario adjunta una
imagen, un documento PDF o un mensaje de voz, el sistema extrae el texto
correspondiente antes de pasarlo por el mismo pipeline de preguntas y
respuestas:

- Audio: transcripcion con `faster-whisper`.
- Imagenes: reconocimiento optico de caracteres con `EasyOCR`.
- Documentos PDF: extraccion de texto con `PyMuPDF`, con reconocimiento
  optico de caracteres como respaldo si el PDF contiene paginas escaneadas.

**API REST (componente opcional).** El proyecto incluye tambien una API
construida con FastAPI (`api/main.py`), que expone los mismos endpoints de
consulta, incluyendo los multimodales, bajo un contrato HTTP. No es necesaria
para usar la interfaz de Streamlit, que llama directamente a la logica de
Python, pero queda disponible para integraciones futuras con otros sistemas,
como la aplicacion movil de ventas u otras herramientas internas.

### Diagrama de arquitectura

```
                              USUARIO
                                 |
                                 v
                 +---------------------------------+
                 |     APLICACION DE STREAMLIT      |
                 |   (interfaz, sidebar, adjuntos)  |
                 +---------------------------------+
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
   +-------------------+  +-------------+  +----------------------+
   |  MODULOS MULTI-    |  | LANGCHAIN   |  |  MLFLOW (tracking)   |
   |  MODALES           |  | (orquesta   |  |  registra cada       |
   |  faster-whisper    |  |  el flujo   |  |  consulta y cada     |
   |  EasyOCR           |  |  RAG)       |  |  construccion del    |
   |  PyMuPDF           |  |             |  |  indice vectorial    |
   +-------------------+  +-------------+  +----------------------+
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
   +-----------------------+          +--------------------------+
   |  SENTENCE-TRANSFORMERS |          |      MISTRAL AI (nube)   |
   |  (embeddings, local,   |          |      genera la respuesta |
   |  sin API key)          |          |      final en lenguaje   |
   +-----------------------+          |      natural              |
              |                       +--------------------------+
              v                                     ^
   +-----------------------------------------------+|
   |                  SUPABASE (nube)                |
   |   Postgres + extension pgvector                 |
   |                                                  |
   |   tabla documents        tabla conversaciones    |
   |   (tickets embebidos,    (historial de cada      |
   |    filtrables por        sesion de chat,         |
   |    compania/categoria)   memoria persistente)    |
   +--------------------------------------------------+
              ^
              |
   +-----------------------------------------------+
   |         DATOS DE ORIGEN (fuera del sistema)     |
   |   Exportes de HubSpot (.xlsx / .csv)            |
   |   procesados por src/ingest.py y subidos con    |
   |   src/rag_chain.py -> build_vectorstore()       |
   +-----------------------------------------------+
```

La API REST (`api/main.py`, FastAPI) es un componente opcional que no
aparece en este diagrama porque no participa en el flujo principal de la
aplicacion de Streamlit: se conecta a los mismos modulos de `src/` de forma
independiente, unicamente cuando se necesita exponer el sistema por HTTP a
otra aplicacion externa.

### Diagrama de flujo de una consulta

```
Usuario escribe una pregunta (o adjunta imagen/PDF/audio) en Streamlit
        |
        v
Si hay archivo adjunto, se extrae el texto (OCR o transcripcion)
        |
        v
El texto de la pregunta se convierte en un vector de embedding
(sentence-transformers, local)
        |
        v
Se buscan los tickets mas similares en Supabase (pgvector),
aplicando el filtro de compania y categoria si el usuario lo eligio
        |
        v
Si el filtro no encuentra resultados, se repite la busqueda sin filtro
        |
        v
El contexto recuperado, el historial de la conversacion (desde Supabase)
y la pregunta se envian a Mistral AI
        |
        v
La respuesta generada se muestra al usuario y se guarda en Supabase
```

## 3. Estructura de carpetas

```
genai-rag-eva/
    api/
        main.py                    API REST opcional (FastAPI)
    app/
        streamlit_app.py           Interfaz principal de usuario
        assets/                    Logos, banner e iconos
    data/
        raw/                       Archivos originales de tickets (.xlsx o .csv)
        processed/                 Tickets combinados y limpios (.jsonl)
    scripts/
        generar_filial_categorias.py   Calcula que categorias aplican a cada
                                        compania, a partir de los datos reales
    src/
        config.py                  Configuracion central y variables de entorno
        ingest.py                  Combina y limpia los archivos de tickets
        rag_chain.py                Pipeline RAG completo (recuperacion,
                                        generacion, memoria)
        memory.py                   Memoria de conversaciones en Supabase
        multimodal/
            audio.py                Transcripcion de audio
            image.py                 Extraccion de texto de imagenes
            document.py               Extraccion de texto de documentos PDF
        evaluate.py                 Evaluacion de calidad del sistema RAG
    reports/
        evaluation/                 Resultados de evaluaciones
    tests/
        test_api.py                  Pruebas automatizadas
    .streamlit/
        config.toml                 Configuracion de tema y limites de Streamlit
    .env.example                    Plantilla de variables de entorno
    requirements.txt                Dependencias de Python
    README.md
```

Los archivos `.env`, `data/raw/*.xlsx`, `data/raw/*.csv` y
`data/processed/*.jsonl` estan excluidos del control de versiones mediante
`.gitignore`, ya que contienen informacion real de tickets y credenciales.
Nunca deben subirse al repositorio.

## 4. Requisitos previos

- Python 3.11
- El gestor de paquetes `uv` (o `pip`, si se prefiere)
- Git
- Una cuenta de Mistral AI (nivel gratuito disponible en
  console.mistral.ai)
- Una cuenta de Supabase (nivel gratuito disponible en supabase.com)
- Una cuenta de GitHub, para el despliegue en Streamlit Community Cloud

## 5. Instalacion paso a paso

### 5.1. Clonar el repositorio

```powershell
git clone https://github.com/dimarior/genai-rag-eva.git
cd genai-rag-eva
```

### 5.2. Crear el entorno virtual e instalar dependencias

```powershell
uv python install 3.11
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

La instalacion puede tardar varios minutos, ya que incluye librerias de
aprendizaje automatico de tamano considerable, como `torch` y
`sentence-transformers`, necesarias para el procesamiento multimodal y los
embeddings locales.

### 5.3. Crear el archivo de variables de entorno

Copia la plantilla y crea tu archivo real:

```powershell
copy .env.example .env
```

Abre `.env` y completa los siguientes valores (los de Supabase se explican
en la seccion 6):

```
MISTRAL_API_KEY=
MISTRAL_MODEL=mistral-small-latest
EMBEDDING_MODEL_NAME=paraphrase-multilingual-mpnet-base-v2
SUPABASE_URL=
SUPABASE_KEY=
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
WHISPER_MODEL=small
```

La clave de Mistral se obtiene de forma gratuita en console.mistral.ai. El
nivel gratuito no requiere tarjeta de credito e incluye una cuota mensual de
tokens generosa para un proyecto de este tamano.

Este archivo `.env` nunca debe compartirse ni subirse a un repositorio de
codigo, ya que contiene credenciales privadas.

## 6. Configuracion de Supabase

Supabase reemplaza tanto a la base de datos vectorial local como al archivo
de memoria de conversaciones, y es indispensable para poder desplegar el
proyecto en la nube.

### 6.1. Crear el proyecto

1. Registrate en supabase.com.
2. Crea un nuevo proyecto, eligiendo una region cercana a Colombia.
3. Guarda la contraseña de la base de datos que definas durante la creacion.
4. Espera a que el proyecto termine de aprovisionarse, lo cual toma uno o
   dos minutos.

### 6.2. Obtener las credenciales

Dentro del proyecto, ve a Project Settings:

- En la seccion Data API, copia el valor de API URL. Este valor va en la
  variable `SUPABASE_URL` del archivo `.env`.
- En la seccion API Keys, copia la Secret key (no la Publishable key, que
  esta pensada para exponerse en un navegador). Este valor va en la variable
  `SUPABASE_KEY`.

### 6.3. Crear las tablas y funciones necesarias

En el panel de Supabase, ve a SQL Editor, crea una nueva consulta, pega el
siguiente script completo y ejecutalo:

```sql
create extension if not exists vector;

create table documents (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(768)
);

create index on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create function match_documents (
    query_embedding vector(768),
    match_count int default null,
    filter jsonb default '{}'
) returns table (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where documents.metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;

create table conversaciones (
    id bigserial primary key,
    session_id text not null,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now()
);

create index idx_session on conversaciones (session_id, created_at);
```

La dimension 768 corresponde al modelo de embeddings usado en este
proyecto. Si en el futuro se cambia de modelo de embeddings, este numero
debe actualizarse tanto en la tabla `documents` como en la funcion
`match_documents`, y el indice vectorial debe reconstruirse desde cero.

## 7. Procesamiento de los datos de tickets

### 7.1. Copiar los archivos de tickets

Coloca los archivos exportados de HubSpot, en formato `.xlsx` o `.csv`,
dentro de la carpeta `data/raw/`. El sistema procesa automaticamente todos
los archivos que encuentre en esa carpeta, sin importar el nombre ni el ano
al que correspondan, siempre que las columnas coincidan con las esperadas o
con alguno de sus nombres alternativos configurados en `src/ingest.py`.

### 7.2. Combinar y limpiar los datos

```powershell
python -m src.ingest
```

Este paso lee todos los archivos de `data/raw/`, descarta los tickets que no
tengan una solucion documentada por el agente, elimina duplicados por
identificador de ticket entre archivos que se solapen, y genera un unico
archivo combinado en `data/processed/tickets_recamier.jsonl`.

### 7.3. Construir el indice vectorial en Supabase

```powershell
python -c "from src.rag_chain import build_vectorstore; build_vectorstore()"
```

Este proceso calcula el embedding de cada fragmento de texto y lo sube a la
tabla `documents` de Supabase. Con un volumen de tickets de varios miles de
registros, este paso puede tardar entre media hora y una hora, dependiendo
de la velocidad de la conexion a internet, ya que cada lote de fragmentos
requiere una llamada de red a Supabase. El proceso muestra el progreso y una
estimacion del tiempo restante en la terminal, y no debe interrumpirse antes
de que finalice.

## 8. Ejecucion local

### 8.1. Servidor de MLflow (opcional, para trazabilidad)

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

### 8.2. Aplicacion de Streamlit

```powershell
streamlit run app/streamlit_app.py
```

La aplicacion se abre automaticamente en el navegador, normalmente en
`http://localhost:8501`. La primera vez que se ejecuta, el modelo de
embeddings se descarga y se guarda en cache local, lo que puede tardar unos
minutos adicionales.

### 8.3. API REST (opcional)

Solo necesaria si se requiere exponer el sistema a otra aplicacion mediante
HTTP:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
```

La documentacion interactiva de la API queda disponible en
`http://127.0.0.1:8080/docs`.

## 9. Despliegue en Streamlit Community Cloud

### 9.1. Preparar el repositorio

Antes de subir el proyecto a GitHub, confirma que el archivo `.gitignore`
excluye los siguientes elementos, ya que no deben quedar publicos:

```
.env
data/raw/*.xlsx
data/raw/*.csv
data/processed/
```

Sube los cambios:

```powershell
git add .
git commit -m "Preparar proyecto para despliegue"
git push
```

### 9.2. Conectar el repositorio a Streamlit Community Cloud

1. Ingresa a share.streamlit.io con tu cuenta de GitHub.
2. Selecciona "New app".
3. Elige el repositorio, la rama, y como archivo principal
   `app/streamlit_app.py`.
4. Antes de desplegar, ve a la seccion de configuracion avanzada para
   agregar las variables de entorno secretas (ver siguiente paso).

### 9.3. Configurar las variables de entorno secretas

Streamlit Community Cloud no lee el archivo `.env` local, ya que ese archivo
nunca se sube al repositorio. En su lugar, las credenciales se configuran
dentro del panel de la aplicacion, en la seccion Secrets, usando el mismo
formato que un archivo `.env`:

```
MISTRAL_API_KEY = "tu_key_real"
MISTRAL_MODEL = "mistral-small-latest"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SUPABASE_URL = "tu_url_real"
SUPABASE_KEY = "tu_key_real"
WHISPER_MODEL = "small"
```

No es necesario configurar `MLFLOW_TRACKING_URI` en la nube, ya que MLflow
es una herramienta de uso local para desarrollo.

### 9.4. Desplegar

Al confirmar el despliegue, Streamlit Community Cloud instala las
dependencias del archivo `requirements.txt` y arranca la aplicacion. El
primer arranque puede tardar varios minutos debido al tamano de las
dependencias de aprendizaje automatico. Como el indice vectorial ya vive en
Supabase, la aplicacion no necesita reconstruirlo al iniciar: se conecta
directamente a la base de datos ya poblada.

## 10. Mantenimiento: agregar nuevos tickets

Cuando llegue un archivo de tickets nuevo, por ejemplo el correspondiente a
un nuevo trimestre o un nuevo ano:

1. Copia el archivo nuevo en `data/raw/`, sin borrar los anteriores.
2. Vuelve a correr `python -m src.ingest`, que combina todos los archivos y
   elimina duplicados automaticamente.
3. Vuelve a correr el paso de construccion del indice vectorial. Los
   fragmentos que ya existan en Supabase se agregan de nuevo; para evitar
   duplicados en cargas repetidas, se recomienda revisar el conteo de filas
   en la tabla `documents` antes y despues del proceso.
4. Si la distribucion de categorias por compania cambio de forma
   significativa, ejecuta `python scripts/generar_filial_categorias.py` para
   recalcular que categorias debe mostrar el selector de cada compania en la
   interfaz, con base en los datos reales actualizados.

## 11. Preguntas frecuentes de configuracion

**El proceso de construccion del indice vectorial parece congelado.**
No lo esta: al subir los datos a Supabase, cada lote de fragmentos requiere
una peticion de red, por lo que el progreso se ve mas lento que al usar una
base de datos local. La terminal muestra el avance y una estimacion de
tiempo restante en cada lote procesado; mientras esos numeros sigan
avanzando, el proceso continua con normalidad.

**Aparece un aviso sobre peticiones no autenticadas a HuggingFace Hub.**
Es un aviso informativo de la libreria `sentence-transformers` al verificar
la version del modelo de embeddings. No afecta el funcionamiento del
sistema y puede ignorarse con seguridad.

**La aplicacion no encuentra la sidebar o parece haber perdido elementos de
la interfaz.**
Verifica que el navegador no tenga un estado de sesion antiguo guardado;
una recarga forzada del navegador (Ctrl+Shift+R en Windows) suele resolver
inconsistencias visuales tras actualizar el codigo.

**Quiero cambiar el modelo de generacion de texto.**
El modelo se configura en la variable `MISTRAL_MODEL` del archivo `.env` (o
de los secretos de Streamlit Cloud, en produccion). Cambiarlo no requiere
modificar el codigo del proyecto.