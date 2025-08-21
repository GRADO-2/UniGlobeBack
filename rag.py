import pickle
import faiss
import tempfile
import os
import json
from google.cloud import storage
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from yandex_cloud_ml_sdk import YCloudML

# Configuración - ¡USA TU BUCKET REAL!
BUCKET_NAME = "rag-vectorstore-1755763542"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Variables globales que se inicializarán después
index = None
metadata = None
model = None
yc_model = None

# Configurar autenticación de Google Cloud
if "GOOGLE_CREDENTIALS_JSON" in os.environ:
    creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
    with open("/tmp/credentials.json", "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# Inicializar GCS
storage_client = storage.Client()

def load_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Descarga un archivo desde GCS"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

# Cargar datos al iniciar
def load_vectorstore():
    with tempfile.TemporaryDirectory() as tmp_dir:
        faiss_path = os.path.join(tmp_dir, "vectorstore.faiss")
        meta_path = os.path.join(tmp_dir, "vectorstore_meta.pkl")
        
        print("Descargando FAISS index desde GCS...")
        load_from_gcs(BUCKET_NAME, "vectorstore.faiss", faiss_path)
        
        print("Descargando metadata desde GCS...")
        load_from_gcs(BUCKET_NAME, "vectorstore_meta.pkl", meta_path)
        
        index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        print("Vectorstore cargado exitosamente!")
        return index, metadata

def initialize_app():
    """Inicializar todos los componentes de la app"""
    global index, metadata, model, yc_model
    
    print("Iniciando carga de vectorstore...")
    index, metadata = load_vectorstore()
    model = SentenceTransformer(MODEL_NAME)
    
    # Configuración de Yandex
    token = os.getenv("YANDEX_TOKEN")
    if token:
        sdk = YCloudML(folder_id="b1go6qinn0muj8gb8k4o", auth=token)
        yc_model = sdk.models.completions("yandexgpt-32k", model_version="latest")
        yc_model = yc_model.configure(temperature=0.3)
        print("Yandex GPT configurado exitosamente!")
    else:
        print("⚠️  YANDEX_TOKEN no encontrado")

app = FastAPI()

# Event handler para inicializar al startup
@app.on_event("startup")
async def startup_event():
    initialize_app()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = TOP_K

def query_rag(question, k=TOP_K):
    if index is None or metadata is None:
        return "Error: Vectorstore no cargado"
    
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    context = ""
    for i in I[0]:
        chunk = metadata[i]["text_snippet"]
        context += f"{chunk}\n\n"
    return context

def ask_yandex(question):
    if yc_model is None:
        return "Error: Yandex GPT no configurado"
    
    context = query_rag(question)
    behavior = """
    You are an expert assistant specialized in topics related to settling abroad as an international student.
    Answer all questions in English clearly and concisely. Use the context provided to give accurate advice.
    """
    user_input = f"Context:\n{context}\n\nQuestion: {question}"
    result = yc_model.run(
        [
            {"role": "system", "text": behavior},
            {"role": "user", "text": user_input}
        ]
    )
    return result.alternatives[0].text if result else ""

@app.get("/")
def root():
    return {"status": "RAG + Yandex GPT service running"}

@app.get("/health")
def health_check():
    """Endpoint para verificar el estado de la app"""
    status = {
        "vectorstore_loaded": index is not None,
        "model_loaded": model is not None,
        "yandex_configured": yc_model is not None
    }
    return status

@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_yandex(req.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, reload=False)
