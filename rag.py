# rag.py
import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from yandex_cloud_ml_sdk import YCloudML
import os

VECTORSTORE_FILE = "./vectorstore.faiss"
METADATA_FILE = "./vectorstore_meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

index = faiss.read_index(VECTORSTORE_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)


token = os.getenv("YANDEX_TOKEN")
sdk = YCloudML(folder_id="b1go6qinn0muj8gb8k4o", auth=token)
yc_model = sdk.models.completions("yandexgpt-32k", model_version="latest")
yc_model = yc_model.configure(temperature=0.3)


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = TOP_K

def query_rag(question, k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    context = ""
    for i in I[0]:
        chunk = metadata[i]["text_snippet"]
        context += f"{chunk}\n\n"
    return context

def ask_yandex(question):
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

@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_yandex(req.question)
    return {"answer": answer}
