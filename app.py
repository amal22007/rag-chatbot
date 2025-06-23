from fastapi import FastAPI, Query
from pydantic import BaseModel
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load data and models once
with open("clean_btech_faq_dataset_2000.json") as f:
    data = json.load(f)

documents = [item['answer'] for item in data]
questions = [item['question'] for item in data]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

app = FastAPI()

# Intent classification
def classify_intent(query):
    q = query.lower().strip()
    if any(greet in q for greet in ['hi', 'hello', 'hey', 'hai']):
        return 'greeting'
    if 'thank' in q:
        return 'thanks'
    if len(q) < 5:
        return 'unknown'
    return 'faq'

def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [documents[i] for i in I[0]], doc_embeddings[I[0][0]], q_emb[0]

@app.get("/ask")
def ask(query: str = Query(...)):
    intent = classify_intent(query)

    if intent == 'greeting':
        return {"answer": "Hello! How can I help you with BTech admissions today?"}
    if intent == 'thanks':
        return {"answer": "You're welcome! Ask me anything else about university admissions."}
    if intent == 'unknown':
        return {"answer": "Could you please rephrase your question clearly?"}

    retrieved_docs, top_doc_emb, query_emb = retrieve(query)
    sim_score = cosine_similarity([query_emb], [top_doc_emb])[0][0]

    if sim_score < 0.4:
        return {"answer": "I couldn't understand your question. Please check and rephrase it."}

    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    output = gen_pipeline(prompt, max_length=256)[0]["generated_text"]
    return {"answer": output}
