from fastapi import FastAPI, Query
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
with open("clean_btech_faq_dataset_2000.json") as f:
    data = json.load(f)

documents = [item['answer'] for item in data]

# Load small sentence embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(documents)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load tiny text generator (tiny-t5)
gen_pipeline = pipeline("text2text-generation", model="sshleifer/tiny-t5")

# FastAPI app
app = FastAPI()

# Simple intent logic
def classify_intent(query):
    query = query.lower().strip()
    if any(greet in query for greet in ['hi', 'hello', 'hey', 'hai']):
        return "greeting"
    elif "thank" in query:
        return "thanks"
    elif len(query) < 5:
        return "unknown"
    return "faq"

def retrieve(query, top_k=3):
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    return [documents[i] for i in I[0]], doc_embeddings[I[0][0]], query_emb[0]

@app.get("/ask")
def ask(query: str = Query(...)):
    intent = classify_intent(query)

    if intent == "greeting":
        return {"answer": "Hello! How can I help you with BTech admissions today?"}
    elif intent == "thanks":
        return {"answer": "You're welcome! Ask me anything else about university admissions."}
    elif intent == "unknown":
        return {"answer": "Please rephrase your question clearly."}

    retrieved_docs, top_doc_emb, query_emb = retrieve(query)
    sim_score = cosine_similarity([query_emb], [top_doc_emb])[0][0]

    if sim_score < 0.4:
        return {"answer": "I couldn't understand your question. Please check and rephrase it."}

    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    result = gen_pipeline(prompt, max_length=50)[0]['generated_text']
    return {"answer": result}
