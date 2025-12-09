import os
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret")


## Global in-memory state (per server instance) 
vectorizer: TfidfVectorizer | None = None
tfidf_matrix = None
documents: list[dict] | None = None
qa_model = None


# Load QA model lazily

def get_qa_model():
    global qa_model
    if qa_model is None:
        # Small, free seq2seq model (runs on our cpu )
        qa_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )
    return qa_model



## File -> text extraction

def extract_text_from_file(file_storage) -> str:
    """Accepts a Werkzeug FileStorage, returns raw text."""
    filename = file_storage.filename or ""
    name = filename.lower()
    ext = name.split(".")[-1]

    file_bytes = BytesIO(file_storage.read())
    file_bytes.seek(0)

    if ext == "pdf":
        reader = PdfReader(file_bytes)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif ext == "txt":
        data = file_bytes.read()
        return data.decode("utf-8", errors="ignore")

    elif ext == "csv":
        df = pd.read_csv(file_bytes)
        return df.to_csv(index=False)

    elif ext == "docx":
        doc = DocxDocument(file_bytes)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        return ""



## Chunking into pieces

def chunk_text(text: str, source_name: str,
               chunk_size_words: int = 400,
               overlap_words: int = 50) -> list[dict]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text_str,
            "metadata": {
                "source": source_name,
                "chunk_id": chunk_id,
            }
        })

        chunk_id += 1
        start += chunk_size_words - overlap_words

    return chunks



## Build TF-IDF index in memory

def build_tfidf_index(doc_chunks):
    texts = [d["text"] for d in doc_chunks]

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words="english"
    )
    matrix = vec.fit_transform(texts)

    return vec, matrix



## QA over uploaded docs

def answer_question(question: str,
                    top_k: int = 5,
                    min_similarity: float = 0.05):
    global vectorizer, tfidf_matrix, documents

    if vectorizer is None or tfidf_matrix is None or documents is None:
        return "No documents indexed yet. Please upload and index documents first.", []

    qa = get_qa_model()

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_indices = np.argsort(-sims)[:top_k]

    valid_hits = [(int(i), float(sims[i])) for i in top_indices if sims[i] >= min_similarity]

    if not valid_hits:
        return "I don't have enough information in the uploaded documents.", []

    context_parts = []
    sources = []
    for idx, score in valid_hits:
        doc = documents[idx]
        context_parts.append(doc["text"])
        sources.append({
            "source": doc["metadata"]["source"],
            "chunk_id": doc["metadata"]["chunk_id"],
            "score": score,
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a question-answering assistant.\n"
        "Use ONLY the information in the provided context to answer the question.\n"
        "If the answer cannot be found in the context, reply exactly with:\n"
        "\"I don't have enough information in the uploaded documents.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        result = qa(
            prompt,
            max_new_tokens=128,
            do_sample=False
        )[0]["generated_text"].strip()
    except Exception as e:
        return f"Local model error: {e}", sources

    if not result:
        result = "I don't have enough information in the uploaded documents."

    return result, sources


## Routes

@app.route("/", methods=["GET"])
def index():
    # Show if index is ready
    has_index = documents is not None and vectorizer is not None
    return render_template("index.html", has_index=has_index)


@app.route("/upload", methods=["POST"])
def upload():
    global vectorizer, tfidf_matrix, documents

    files = request.files.getlist("documents")
    if not files or (len(files) == 1 and files[0].filename == ""):
        flash("Please select at least one document.", "error")
        return redirect(url_for("index"))

    all_chunks: list[dict] = []

    for f in files:
        filename = secure_filename(f.filename)
        if not filename:
            continue
        text = extract_text_from_file(f)
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)

    if not all_chunks:
        flash("No readable text found in the uploaded documents.", "error")
        return redirect(url_for("index"))

    vectorizer, tfidf_matrix = build_tfidf_index(all_chunks)
    documents = all_chunks

    flash(f"Indexed {len(all_chunks)} chunks from {len(files)} file(s).", "success")
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        flash("Please enter a question.", "error")
        return redirect(url_for("index"))

    answer, sources = answer_question(question)

  
    has_index = documents is not None and vectorizer is not None
    return render_template(
        "index.html",
        has_index=has_index,
        question=question,
        answer=answer,
        sources=sources
    )


if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=8000, debug=True)
