📄 Local GPT Assistant (RAG System)

A lightweight Retrieval-Augmented Generation (RAG) application that answers questions strictly from uploaded documents.

⭐ Overview

This project implements a local RAG-style assistant that allows users to upload documents and ask questions.
The system retrieves relevant text chunks from the uploaded files and uses a lightweight local model to generate an answer.
If the answer cannot be found within the provided documents, the system responds:

"I don't have enough information in the uploaded documents."

This ensures zero hallucination, full transparency, and strict document-grounded reasoning — as required in the assignment. 

AI Intern_Jr AI Developer_Asses…

🧠 System Architecture (Approach)
1. Document Upload & Processing

Supports: TXT, PDF, CSV, DOCX

Each file is converted into raw text

Text is split into overlapping chunks (300–500 words) for better retrieval

Each chunk is tagged with:

file name

chunk ID

metadata

2. Embedding & Vector Search

Since the project must run locally without heavy dependencies or paid APIs, the system uses:

TF-IDF vectorization (scikit-learn)

Cosine similarity to find the most relevant chunks

This mimics a classical RAG retrieval pipeline while staying lightweight.

(Note: In production, models like all-mpnet-base-v2 + FAISS would be ideal, as suggested in the assignment. This implementation preserves the same logic without GPU requirements.) 

AI Intern_Jr AI Developer_Asses…

3. Answer Generation

Retrieved chunks are combined into a single context

A free local LLM (google/flan-t5-small) generates the final answer

The model is instructed to use only the retrieved context

If similarity scores are too low → fallback to:

“I don’t have enough information in the uploaded documents.”

4. Simple Web Interface

Built using Flask:

Upload documents

Build index

Ask questions

View answers + source chunks

Exactly as required in the assignment’s UI section. 

AI Intern_Jr AI Developer_Asses…

🔧 Tech Stack

Following the assignment’s recommended technologies: 

AI Intern_Jr AI Developer_Asses…

Component	Technology Used
Language	Python 3.10+
Backend	Flask
Vector Search	TF-IDF + cosine similarity (scikit-learn)
LLM	google/flan-t5-small (local, free)
File Parsing	PyPDF2, python-docx, pandas
Deployment	Vercel (serverless Python)
📂 Example Input & Output
User Uploads:

resume.docx

company_policy.pdf

project_notes.txt

User Question:

"What is the minimum notice period mentioned in the policy?"

System Process:

Retrieve top 3 most similar chunks

Pass context → FLAN-T5

If answer found → return it

If not → return fallback

Example Output:

"The minimum notice period mentioned in the provided policy document is 30 days."

If Out-of-Scope:

"I don’t have enough information in the uploaded documents."

(Exactly as required in the assignment.) 

AI Intern_Jr AI Developer_Asses…

🚫 Out-of-Scope Handling

Per assignment rules:

If similarity between user query and stored chunks falls below the threshold → system must not guess

Instead, it returns:

"I don't have enough information in the uploaded documents." 

AI Intern_Jr AI Developer_Asses…

This eliminates hallucination completely.

▶️ How to Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Run the app
python app.py

3. Open in browser
http://127.0.0.1:8000

☁️ Deployment

The project includes vercel.json and is ready for Vercel Python Serverless deployment, as required for a working demo link submission.
