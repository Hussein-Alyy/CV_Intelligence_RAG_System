# 🧠 CV Intelligence RAG System

An AI-powered resume analysis and comparison system built using **Retrieval-Augmented Generation (RAG)**.

The system allows you to upload multiple CVs, convert them into embeddings, store them in a vector database (FAISS), and interactively ask questions using Google Gemini models.

---

## 🚀 Features

* 📄 Upload and process multiple CVs
* 🔍 Vector search using FAISS
* 🧠 Semantic embeddings with Google Generative AI
* 🤖 LLM-powered analysis using Gemini
* 📊 Intelligent candidate comparison
* 💬 Interactive Streamlit chat interface
* 🗂 Metadata tracking per candidate
* 🔁 Persistent vector storage
* 🧾 Structured HR-style evaluation responses

---

## 🏗 Architecture

1. **PDF Parsing** → Extract text from CVs
2. **Chunking** → Split into overlapping segments
3. **Embeddings** → Generate vector representations
4. **FAISS Indexing** → Store and retrieve similar chunks
5. **RAG Pipeline** → Retrieve relevant context
6. **LLM Generation** → Generate expert-level answers

---

## 🛠 Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Google Generative AI (Gemini)
* PyPDF2
* dotenv

---

## 📁 Project Structure

```
CV_Chat_Project/
│
├── CVS/                  # Folder containing 5 CV PDFs
├── application.py        # Main Streamlit app
├── faiss_cv_index/       # Generated vector database
├── .env
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add your API key

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

### 3️⃣ Add CVs

Place exactly **5 PDF CVs** inside the `CVS/` folder.

### 4️⃣ Run the app

```bash
streamlit run application.py
```

---

## 🧩 Example Use Cases

* Compare multiple candidates
* Rank applicants for a data science role
* Analyze skills across CVs
* Identify the best candidate for a job description
* Extract structured insights from resumes

---

## 🧠 What Makes This Project Advanced?

* Custom RAG implementation (not basic RetrievalQA)
* Candidate-aware metadata filtering
* Structured comparison prompting
* Controlled chunking strategy
* Persistent vector storage
* HR-level analytical reasoning

---

## 📌 Future Improvements

* Add reranking layer
* Implement hybrid search (BM25 + embeddings)
* Add evaluation metrics
* Deploy on cloud (GCP / AWS)
* Add authentication layer
