
# 📘 Retrieval-Augmented Generation (RAG) Pipeline

This project demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** system using **LangChain, ChromaDB, and Sentence Transformers**.

The pipeline covers every step from:
✅ **Data ingestion**
✅ **Document loading**
✅ **Chunking**
✅ **Embeddings generation**
✅ **Vector storage**
✅ **Query-based retrieval**

---

## 🚀 Features

* **Data Ingestion & Document Handling**

  * Supports `.txt` and `.pdf` files.
  * Custom metadata support (author, source, date created, etc.).

* **Text Loading**

  * Load individual text files (`TextLoader`).
  * Bulk directory loading (`DirectoryLoader`).
  * PDF support with `PyMuPDFLoader`.

* **Chunking**

  * Smart document splitting with `RecursiveCharacterTextSplitter`.
  * Configurable `chunk_size` and `chunk_overlap` for optimal retrieval.

* **Embeddings**

  * Uses **`all-MiniLM-L6-v2`** from HuggingFace Sentence Transformers.
  * Converts text into high-dimensional vectors for semantic similarity.

* **Vector Store**

  * Powered by **ChromaDB** with persistent storage.
  * Supports adding, managing, and querying embeddings.

* **Retriever**

  * Retrieves the most relevant chunks for a given query.
  * Ranks results by similarity score with configurable thresholds.

---

## 📂 Project Structure

```
RAG_Pipeline_Project/
│── data/
│   ├── text_files/          # Sample text files
│   ├── pdf/                 # PDF documents
│   └── vector_store/        # Persistent ChromaDB storage
│
│── main.py                  # End-to-end RAG pipeline
│── README.md                # Project documentation
```

---

## ⚙️ Workflow

1️⃣ **Data Ingestion** → Load `.txt` / `.pdf` into LangChain `Document` objects with metadata.
2️⃣ **Chunking** → Split documents into smaller chunks using `RecursiveCharacterTextSplitter`.
3️⃣ **Embedding Generation** → Convert chunks into vectors with **SentenceTransformer**.
4️⃣ **Vector Storage** → Store vectors + metadata in **ChromaDB**.
5️⃣ **Retriever** → Query the DB and fetch top-matching chunks.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/VaishnaviSh14/RAG_Pipeline_Project.git
cd RAG_Pipeline_Project
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📜 Example Usage

### 🔹 Load and Chunk Documents

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader("../data/text_files", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(documents)
```

### 🔹 Generate Embeddings & Store in Vector DB

```python
embedding_manager = EmbeddingManager()
texts = [doc.page_content for doc in chunked_documents]
embeddings = embedding_manager.generate_embeddings(texts)

vectorstore = VectorStore()
vectorstore.add_documents(chunked_documents, embeddings)
```

### 🔹 Retrieve Relevant Chunks

```python
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
results = rag_retriever.retrieve("large language models overview")

for res in results:
    print(res["rank"], res["similarity_score"], res["content"][:200])
```

---

## 📊 Example Output

```
Retrieveing docs for query: 'large language models overview'
Top k : 5, score threshold: 0.0
Retrieved 2 documents (after filtering)

1 0.873 LangChain is a powerful framework designed to help developers build applications with large language models (LLMs)...
2 0.841 With LangChain, you can implement Retrieval-Augmented Generation (RAG)...
```

---

## 🔮 Future Improvements

* Integration with **LLMs (OpenAI, Llama2, Gemini, etc.)** for end-to-end RAG answers.
* Support for more file types (CSV, DOCX, JSON).
* REST API / Web UI for easy querying.
* Advanced ranking with rerankers.

---

## 👩‍💻 Author

**Vaishnavi Sharma**
📧 [LinkedIn](https://linkedin.com/in/vaishnavi-sharma)
🌐 GitHub: [VaishnaviSh14](https://github.com/VaishnaviSh14)

---

✨ This project provides a **strong foundation** for building real-world RAG-powered applications like **chatbots, document Q\&A, and knowledge retrieval systems**.

---

Would you like me to also style this with **badges (Python version, LangChain, ChromaDB, HuggingFace, etc.)** at the top so it looks even more professional on GitHub?
