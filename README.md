# Academic PDF Search Assistant

A powerful and efficient semantic search tool built to index and query large volumes of academic PDFs. This project leverages FAISS vector stores, SentenceTransformers for embedding, and Google Gemini for natural language responses. Ideal for university course catalogs, syllabus PDFs, or large-scale document search tasks.

---

## Features

- Automatically parses and chunks academic PDFs using `PyMuPDF` and `LangChain`.
- Embeds and stores chunks in a `FAISS` vector index for efficient semantic search.
- Uses Google Gemini to answer user queries based on retrieved documents.
- De-duplicates and filters search results for relevance and source diversity.
- Modular and extensible — built with clean architecture for future additions.

---

## File Structure
```
.
├── data/ # Place your academic PDFs here
├── faiss_index_multi_new/ # Saved FAISS index
├── ext_faiss2.py # Index builder: loads and vectorizes PDFs
├── query.py # Query engine: retrieves and prompts Gemini
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-search-assistant.git
cd pdf-search-assistant
```

---
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Add Academic PDFs
Place all academic/course PDFs you want to index in the data/ directory.

### Set Up Environment Variables
Create a .env file in the project root with the following content:
```
Gemini_api_key=your_google_gemini_api_key
```
---
### Usage
Step 1: Build the Vector Index
Run the following script to parse, split, embed, and index all PDFs in the data/ folder:

```bash
python ext_faiss2.py
```

Step 2: Run a Query
After the index is built and saved locally, run:
```bash
python query.py
```
This will:

- Load the local FAISS vector store
- Perform a similarity search using your query
- Filter top matching documents for diversity and quality
- Generate an answer using the Gemini API
-Print the final answer along with the source PDF and page numbers

---
Configuration:

- Embedding Model: all-mpnet-base-v2 (from SentenceTransformers)
- Chunk Size: 850 characters
- Chunk Overlap: 50 characters
- FAISS Backend: CPU version
- Gemini Model: gemini-2.5-flash (via google-generativeai)
- Compute: All embeddings are processed on CPU for compatibility
