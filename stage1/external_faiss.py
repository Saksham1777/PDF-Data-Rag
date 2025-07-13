# Imports
import glob
from pathlib import Path
from langchain.text_splitter import PythonCodeTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pymupdf as fitz 
from pypdf import PdfReader

# Embedding class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Initialize embedding model
embedding_model = SentenceTransformerEmbeddings("all-mpnet-base-v2")

# Collect PDF paths
pdf_folder = Path("data")
pdf_paths  = glob.glob(str(pdf_folder / "*.pdf"))

if not pdf_paths:
    raise FileNotFoundError("No PDF files found in the 'data' folder.")

# Read and chunk PDFs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
documents = []

for pdf_path in pdf_paths:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    chunks = splitter.split_text(full_text)
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": str(pdf_path)}
        )
        for chunk in chunks
    ]
    documents.extend(docs)

print("Files read")

# Build and save FAISS index
library = FAISS.from_documents(documents=documents, embedding=embedding_model)
library.save_local("faiss_index_multi")

print(f"Indexed {len(pdf_paths)} PDFs — FAISS index stored at 'faiss_index_multi'")
