
# Imports
import glob
from pathlib import Path
import PyMuPDF as fitz                             
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


# Minimal wrapper around Sentence-Transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.encode(texts, batch_size=32)]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

embedder = SentenceTransformerEmbeddings()


# Locate every PDF in ./data
pdf_dir   = Path("data")
pdf_paths = glob.glob(str(pdf_dir / "*.pdf"))
if not pdf_paths:
    raise FileNotFoundError("No PDF files found in the 'data' folder.")
print(f"Found {len(pdf_paths)} PDF(s)")


# Configure an adaptive splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=850,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


# Load, chunk, and tag every file
documents = []
for pdf_path in pdf_paths:
    with fitz.open(pdf_path) as doc:              
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            for chunk in splitter.split_text(text): 
                header = f"[FILE: {Path(pdf_path).name} | PAGE: {page_no}]\n"
                text_for_embed = header + chunk
                documents.append(
                    Document(
                        page_content = text_for_embed,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_no, 
                        }
                    )
                )

print(f"Prepared {len(documents)} chunks")
print("Vecotrizing Now...")


# Build & persist FAISS index
vector_store = FAISS.from_documents(documents, embedder)
vector_store.save_local("faiss_index_multi_new")
print(f"Indexed {len(pdf_paths)} PDFs FAISS index saved to ./faiss_index_multi_new")
