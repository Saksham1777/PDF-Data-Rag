import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types, errors

# Utils 
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D NumPy vectors (unit-safe)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def filter_hits(
    hits,                       
    embedder,                   
    k: int = 5,
    score_floor: float = 0.5,
    dup_thresh: float = 0.80,
    max_per_src: int = 3,
):
    """
    Return up to k documents that are (1) relevant, (2) not near-duplicates,
    (3) not dominated by a single PDF.
    """
    keep, seen_src, cache = [], {}, {}

    def emb(doc):
        # use chunk_id if present; else fallback to source:page
        key = doc.metadata.get("chunk_id") or f"{doc.metadata['source']}:{doc.metadata['page']}"
        if key not in cache:
            cache[key] = np.asarray(embedder.embed_query(doc.page_content), dtype=np.float32)
        return cache[key]

    for doc, score in hits:
        if score < score_floor:
            continue

        src = doc.metadata["source"]
        if seen_src.get(src, 0) >= max_per_src:
            continue

        e = emb(doc)
        if any(cosine(e, emb(d)) >= dup_thresh for d in keep):
            continue

        keep.append(doc)
        seen_src[src] = seen_src.get(src, 0) + 1
        if len(keep) == k:
            break
    return keep


# Embedding wrapper
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()


# Load env & models
load_dotenv()
api_key = os.getenv("Gemini_api_key2")
print(f"GOOGLE_API_KEY loaded: {api_key is not None}\n")

embedding_model = SentenceTransformerEmbeddings("all-mpnet-base-v2")
vector_store = FAISS.load_local(
    "faiss_index_multi_new",
    embedding_model,
    allow_dangerous_deserialization=True,
)

client = genai.Client()
ai_model = "gemini-2.5-flash"

# User query 
query = "tell me the course textbook name of bits f312."

# Retrieve extra neighbours, then filter down
raw_hits = vector_store.similarity_search_with_score(query, k=5 * 4)
top_docs = filter_hits(raw_hits, embedding_model, k=10)

# Build context 
def fmt(doc):
    src = Path(doc.metadata["source"]).name
    pg = doc.metadata["page"]
    return f"\n### Source: {src} — page {pg}\n{doc.page_content}"

context = "".join(fmt(d) for d in top_docs)
print("CONTEXT:\n", context, "\nCONTEXT END\n")

# Prompt 
prompt_template  = """
You are an academic course-catalog assistant.

Task:
1. Read the numbered context snippets.
2. Decide which snippet(s) fully answer the question.
3. Copy or paraphrase only what is needed.

question: <repeat the question>
answer: <answer extracted only from the relevant course information; write "Not found" if not available>
source: <mention the PDF file name and page number(s) where the information was found, or "Not found" if unavailable>

Examples:

question: what is the make up policy  
answer: Students must inform the instructor within 3 days to schedule a makeup exam.  
source: course_policies.pdf, page 2

question: who is the Instructor-in-charge/Professor in course - MA101?  
answer: Dr. R. Kumar  
source: MA101_syllabus.pdf, page 1

question: what are the textbooks for course CS102?  
answer: Introduction to Algorithms by Cormen et al.  
source: CS102_outline.pdf, page 3

IMPORTANT - Answer strictly from these excerpts. If the answer is not present, say “Not found”
---
Context : {context}
Query : {query}
"""
prompt_filled = prompt_template.format(context=context, query=query)
system_instructions = "IMPORTANT Answer strictly from these excerpts. If the answer is not present, say “Not found”."
#api calling
cfg = types.GenerateContentConfig(
    temperature=0.2,
    top_p=0.7,
    system_instruction=system_instructions
)
try: 
    response = client.models.generate_content(
        model=ai_model,
        contents=[prompt_filled],
        config=cfg
    )
    print("Response:",response.text)
except errors.APIError as e:
    print(e.code)
    print(e.message)

    
