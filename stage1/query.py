# Imports
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types, errors

# Load environment variables
load_dotenv()

# Embedding setup
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# API initialization
api_key = os.getenv("Gemini_api_key")
print(f"GOOGLE_API_KEY loaded: {api_key is not None}\n")

ai_model = "gemini-2.5-pro"
client = genai.Client()

# Load vector store
embedding_model = SentenceTransformerEmbeddings("all-mpnet-base-v2")
library = FAISS.load_local("faiss_index_multi2", embedding_model, allow_dangerous_deserialization=True)

# Query setup
query = ""
query_answer = library.similarity_search(query, k=5)

# Format search results
def format_chunk(doc):
    src = Path(doc.metadata['source']).name
    page = doc.metadata['page']
    return f"\n### Source: {src} — page {page}\n{doc.page_content}"

context = "\n".join(format_chunk(doc) for doc in query_answer)

print("CONTEXT:\n", context)
print("CONTEXT OVER\n")

# Prompt template
prompt_template = """
You are a helpful assistant. Use only the provided context to answer the question. Do not use any external knowledge, assumptions, or inferred information.
Answer strictly in the following format:

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

# API call
config = types.GenerateContentConfig(
    temperature=0.2,
    system_instruction="IMPORTANT Answer strictly from these excerpts. If the answer is not present, say “Not found”."
)

try:
    response = client.models.generate_content(
        model=ai_model,
        contents=[prompt_filled],
        config=config
    )
    print("Response:", response.text)
except errors.APIError as e:
    print(e.code)
    print(e.message)
