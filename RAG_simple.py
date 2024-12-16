#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
import os
import tempfile
import requests
import ollama
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import CrossEncoder
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
import chromadb

# Constants
FAISS_INDEX_PATH = "./faiss_index"
CHROMADB_PATH = "./chroma_store"

# System Prompt
system_prompt = """
To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Include all relevant history from previous user questions and responses (if applicable).
3. Organize your response logically, ensuring all parts of the question are addressed.
4. If the context doesn't contain sufficient information, state this clearly, or if relevant, add suggestions based on your knowledge.
Guidelines:
1. For unrelated or vague questions, respond appropriately without referencing documents.
2. If the question relates to a career or general topic (not in the documents), provide thoughtful suggestions, but clearly mention this is not sourced from the documents.
3. If the user asks you about a previous question asked, look at it in your "history," re-state it, and answer it.
Format:
1. Use bullet points, numbered lists, or headings for readability.
2. Ensure responses are structured and concise.

Important: Base your answers solely on the provided context and history, unless instructed otherwise.
"""

# Embedding Function
class OllamaEmbeddings(Embeddings):
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        response = requests.post(
            self.url,
            json={"prompt": text, "model": self.model_name},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if "embedding" not in data:
            raise ValueError(f"Expected 'embedding' in response. Got: {data}")
        return data["embedding"]

EMBEDDINGS = OllamaEmbeddings(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

# Document Processing
def process_document(uploaded_file: UploadedFile, chunk_size: int, chunk_overlap: int) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.flush()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

# Initialize Vector Store
def initialize_vector_store(library: str, distance_metric: str, documents=None):
    if library == "FAISS":
        if os.path.exists(FAISS_INDEX_PATH):
            return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
        elif documents and len(documents) > 0:
            # Initialize FAISS with provided documents
            return FAISS.from_texts(documents, EMBEDDINGS)
        else:
            raise ValueError("FAISS requires at least one document to initialize.")
    elif library == "ChromaDB":
        chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
        return chroma_client.get_or_create_collection(
            name="rag_app",
            embedding_function=EMBEDDINGS,
            metadata={"hnsw:space": distance_metric},
        )
    else:
        raise ValueError(f"Unsupported library: {library}")

# Add to Vector Store
def add_to_vector_collection(all_splits: list[Document], vector_store, library: str):
    documents = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata if doc.metadata else {} for doc in all_splits]

    if library == "FAISS":
        vector_store.add_texts(documents, metadatas=metadatas)
        vector_store.save_local(FAISS_INDEX_PATH)
    elif library == "ChromaDB":
        vector_store.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    else:
        raise ValueError(f"Unsupported library: {library}")
    st.success(f"Data has been stored in the {library} vector store!")

# Query Vector Store
def query_collection(prompt: str, vector_store, library: str, n_results: int = 10):
    if library == "FAISS":
        results = vector_store.similarity_search(prompt, k=n_results)
        documents = [doc.page_content for doc in results]
        return documents
    elif library == "ChromaDB":
        results = vector_store.query(query_texts=[prompt], n_results=n_results)
        return results["documents"]
    else:
        raise ValueError(f"Unsupported library: {library}")

# Call LLM
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

# Main Application
if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")

    with st.sidebar:
        st.header("🗣️ RAG Question Answer")
        library = st.selectbox("Select Vector Store:", ["FAISS", "ChromaDB"], index=0)

        # Choose Distance Metric
        distance_metric = st.selectbox(
            "Choose Distance Metric:",
            options=["cosine", "euclidean", "dot"],
            index=0
        )

        chunk_size = st.number_input("Chunk Size:", min_value=100, max_value=2000, value=400, step=100)
        chunk_overlap = int(chunk_size * 0.2)

        uploaded_files = st.file_uploader("Upload PDFs:", type=["pdf"], accept_multiple_files=True)
        process = st.button("Process Documents")

        if uploaded_files and process:
            documents = []
            for uploaded_file in uploaded_files:
                all_splits = process_document(uploaded_file, chunk_size, chunk_overlap)
                documents.extend([doc.page_content for doc in all_splits])
            vector_store = initialize_vector_store(library, distance_metric, documents=documents)
            add_to_vector_collection(all_splits, vector_store, library)

    st.header("🗣️ Ask a Question")
    user_prompt = st.text_area("Your Question:")
    ask = st.button("Ask")

    if ask and user_prompt:
        vector_store = initialize_vector_store(library, distance_metric)
        results = query_collection(user_prompt, vector_store, library)
        context = "\n\n".join(results)

        response = call_llm(context, user_prompt)
        placeholder = st.empty()
        full_response = ""
        for r in response:
            full_response += r
            placeholder.markdown(full_response)

