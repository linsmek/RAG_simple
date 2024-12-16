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
from typing import Any
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
import chromadb
from chromadb.config import Settings

# Enhanced system prompt
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

FAISS_INDEX_PATH = "./faiss_index"

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
            raise ValueError("Expected 'embedding' in response. Got: {}".format(data))
        return data["embedding"]

EMBEDDINGS = OllamaEmbeddings(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

def process_document(uploaded_file: Any, chunk_size: int, chunk_overlap: int) -> list[Document]:
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

def initialize_faiss() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    return None

def initialize_chromadb() -> chromadb.Client:
    # Updated ChromaDB initialization
    settings = Settings(
        persist_directory="./chroma_data",
        chroma_api_impl="duckdb+parquet",
    )
    return chromadb.Client(settings=settings)

def add_to_vector_collection(all_splits: list[Document], vector_store, library: str):
    documents = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata or {} for doc in all_splits]

    if library == "faiss":
        if not vector_store:
            vector_store = FAISS.from_texts(documents, EMBEDDINGS, metadatas=metadatas)
        else:
            vector_store.add_texts(documents, metadatas=metadatas)
        vector_store.save_local(FAISS_INDEX_PATH)
    elif library == "chromadb":
        collection = vector_store.get_or_create_collection(name="rag_app")
        collection.add(documents=documents, metadatas=metadatas)
    st.success("Data has been successfully added to the vector store!")

def query_vector_store(prompt: str, vector_store, library: str, k: int = 10):
    if library == "faiss":
        results_docs = vector_store.similarity_search(prompt, k=k)
        return [doc.page_content for doc in results_docs]
    elif library == "chromadb":
        collection = vector_store.get_or_create_collection(name="rag_app")
        results = collection.query(query_texts=[prompt], n_results=k)
        return results.get("documents", [[]])[0]

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}",
            },
        ],
    )
    return "".join(chunk["message"]["content"] for chunk in response if not chunk["done"])

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")

    with st.sidebar:
        st.header("RAG Question Answer")
        chunk_size = st.number_input("Set Chunk Size", min_value=100, max_value=2000, value=400)
        chunk_overlap = int(chunk_size * 0.2)
        library = st.selectbox("Library", ["faiss", "chromadb"])
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        process = st.button("Process All")

        vector_store = initialize_faiss() if library == "faiss" else initialize_chromadb()

        if process and uploaded_files:
            for file in uploaded_files:
                splits = process_document(file, chunk_size, chunk_overlap)
                add_to_vector_collection(splits, vector_store, library)

    st.header("Ask a Question")
    prompt = st.text_area("Your Question:")
    ask = st.button("Submit")

    if ask and prompt:
        context = query_vector_store(prompt, vector_store, library)
        if not context:
            st.write("No documents available.")
        else:
            response = call_llm(context="\n".join(context), prompt=prompt)
            st.write(response)
