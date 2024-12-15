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
        # Call Ollama embeddings endpoint
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

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.flush()

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def load_vectorstore() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS)
    return None

def save_vectorstore(vectorstore: FAISS):
    vectorstore.save_local(FAISS_INDEX_PATH)

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    vectorstore = load_vectorstore()

    documents = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata if doc.metadata else {} for doc in all_splits]
    ids = [f"{file_name}_{idx}" for idx in range(len(documents))]

    for i, mid in enumerate(ids):
        metadatas[i]["id"] = mid

    if vectorstore is None:
        # Create a new FAISS vectorstore from these texts
        vectorstore = FAISS.from_texts(documents, EMBEDDINGS, metadatas=metadatas)
    else:
        # Add texts to existing vectorstore
        vectorstore.add_texts(documents, metadatas=metadatas)

    save_vectorstore(vectorstore)
    st.success("Data added to the vector store!")

def query_collection(prompt: str, n_results: int = 10):
    vectorstore = load_vectorstore()
    if vectorstore is None:
        # No documents have been added yet
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

    results_docs = vectorstore.similarity_search(prompt, k=n_results)

    documents = [doc.page_content for doc in results_docs]
    metadatas = [doc.metadata for doc in results_docs]
    ids = [m.get("id", f"doc_{i}") for i, m in enumerate(metadatas)]

    # Mimic Chroma's return format
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "ids": [ids],
    }

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
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str], query: str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Compute scores for each document
    pairs = [(query, doc) for doc in documents]
    scores = encoder_model.predict(pairs)
    # Sort documents by score (descending)
    ranked = sorted(enumerate(documents), key=lambda x: scores[x[0]], reverse=True)
    top_3 = ranked[:3]

    for idx, doc_text in top_3:
        relevant_text += doc_text + "\n"
        relevant_text_ids.append(idx)

    return relevant_text, relevant_text_ids

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")

    with st.sidebar:
        st.header("🗣️ RAG Question Answer")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )
        process = st.button("⚡️ Process")

        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    st.header("🗣️ RAG Question Answer")
    user_prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and user_prompt:
        results = query_collection(user_prompt)
        context_docs = results.get("documents", [[]])[0]

        if not context_docs:
            st.write("No documents available. Please upload and process a PDF first.")
        else:
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context_docs, user_prompt)
            response = call_llm(context=relevant_text, prompt=user_prompt)

            # Stream response chunks
            for r in response:
                st.write(r)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

