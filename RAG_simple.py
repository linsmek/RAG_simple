#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
import os
import tempfile
import ollama
import requests
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import CrossEncoder

from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

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

# Custom Ollama Embeddings class
class OllamaEmbeddings(Embeddings):
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        payload = {"prompt": text, "model": self.model_name}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        # Ensure your Ollama endpoint returns embeddings in a field named "embedding"
        return data["embedding"]

# Initialize embeddings and FAISS vector store once
EMBEDDINGS = OllamaEmbeddings(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

FAISS_INDEX_PATH = "./faiss_index"  # path where we'll store/load the FAISS index


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
    # If a FAISS index already exists, load it; otherwise return a new empty store.
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS)
    else:
        # Return an empty FAISS store (we'll add documents to it later)
        return FAISS.from_texts([], EMBEDDINGS)


def save_vectorstore(vectorstore: FAISS):
    # Save the FAISS index locally
    vectorstore.save_local(FAISS_INDEX_PATH)


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    vectorstore = load_vectorstore()

    # Extract the text and metadata from the documents
    texts = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata for doc in all_splits]

    # Add the new texts to the FAISS vectorstore
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=[f"{file_name}_{i}" for i in range(len(texts))])

    save_vectorstore(vectorstore)
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    vectorstore = load_vectorstore()
    # Use similarity_search on FAISS
    docs = vectorstore.similarity_search(prompt, k=n_results)
    # Convert docs to a format similar to what you used before
    # docs is a list of Documents, so let's return just the texts for now.
    return docs


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
    # CrossEncoder's rank method expects two parameters: query and documents
    # The returned object typically contains scores or ranks. Check the CrossEncoder docs.
    # If `rank()` isn't available, you might have to sort by scores manually.
    scores = encoder_model.predict([(query, d) for d in documents])
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
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")
    
    if ask and prompt:
        results = query_collection(prompt)
        # results is a list of Documents
        context_docs = [doc.page_content for doc in results]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context_docs, prompt)
        response = call_llm(context=relevant_text, prompt=prompt)

        # Stream the response to the UI
        for r in response:
            st.write(r, end="")

        with st.expander("See retrieved documents"):
            for doc in results:
                st.write(doc.page_content)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)

