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
        # Implement the logic to call Ollama's embedding endpoint and return a vector.
        # The endpoint should return a JSON with "embedding": [...]
        # Example:
        # response = requests.post(self.url, json={"prompt": text, "model": self.model_name})
        # data = response.json()
        # return data["embedding"]
        raise NotImplementedError("Implement Ollama embedding retrieval logic here.")

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
    # If the index file exists, load it; otherwise return None to indicate we must create it later.
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

    # If no vectorstore exists yet, create one with these documents.
    if vectorstore is None:
        vectorstore = FAISS.from_texts(documents, EMBEDDINGS, metadatas=metadatas)
    else:
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

    # Mimic Chroma's return format:
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

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 'rank' method usage: 
    # According to sentence-transformers docs, rank() doesn't exist in all versions. 
    # If not available, manually compute scores and sort.
    # Here's a possible fallback:
    # scores = encoder_model.predict([(prompt, doc) for doc in documents])
    # ranked = sorted(enumerate(documents), key=lambda x: scores[x[0]], reverse=True)
    # top_3 = ranked[:3]

    # If 'rank' is available as you had it:
    # ranks = encoder_model.rank(prompt, documents, top_k=3)
    # Adapt as needed if rank() is not a valid method in your environment.

    # Assuming 'rank' works as you wrote:
    global prompt  # Ensure prompt is defined in this scope or pass it as a parameter
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank_info in ranks:
        relevant_text += documents[rank_info["corpus_id"]]
        relevant_text_ids.append(rank_info["corpus_id"])

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
        context_docs = results.get("documents", [[]])[0]

        if not context_docs:
            st.write("No documents available. Please upload and process a PDF first.")
        else:
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context_docs)
            response = call_llm(context=relevant_text, prompt=prompt)

            # Stream the response
            for r in response:
                st.write(r)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

