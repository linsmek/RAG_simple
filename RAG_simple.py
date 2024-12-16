#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
import os
import tempfile
import ollama
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# Enhanced system prompt
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

# Initialize search history
if "history" not in st.session_state:
    st.session_state.history = []

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

def get_vector_collection(space: str) -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": space},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str, space: str):
    collection = get_vector_collection(space)
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success(f"Data from {file_name} added to the vector store!")

def query_collection(prompt: str, space: str, n_results: int = 10):
    collection = get_vector_collection(space)
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str, history: list[dict]):
    # Include history in the prompt
    history_text = "\n\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )
    full_context = f"{history_text}\n\n{context}"
    
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
                "content": f"Context: {full_context}\n\nQuestion: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer")

    with st.sidebar:
        st.header("🗣️ RAG Question Answer")
        # Input for dynamic chunk size
        chunk_size = st.number_input(
            "**Set Chunk Size (characters):**", min_value=100, max_value=2000, value=400, step=100
        )
        chunk_overlap = int(chunk_size * 0.2)  # 20% of chunk size

        # Select distance metric
        space = st.selectbox(
            "**Choose Distance Metric:**",
            options=["cosine", "euclidean", "dot"],
            index=0
        )

        uploaded_files = st.file_uploader(
            "**📑 Upload PDF files for QnA**",
            type=["pdf"],
            accept_multiple_files=True,
        )
        process = st.button("⚡️ Process All")

        if uploaded_files and process:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )
                all_splits = process_document(uploaded_file, chunk_size, chunk_overlap)
                add_to_vector_collection(all_splits, file_name, space)
    
    st.header("🗣️ RAG Question Answer")
    user_prompt = st.text_area("**Ask a question related to your documents:**")
    ask = st.button("🔥 Ask")

    if ask and user_prompt:
        results = query_collection(user_prompt, space)
        context_docs = results.get("documents", [[]])[0]

        if not context_docs:
            st.write("No documents available. Please upload and process PDFs first.")
        else:
            # Concatenate all retrieved documents into a single context
            concatenated_context = "\n\n".join(context_docs)

            # Include search history in the call to LLM
            placeholder = st.empty()
            full_response = ""
            response_stream = call_llm(
                context=concatenated_context, prompt=user_prompt, history=st.session_state.history
            )
            for r in response_stream:
                full_response += r
                placeholder.markdown(full_response)

            # Update search history
            st.session_state.history.append({"question": user_prompt, "answer": full_response})

            # Display search history
            with st.expander("Search History"):
                for entry in st.session_state.history:
                    st.write(f"**Q:** {entry['question']}\n**A:** {entry['answer']}\n---")

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document IDs"):
                st.write(results.get("ids", [[]])[0])
                st.write(concatenated_context)

