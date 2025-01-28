#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:15:41 2025

@author: linamekouar
"""

import os
import re
import shutil
import tempfile
import requests
import fitz  # PyMuPDF
import streamlit as st

# LangChain imports
from langchain.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

# Chroma imports
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# For memory and chain usage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------------------------------------------------------------------
# 1) (Optional) Page Config
# ---------------------------------------------------------------------------------
# st.set_page_config(page_title="RAG PDF Chatbot")

# ---------------------------------------------------------------------------------
# 2) System Prompt
# ---------------------------------------------------------------------------------
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. 
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
6. If the question or request has no link whatsoever and is just a general question such as "hi" or "how are you," please use your knowledge to answer it.
7. If the question is kind of related but no information is found in the documents, mention that no information was found, then provide general suggestions if appropriate.
8. If the user asks you about a previous question, refer to the 'history' and re-state it to the user, then answer it again.
9. If the question is about multiple documents, go through each of them to find the answer.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Please respond in a single sentence whenever the question is a simple greeting or short query.

Important: Do not include any external knowledge or assumptions not present in the given text except in (6) or (7) situations. Don't ever hallucinate an answer.
"""

# ---------------------------------------------------------------------------------
# 3) Initialize Chat Messages & Memory
# ---------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are now chatting with a PDF-based RAG assistant. "
                "Feel free to ask questions about your uploaded documents or any general questions."
            )
        }
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Path constants
FAISS_INDEX_PATH = "./faiss_index"
CHROMA_DB_PATH = "./chroma_store"

# ---------------------------------------------------------------------------------
# 4) OllamaEmbeddings Class
# ---------------------------------------------------------------------------------
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

# Instantiate embeddings
EMBEDDINGS = OllamaEmbeddings(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

# ---------------------------------------------------------------------------------
# 5) Automatic Metadata Extraction Helpers
# ---------------------------------------------------------------------------------
def extract_methods_and_results(full_text: str):

    # e.g. from 'Methods' up to 'Results' or EOF
    methods_pattern = r"(?is)(?<=methods)(.*?)(?=results|$)"
    # e.g. from 'Results' up to 'Discussion' or 'Conclusion' or 'References' or EOF
    results_pattern = r"(?is)(?<=results)(.*?)(?=discussion|conclusion|references|$)"

    # Convert to lower for matching, but we store original substring
    text_lower = full_text.lower()

    methods_match = re.search(methods_pattern, text_lower)
    results_match = re.search(results_pattern, text_lower)

    # If we found them, let's slice them out
    methods_text = ""
    if methods_match:
        start, end = methods_match.span()
        methods_text = full_text[start:end].strip()

    results_text = ""
    if results_match:
        start, end = results_match.span()
        results_text = full_text[start:end].strip()

    return methods_text, results_text

def extract_metadata_from_pdf(pdf_path: str):


    
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
    author = meta.get("author", "").strip()
    title = meta.get("title", "").strip()

    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()  

    full_text = "\n".join(d.page_content for d in docs)

    methods_text, results_text = extract_methods_and_results(full_text)

    return author, title, methods_text, results_text

# ---------------------------------------------------------------------------------
# 6) Primary PDF Processing (Automatic Metadata)
# ---------------------------------------------------------------------------------
def process_document_auto_metadata(
    uploaded_file: UploadedFile,
    chunk_size: int,
    chunk_overlap: int
) -> list[Document]:

    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    pdf_path = temp_file.name

    # Extract metadata
    author, title, methods_text, results_text = extract_metadata_from_pdf(pdf_path)

    # Load again for chunking
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Delete the temp file 
    os.unlink(pdf_path)

    # Split the PDF content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    split_docs = text_splitter.split_documents(docs)

    # Attach the found metadata to each chunk
    for doc in split_docs:
        doc.metadata["author"] = author
        doc.metadata["title"] = title
        doc.metadata["method"] = methods_text
        doc.metadata["results"] = results_text

    return split_docs

# ---------------------------------------------------------------------------------
# 7) FAISS Helper Functions
# ---------------------------------------------------------------------------------
def load_faiss_vectorstore() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    return None

def save_faiss_vectorstore(vectorstore: FAISS):
    vectorstore.save_local(FAISS_INDEX_PATH)

# ---------------------------------------------------------------------------------
# 8) ChromaDB Helper Functions
# ---------------------------------------------------------------------------------
def get_chromadb_collection(space: str) -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": space},
    )

# ---------------------------------------------------------------------------------
# 9) Add Documents to Vector Collection
# ---------------------------------------------------------------------------------
def add_to_vector_collection(all_splits: list[Document], file_name: str, space: str, backend: str):
    documents = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata if doc.metadata else {} for doc in all_splits]
    ids = [f"{file_name}_{idx}" for idx in range(len(documents))]

    if backend == "FAISS":
        vectorstore = load_faiss_vectorstore()
        if vectorstore is None:
            vectorstore = FAISS.from_texts(documents, EMBEDDINGS, metadatas=metadatas)
        else:
            vectorstore.add_texts(documents, metadatas=metadatas)
        save_faiss_vectorstore(vectorstore)
    elif backend == "ChromaDB":
        collection = get_chromadb_collection(space)
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    st.success(f"Data from '{file_name}' added to the {backend} vector store!")

# ---------------------------------------------------------------------------------
# 10) Query the Collection
# ---------------------------------------------------------------------------------
def query_collection(prompt: str, space: str, backend: str, n_results: int = 10):
    if backend == "FAISS":
        vectorstore = load_faiss_vectorstore()
        if vectorstore is None:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        results_docs = vectorstore.similarity_search(prompt, k=n_results)
        documents = [doc.page_content for doc in results_docs]
        metadatas = [doc.metadata for doc in results_docs]
        ids = [m.get("id", f"doc_{i}") for i, m in enumerate(metadatas)]
        return {"documents": [documents], "metadatas": [metadatas], "ids": [ids]}
    elif backend == "ChromaDB":
        collection = get_chromadb_collection(space)
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results

# ---------------------------------------------------------------------------------
# 11) Call the LLM with ConversationBufferMemory
# ---------------------------------------------------------------------------------
def call_llm_with_memory(context: str, prompt: str, temperature: float) -> str:
    """
    Calls the Ollama LLM using a ConversationBufferMemory for history.
    The system_prompt is combined with 'context' and 'prompt' inside a PromptTemplate.
    The prior conversation is automatically appended via the memory object.
    """
    # 1) Create the LLM with your chosen temperature
    llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3.2",
        temperature=temperature,
        num_ctx=4000
    )

    # 2) Define a prompt template that merges system prompt, retrieved context, and user query.
    template = """{system_prompt}

Context: {context}

Question: {prompt}
"""
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template=template
    )

    # 3) Build an LLMChain that has a memory object (ConversationBufferMemory).
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=st.session_state.memory,  # conversation buffer
        verbose=True
    )

    # 4) Run the chain
    response = chain.run(
        system_prompt=system_prompt,
        context=context,
        prompt=prompt
    )

    return response

# ---------------------------------------------------------------------------------
# 12) Main Streamlit Application
# ---------------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.header("🗣️ RAG Chat Bot (Auto Metadata)")
        backend = st.selectbox("Choose Backend", ["FAISS", "ChromaDB"], index=0)

        if backend == "ChromaDB":
            space = st.selectbox("Choose Distance Metric:", ["cosine", "euclidean", "dot"], index=0)
        else:
            space = "cosine"

        chunk_size = st.number_input(
            "Set Chunk Size (characters):",
            min_value=100,
            max_value=2000,
            value=400,
            step=100
        )
        chunk_overlap = int(chunk_size * 0.2)

        temperature = st.slider("Model Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

        # File uploader
        uploaded_files = st.file_uploader(
            "**📑 Upload PDF files for Q&A**", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        process_button = st.button("⚡️ Process All")

        # Process PDFs
        if uploaded_files and process_button:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name.replace(" ", "_")
                # Automatically parse metadata (author, title, methods, results)
                all_splits = process_document_auto_metadata(
                    uploaded_file,
                    chunk_size,
                    chunk_overlap
                )
                add_to_vector_collection(all_splits, file_name, space, backend)

    st.title("📚 Chat with your PDF(s) — Automatic Metadata Extraction")

    if st.button("Reset Everything (Chat + Vector Stores)"):
        st.session_state.messages.clear()
        st.session_state.memory.clear()

        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)

        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)

        st.write("Chat history and vector stores have been cleared. Please refresh or ask a new question to start fresh.")

    # Display existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            with st.chat_message("system"):
                st.markdown(msg["content"])
        elif msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents (or anything else)...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query vector store for relevant chunks
        results = query_collection(user_query, space, backend)
        context_docs = results.get("documents", [[]])[0]

        concatenated_context = "\n\n".join(context_docs) if context_docs else ""

        # Call LLM
        assistant_reply = call_llm_with_memory(
            context=concatenated_context,
            prompt=user_query,
            temperature=temperature
        )

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

# ---------------------------------------------------------------------------------
# 13) Run the app
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

