
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit PDF RAG App with Ollama-based Metadata Extraction

Instructions:
1. Install necessary packages:
   pip install streamlit langchain chromadb faiss-cpu pymupdf requests

2. Run:
   streamlit run this_script.py
"""

import os
import shutil
import json
import tempfile
import requests
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from typing import List, Dict

# LangChain imports
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

# Streamlit imports
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Vector stores
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

# ---------------------------------------------------------------------------------
# 1) System Prompt (for final answers)
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
# 2) Directory and Global Constants
# ---------------------------------------------------------------------------------
FAISS_INDEX_PATH = "./faiss_index"
CHROMA_DB_PATH = "./chroma_store"

# ---------------------------------------------------------------------------------
# 3) Ollama Embeddings Class
# ---------------------------------------------------------------------------------
class OllamaEmbeddings(Embeddings):
    """
    Simple embeddings class that calls Ollama's /api/embeddings endpoint.
    """
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
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

# Instantiate embeddings globally (adjust model_name to your actual embedding model)
EMBEDDINGS = OllamaEmbeddings(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)

# ---------------------------------------------------------------------------------
# 4) LLM-based Function to Extract Metadata from a Paragraph
# ---------------------------------------------------------------------------------
def extract_paragraph_metadata(llm: Ollama, paragraph_text: str) -> Dict[str, str]:
    """
    Calls Ollama to identify a likely section title (e.g., Introduction) and produce a short summary.
    Returns a dict: {"section_title": "...", "summary": "..."}
    """
    prompt = f"""
You are a helpful assistant. 
You will analyze the paragraph below and determine:
1. A likely section title or label (e.g., "Introduction", "Background", "Methods", "Discussion").
2. A concise summary of the paragraph content.

Return the result as valid JSON with keys "section_title" and "summary" only.

Paragraph:
\"\"\"{paragraph_text}\"\"\"
"""

    response = llm(prompt)  # response is a string

    # Attempt to parse JSON from the response
    try:
        metadata_dict = json.loads(response)
        # Ensure keys exist
        if "section_title" not in metadata_dict:
            metadata_dict["section_title"] = "Unknown"
        if "summary" not in metadata_dict:
            metadata_dict["summary"] = ""
    except json.JSONDecodeError:
        # If not valid JSON, fallback
        metadata_dict = {
            "section_title": "Unknown",
            "summary": ""
        }

    return metadata_dict

# ---------------------------------------------------------------------------------
# 5) Process a PDF and Split into Chunks, Using Ollama for Metadata
# ---------------------------------------------------------------------------------
def process_document(uploaded_file: UploadedFile, chunk_size: int, chunk_overlap: int, llm: Ollama) -> List[Document]:
    """
    1) Load PDF with PyMuPDFLoader.
    2) Split into paragraphs/chunks with RecursiveCharacterTextSplitter.
    3) Use Ollama to extract metadata for each chunk (e.g., section_title, summary).
    4) Return a list of Document objects with that metadata attached.
    """
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
    splitted_docs = text_splitter.split_documents(docs)

    # For each chunk, call Ollama to get metadata
    enriched_docs = []
    for doc in splitted_docs:
        paragraph_text = doc.page_content
        extracted_meta = extract_paragraph_metadata(llm, paragraph_text)
        # Merge with existing metadata (e.g., page number) if any
        merged_meta = {**(doc.metadata or {}), **extracted_meta}
        doc.metadata = merged_meta
        enriched_docs.append(doc)

    return enriched_docs

# ---------------------------------------------------------------------------------
# 6) FAISS Helper Functions
# ---------------------------------------------------------------------------------
def load_faiss_vectorstore() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    return None

def save_faiss_vectorstore(vectorstore: FAISS):
    vectorstore.save_local(FAISS_INDEX_PATH)

# ---------------------------------------------------------------------------------
# 7) ChromaDB Helper Functions
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
# 8) Add Documents to Vector Collection
# ---------------------------------------------------------------------------------
def add_to_vector_collection(all_splits: List[Document], file_name: str, space: str, backend: str):
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
# 9) Query the Collection
# ---------------------------------------------------------------------------------
def query_collection(prompt: str, space: str, backend: str, n_results: int = 5):
    """Return docs+metadata from FAISS or ChromaDB based on the user's prompt."""
    if backend == "FAISS":
        vectorstore = load_faiss_vectorstore()
        if vectorstore is None:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        results_docs = vectorstore.similarity_search(prompt, k=n_results)
        documents = [doc.page_content for doc in results_docs]
        metadatas = [doc.metadata for doc in results_docs]
        ids = [f"doc_{i}" for i, _ in enumerate(results_docs)]
        return {"documents": [documents], "metadatas": [metadatas], "ids": [ids]}

    elif backend == "ChromaDB":
        collection = get_chromadb_collection(space)
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results

# ---------------------------------------------------------------------------------
# 10) Call Ollama for the Final Answer
# ---------------------------------------------------------------------------------
def call_llm(context: str, prompt: str, history: List[Dict], temperature: float) -> str:
    """
    Calls the Ollama LLM with a combined system prompt, context, conversation history, and user question.
    """
    history_text = "\n\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )
    full_context = f"{history_text}\n\n{context}"

    # Adjust the model name / base_url as needed
    llm = Ollama(
        base_url="http://localhost:11434",
        model="llama2",
        temperature=temperature
    )

    full_prompt = f"{system_prompt}\n\nContext: {full_context}\n\nQuestion: {prompt}"
    response = llm(full_prompt)
    return response

# ---------------------------------------------------------------------------------
# 11) Main Streamlit Application
# ---------------------------------------------------------------------------------
def main():
    # Page config (optional)
    st.set_page_config(page_title="RAG PDF Chatbot with Ollama Metadata Extraction")

    # Initialize chat & history
    if "history" not in st.session_state:
        st.session_state.history = []

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are now chatting with a PDF-based RAG assistant. "
                    "Ask questions about your uploaded documents or general queries."
                )
            }
        ]

    # Sidebar
    with st.sidebar:
        st.header("🗣️ RAG Chat Bot (Ollama)")

        # Backend selection
        backend = st.selectbox("Choose Backend", ["FAISS", "ChromaDB"], index=0)
        if backend == "ChromaDB":
            space = st.selectbox("Choose Distance Metric:", ["cosine", "euclidean", "dot"], index=0)
        else:
            space = "cosine"

        # Chunk configuration
        chunk_size = st.number_input("Set Chunk Size (characters):", min_value=100, max_value=2000, value=400, step=50)
        chunk_overlap = int(chunk_size * 0.2)

        # Model temperature
        temperature = st.slider("Model Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

        # PDF uploader
        uploaded_files = st.file_uploader("**📑 Upload PDF files for Q&A**", type=["pdf"], accept_multiple_files=True)
        process_button = st.button("⚡️ Process All")

        # Process the uploaded files
        if uploaded_files and process_button:
            # Ollama for metadata extraction
            ollama_llm = Ollama(
                base_url="http://localhost:11434",
                model="llama2",
                temperature=0.0  # you can set a low temp for more deterministic metadata extraction
            )

            for uploaded_file in uploaded_files:
                # Replace special chars in file name
                file_name = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
                # Enrich each chunk with metadata using Ollama
                all_splits = process_document(uploaded_file, chunk_size, chunk_overlap, ollama_llm)
                add_to_vector_collection(all_splits, file_name, space, backend)

    st.title("📚 Chat with your PDF(s) - Ollama Metadata Extraction")

    # Button to reset everything
    if st.button("Reset Everything (Chat + Vector Stores)"):
        st.session_state.history.clear()
        st.session_state.messages.clear()

        # Delete FAISS index folder if it exists
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)

        # Delete ChromaDB store if it exists
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)

        st.write("Chat history and vector stores have been cleared. Please refresh or ask a new question to start fresh.")

    # Display conversation so far
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

    # Input box for user
    user_query = st.chat_input("Ask a question about your documents (or anything else)...")
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query vector store
        results = query_collection(user_query, space, backend)
        context_docs = results.get("documents", [[]])[0]
        meta_docs = results.get("metadatas", [[]])[0]

        # Build a combined context from all retrieved chunks
        if not context_docs:
            concatenated_context = ""
        else:
            # Optionally, incorporate chunk metadata
            # e.g., "Section Title: {meta['section_title']}\nParagraph Summary: {meta['summary']}\nFull Text: {doc}\n"
            combined_snippets = []
            for doc_text, doc_meta in zip(context_docs, meta_docs):
                section_title = doc_meta.get("section_title", "Unknown")
                summary = doc_meta.get("summary", "")
                combined_snippets.append(
                    f"Section: {section_title}\nSummary: {summary}\nText: {doc_text}"
                )
            concatenated_context = "\n\n".join(combined_snippets)

        # Call Ollama for final answer
        raw_answer = call_llm(
            context=concatenated_context,
            prompt=user_query,
            history=st.session_state.history,
            temperature=temperature
        )

        assistant_reply = raw_answer

        # Update Q&A history
        st.session_state.history.append({"question": user_query, "answer": assistant_reply})

        # Display assistant reply
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

# ---------------------------------------------------------------------------------
# 12) Run the App
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
