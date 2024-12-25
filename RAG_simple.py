#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
import os
import tempfile
import requests
import streamlit as st
from langchain.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction


# System Prompt
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
6. If the question or request has no link whatsoever and is just a general question such as "hi" or "how are you" please use your knowledge to answer it
7. If the question is kind of related but you don't have any information such as "what type of career can someone do with the background acquired with this course?" make some suggestions after mentioning that no information was to be found in the documents.
8. If the user asks you about the previous question find it in the 'history' and re-state it to the user and answer it again.
9. If the question is about multiple documents don't forget to go through each of them to find the answer

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Do not include any external knowledge or assumptions not present in the given text except in 6. or 7. situations. Don't ever hallucinate an answer.
"""

# Initialize search history
if "history" not in st.session_state:
    st.session_state.history = []

# FAISS index path
FAISS_INDEX_PATH = "./faiss_index"

# Embeddings class for Ollama
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

# Function to process PDF documents
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

# FAISS functions
def load_faiss_vectorstore() -> FAISS:
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    return None

def save_faiss_vectorstore(vectorstore: FAISS):
    vectorstore.save_local(FAISS_INDEX_PATH)

# ChromaDB functions
def get_chromadb_collection(space: str) -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": space},
    )

# Function to add documents to the vector collection
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
    st.success(f"Data from {file_name} added to the {backend} vector store!")

# Function to query the collection
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
    

# Function to call the LLM
def call_llm(context: str, prompt: str, history: list[dict]):
    history_text = "\n\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history]
    )
    full_context = f"{history_text}\n\n{context}"

    llm = Ollama(
        base_url="http://localhost:11434",  # Adjust the base URL if needed
        model="llama3.2",  # Replace with your specific model name
    )

    # Combine the system prompt and user prompt
    full_prompt = f"{system_prompt}\n\nContext: {full_context}\n\nQuestion: {prompt}"

    response = llm(full_prompt)
    yield response

# Main Streamlit application
if __name__ == "__main__":
    # Configure the page
    st.set_page_config(page_title="RAG Question Answer")
    with st.sidebar:
        st.header("🗣️ RAG Question Answer")
        backend = st.selectbox("Choose Backend", ["FAISS", "ChromaDB"], index=0)
        chunk_size = st.number_input("Set Chunk Size (characters):", min_value=100, max_value=2000, value=400, step=100)
        chunk_overlap = int(chunk_size * 0.2)
        space = st.selectbox("Choose Distance Metric:", ["cosine", "euclidean", "dot"], index=0)
        
        # Slider for dynamic temperature
        temperature = st.slider("Model Temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
        uploaded_files = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=True
        )
        process = st.button("⚡️ Process All")

        if uploaded_files and process:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
                all_splits = process_document(uploaded_file, chunk_size, chunk_overlap)
                add_to_vector_collection(all_splits, file_name, space, backend)

    st.header("🗣️ RAG Question Answer")
    user_prompt = st.text_area("**Ask a question related to your documents:**")
    ask = st.button("🔥 Ask")

    if ask and user_prompt:
        results = query_collection(user_prompt, space, backend)
        context_docs = results.get("documents", [[]])[0]

        if not context_docs:
            st.write(f"No documents available in {backend}. Please upload and process PDFs first.")
        else:
            concatenated_context = "\n\n".join(context_docs)
            placeholder = st.empty()
            full_response = ""
            response_stream = call_llm(
                context=concatenated_context,
                prompt=user_prompt,
                history=st.session_state.history,
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

            # Display retrieved documents
            with st.expander("See retrieved documents"):
                st.write(results)

            # Display IDs of the most relevant documents
            with st.expander("See most relevant document IDs"):
                st.write(results.get("ids", [[]])[0])
                st.write(concatenated_context)
