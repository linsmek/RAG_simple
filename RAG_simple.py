#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.document import document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(page_title="RAG Question Answer")

with st.sidebar:
    st.header("🗣️ RAG Question Answer")
    uploaded_file = st.file_uploader("**📑 Upload PDF files for QnA**", type=["pdf"])
    process = st.button("⚡️ Process")

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    #store uploaded file as a temp file
    temp_file=tempfile.NamedTemporaryFile("wb",suffix=".pdf",delete=False)
    temp_file.write(uploaded_file.read())
    
    loader=PyMuPDFLoader(temp_file.name)
    docs=loader.load()
    os.unlink(temp_file.name) #Delete temp file
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)