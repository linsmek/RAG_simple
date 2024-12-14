#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:58 2024

@author: linamekouar
"""

import streamlit as st


st.set_page_config(page_title="RAG Question Answer")

with st.sidebar:
    st.header("🗣️ RAG Question Answer")
    uploaded_file = st.file_uploader("**📑 Upload PDF files for QnA**", type=["pdf"])
    process = st.button("⚡️ Process")
