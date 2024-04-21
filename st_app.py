import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

st.title('🦜🔗 Flint, your FinanceBot')
st.markdown("""
## Finance Bot: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework
            
""")
