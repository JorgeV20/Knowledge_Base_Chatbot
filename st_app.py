import streamlit as st
import os
from ingest_docs import create_vector_db

#chatbot
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

#Forecast
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date
#from prophet.plot import plot_plotly
from plotly import graph_objs as go

local_path=os.getcwd()

@st.cache_resource
def load_llm():
        # Load the locally downloaded model here
        llm = CTransformers(
            model = "TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.5
        )
        return llm 

@st.cache_resource
def get_embedding_model():
    embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device':'cpu'})
    return embeddings

DATA_PATH='data/'
DB_FAISS_PATH='vectorstore/db_faiss'

#Create vector database
@st.cache_resource
def create_vector_db():
    #Instanciate the Directory Loader in order to load the pdf files
    loader=DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents=loader.load()

    #Instanciate the Text Splitter in chunks and split the document
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    #Instanciate the embedding model
    embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device':'cpu'})
    
    #Create the FAISS db
    db=FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    db=FAISS.load_local(DB_FAISS_PATH, embeddings)

    return db



llm=load_llm()
embeddings=get_embedding_model()
db=create_vector_db()


st.title('🦜🔗 Flint, your FinanceBot')
st.markdown("""
## Finance Bot: Get instant insights from Finance

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework
            
""")

#col1, col2 = st.columns(2)

#with col1:
    

#with col2:
    
st.header("Ask to Flint")

    #DB_FAISS_PATH = os.path.join(local_path, 'vectorstore_docs/db_faiss')

custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else. Try to make it short. Maximum of 500 words.
    Helpful answer:
    """

def set_custom_prompt():
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    #Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt}
                                        )
        return qa_chain

    #Loading the model
    

    #QA Model Function
def qa_bot(llm,db,embeddings):
        embeddings = embeddings#HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        #model_kwargs={'device': 'cpu'})
        db = db#FAISS.load_local(DB_FAISS_PATH, embeddings)
        llm = llm#load_llm()
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)

        return qa


    #output function
def final_result(query,llm,db,embeddings):
        qa_result = qa_bot(llm,db,embeddings)
        response = qa_result.invoke({'query': query})
        return response
    
    
st.header("Chatbot💁")
    #st.write(DB_FAISS_PATH)
    #filenames = os.listdir(DB_FAISS_PATH)
    #for file in filenames:
       #st.write(file)

user_question = st.text_input("Ask a Question from Finance", key="user_question")
if user_question:
        response=final_result(user_question,llm,db,embeddings)
        st.write("Reply: ", response['result'])

st.header('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
data = load_data('AAPL')
st.subheader('Raw data')
st.write(data.tail())

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m=Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
st.subheader('Forecast data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.write(f'Forecast plot for 1 year')
    #fig1 = plot_plotly(m, forecast)
    #fig1 = m.plot(forecast)
    #st.plotly_chart(fig1)