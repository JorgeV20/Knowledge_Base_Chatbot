import streamlit as st
import os
from ingest_docs import create_vector_db
import re

#chatbot
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

#Forecast
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date
from prophet.plot import plot_plotly
from plotly import graph_objs as go


@st.cache_resource
def load_llm():
        # Load the locally downloaded model here
        llm = GPT4All(
            model="mistral-7b-instruct-v0.1.Q4_0.gguf",
            #max_tokens=300,
            #n_threads = 4,
            #temp=0.3,
            #top_p=0.2,
            top_k=5,#40,
            #n_batch=8,
            #seed=100,
            allow_download=True,
            verbose=True)
        return llm 

@st.cache_resource
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                   model_kwargs = {'device': "cpu"})
    return embeddings


def preprocess_text(text):
    text_lower = text.lower()
    # only allow these characters
    text_no_punctuation = re.sub(r'[^\w\s\$\%\.\,\"\'\!\?\(\)]', '', 
                                 text_lower)
    # removes extra tabs space
    text_normalized_tabs = re.sub(r'(\t)+', '', text_no_punctuation)
    return text_normalized_tabs

#Create vector database
@st.cache_resource
def create_vector_db():
    #Instanciate the Directory Loader in order to load the pdf files
    loader = PyPDFLoader("data/The Alchemy of Finance, Reading the Mind of the Market.pdf")
    documents = loader.load()

    for x in range(len(documents)):
        # do preprocessing
        documents[x].page_content=preprocess_text(documents[x].page_content)


    #Instanciate the Text Splitter in chunks and split the document
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0,separator="\n")
    docs = text_splitter.split_documents(documents)

    #Instanciate the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5",
                                   model_kwargs = {'device': "cpu"})

    
    #Create the FAISS db
    qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="msft_data",
    force_recreate=True
    )  

    return qdrant



llm=load_llm()
embeddings=get_embedding_model()
qdrant=create_vector_db()

def format_docs(query):
    found_docs = qdrant.similarity_search_with_score(query,k=1)
    return "\n\n".join(doc[0].page_content for doc in found_docs)

st.title('🦜🔗 Flint, your FinanceBot')
st.markdown("""
## Finance Bot: Get instant insights from Finance

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework
            
""")

#col1, col2 = st.columns(2)

#with col1:
    

#with col2:
    
#st.header("Ask to Flint")

    #DB_FAISS_PATH = os.path.join(local_path, 'vectorstore_docs/db_faiss')

template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else. Try to make it short. Maximum of 500 words.
    Helpful answer:
    """
rag_prompt = PromptTemplate(template=template, input_variables=["context","question"])


callbacks = [StreamingStdOutCallbackHandler()]
llm_chain = LLMChain(prompt=rag_prompt, llm=llm, verbose=True)

    
st.header("Ask to Flint 🤖")

query = st.text_input("Ask a Question from Finance", key="user_question")
if query:
        response=llm_chain.invoke(
    input={"question":query,
           "context": format_docs(query)
          })
        st.write("Reply: ", response['text'])

st.header('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
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
fig1 = plot_plotly(m, forecast)
#fig1 = m.plot(forecast)
st.plotly_chart(fig1,use_container_width=True)