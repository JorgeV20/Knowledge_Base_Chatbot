import streamlit as st

#chatbot
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

#Forecast
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('🦜🔗 Flint, your FinanceBot')
st.markdown("""
## Finance Bot: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework
            
""")

col1, col2 = st.columns(2)

with col1:
    st.header('Stock Forecast App')
    #st.image("https://static.streamlit.io/examples/cat.jpg")
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
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



with col2:
   st.header("Ask to Flint")
   #st.image("https://static.streamlit.io/examples/dog.jpg")