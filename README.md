# Knowledge Base Chatbot

FinanceBot Knowledge Base Chatbot is a tool designed to provide instant access to financial information and expertise. It has access to real-time stock market data and financial news. It extracts valuable insights and knowledge from a comprehensive finance PDF resource, transforming it into an interactive chatbot experience. FinanceBot was built using Qwen2.5-3B-Instruct, Langchain, FAISS, and Flask as foundational technologies.

![FinanceBot](./static/images/finance_chatbot.png)

## Repository Structure
- [`README.md`](README.md): The file contais the description of the project.
- [`app.py`](app.py): Execute FinanceBot in a Flask server.
- [`model.py`](model.py): The file executes the chatbot instructions in order to generate a response according to user's input.
- [`ingest.py`](ingest.py): The file generates the vectorstore based on the pdf source.
- [`vectoresctore/db_faiss`](vectoresctore/db_faiss): Contains the vectostore database using FAISS.
- [`templates`](templates): The folder contains the html file of the interface application.
- [`static`](static): The folder contains the css, js, and images files used in the application.
- [`data`](data): The folder contains the pdf used to create the knowledge base.

## Source
The pdf sources are:
- The Basics of Finance An Introduction to Financial Markets, Business Finance, and Portfolio Management
- The Alchemy of Finance, Reading the Mind of the Market
- The Nature of Investing

The API sources are:
- [NewsAPI](newsapi.org)
- [Yahoo Finance](https://finance.yahoo.com/)

## Future work
- Increase the speed of answer.
- Increase the number of sources.