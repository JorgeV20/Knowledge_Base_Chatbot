# model.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

DB_FAISS_PATH = 'vectorstore/db_faiss'
MODEL_PATH = "./models/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf"

custom_prompt_template = """<|im_start|>system
You are a professional financial analyst RAG bot. Use the provided context documents and live market data to answer the user's question accurately. If you don't know the answer, say you don't know. Keep your answer concise and accurate.
<|im_end|>
<|im_start|>user
Conversation History:
{chat_history}

Context Documents:
{context}

Live Market Data:
{live_market_data}

News related to the company:
{articles}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['chat_history','context', 'question', 'live_market_data', 'articles']
    )

def load_llm():
    # LlamaCpp allows native integration with LangChain pipeline
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=-1,
        temperature=0.3,
        max_tokens=2048,
        verbose=False
    )


print("Initializing Qwen 2.5 RAG Pipeline on GPU... Please wait.")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={'k': 3}) 
llm = load_llm()
prompt_template = set_custom_prompt()
print("Qwen 2.5 GPU Pipeline fully ready!")

def final_result(user_query, live_data_str, articles, chat_history):
    print("Answer method")
    docs = retriever.invoke(user_query)
    context_chunks = "\n\n".join([doc.page_content for doc in docs])
    
    formatted_prompt = prompt_template.format(
        chat_history=chat_history,
        context=context_chunks,
        live_market_data=live_data_str,
        question=user_query,
        articles=articles

    )
    print(formatted_prompt)
    
    answer = llm.invoke(formatted_prompt)
    
    return {'result': answer}