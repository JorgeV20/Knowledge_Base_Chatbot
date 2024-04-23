from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH='docs/'
DB_FAISS_PATH='vectorstore_docs/db_faiss'

#Create vector database
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

if __name__=='__main__':
    create_vector_db()