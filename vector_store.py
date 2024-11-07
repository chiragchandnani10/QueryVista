from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import Document

def init_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(embedding_function=embeddings, persist_directory="db")
    return vector_store, embeddings

def add_articles_to_store(vector_store, article_contents):
    docs = [Document(page_content=article) for article in article_contents]
    vector_store.add_documents(docs)

def search_articles(vector_store, query, top_k=5):
    results = vector_store.similarity_search(query, top_k=top_k)
    return results
