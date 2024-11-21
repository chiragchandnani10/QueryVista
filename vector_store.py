from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def init_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Initialize Chroma with a persistent directory for storing vectors
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="db"  # This directory will store Chroma’s database files
    )
    
    # Persist and reconnect
    vector_store.persist()
    return vector_store, embeddings

def add_articles_to_store(vector_store, article_contents):
    print(f"Articles fetched from the sites: {article_contents}")
    docs = [Document(page_content=article,id=idx) for idx,article in enumerate(article_contents)]
    vector_store.add_documents(docs)

def search_articles(vector_store, user_query, search_type, top_k=1):
#     retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2,"k": 1}
# )
#     return retriever

    # relevant_articles = vector_store.search(user_query,search_type=search_type, top_k=2)


    relevant_articles = vector_store.similarity_search(user_query, k=top_k)

    return relevant_articles
