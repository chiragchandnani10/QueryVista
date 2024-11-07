import streamlit as st
from scraper import scrape_article
from vector_store import init_vector_store, add_articles_to_store, search_articles
from rag_pipeline import init_rag_pipeline

vector_store, embeddings = init_vector_store()
rag_chain = init_rag_pipeline(vector_store)

# Streamlit UI
st.title("News Article RAG-based Summarizer and Ranker")

st.header("Add News Article URLs")
urls = st.text_area("Enter the article URLs (one per line)", height=150)
submit_articles = st.button("Process Articles")

if submit_articles:
    article_urls = urls.splitlines()
    article_contents = []
    
    for url in article_urls:
        article_content = scrape_article(url)
        if article_content:
            article_contents.append(article_content)
    
    # Add articles to the vector store
    add_articles_to_store(vector_store, article_contents)
    st.success(f"Processed and stored {len(article_contents)} articles in the vectorized database")

st.header("Search for Relevant Articles with RAG")
user_query = st.text_input("Enter your search query", "")
search_button = st.button("Search and Summarize")

if search_button and user_query:
    # Retrieve the top 5 relevant articles
    relevant_articles = search_articles(vector_store, user_query, top_k=5)
    
    st.subheader("Top 5 Relevant Articles Based on Your Query")
    
    for idx, article in enumerate(relevant_articles):
        st.write(f"### Article {idx + 1}:")
        
        # Generate summary for each retrieved article
        summary_result = rag_chain({"query": user_query, "documents": [article]})
        st.write(f"**Summary:** {summary_result['result']}")
        st.write("---")

    # Generate overall summary for the top articles
    overall_summary_result = rag_chain({"query": user_query})
    st.subheader("Overall Summary Based on Retrieved Articles")
    st.write(overall_summary_result['result'])
