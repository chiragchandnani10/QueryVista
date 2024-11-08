import streamlit as st
from scraper import scrape_article
from vector_store import init_vector_store, add_articles_to_store, search_articles
from rag_pipeline import init_rag_pipeline

vector_store, embeddings = init_vector_store()
qa_chain = init_rag_pipeline(vector_store)

# Streamlit UI
st.title("QueryVista - An automated RAG-based Ranker and Summarizer (Summarizes top 2 articles)")
st.header("Project under the course Natural Language Processing - BCSE409L")

st.header("Add News Article URLs")
urls = st.text_area("Enter the article URLs (one per line)", height=150)
submit_articles = st.button("Process Articles")


def format_single_article(document):
    # Ensure that we access only the text content from each Document instance
    return document.page_content if hasattr(document, "page_content") else ""


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
    print("ARTICLE CONTENTS: ",article_contents)

st.header("Search for Relevant Articles with RAG")
user_query = st.text_input("Enter your search query", "")
search_button = st.button("Search and Summarize")

if search_button and user_query:
    # Retrieve the top k relevant articles
    relevant_articles = search_articles(vector_store, user_query, search_type = "similarity")
    print(f"Relevant articles------------------{relevant_articles}")
    # relevant_articles = relevant_articles_retriever.invoke(user_query)
    print(f"User query {user_query} and {type(user_query)}")
    st.subheader("Top Relevant Articles Based on Your Query")
    
    for idx, article in enumerate(relevant_articles):
        st.write(f"### Article {idx + 1}:")
        print(type(article))
        # Generate summary for each retrieved article
        formatted_article = format_single_article(article)
        print('-----------------------------------------------')
        print(type(formatted_article))
        print("Len is ", len(formatted_article))
        formatted_article = formatted_article[:1800] #Truncate the article length to 1800 size token for model limitation



        # summary_result = rag_chain({"query": user_query, "documents": [article]})
        summary_result = qa_chain.invoke({"query":user_query,"documents":formatted_article})
        st.write(f"**Summary:** {summary_result}")
        st.write("---")

    # Generate overall summary for the top articles
    formatted_articles = "\n\n".join(format_single_article(article) for article in relevant_articles)


    print(formatted_articles)
    st.subheader("It gives like this output, as its a Autocompletion GPT2-NEO 125M model")
    st.subheader("You can change the model from rag_pipeline.py")
    overall_summary_result = qa_chain.invoke({"documents":formatted_articles,"query":user_query})


    st.subheader("Overall Summary Based on Retrieved Articles")
    st.write(overall_summary_result)
