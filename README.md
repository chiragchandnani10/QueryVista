
# QueryVista: News Article Summarizer and Ranker with RAG Pipeline (Project Under The Course - Natural Language Processing)

*Demo* - https://youtu.be/BzouxsT9nVQ
**QueryVista** is a powerful application that summarizes and ranks news articles using a Retrieval-Augmented Generation (RAG) pipeline. By leveraging **LangChain**, **ChromaDB**, and **Streamlit**, this app allows users to input news article URLs, store them in a vectorized database, and retrieve relevant summaries based on user queries.

## Features:
- **RAG-based Summarization**: Uses a combination of **retrieval** and **generation** techniques to summarize and rank articles.
- **Vector Database**: Stores articles as vectorized embeddings using **ChromaDB** for fast and efficient retrieval.
- **Streamlit Interface**: A simple web interface for users to input article URLs and query for relevant summaries.
- **Summarization**: Uses a free text summarization model from Hugging Face to summarize articles.
- **Ranking**: Ranks the articles based on relevance to the user’s query.

## Technologies:
- **LangChain**: For orchestrating the RAG pipeline and managing the chain of operations.
- **ChromaDB**: For storing and managing article embeddings in a vector database.
- **Streamlit**: For building the user interface.
- **Hugging Face**: For using pre-trained models like text summarizers.

## Setup Instructions:

### Prerequisites:
Before you begin, make sure you have the following installed:
- **Python** (>= 3.8)
- **pip** (Python package installer)
- **Docker** (for containerized deployment, optional but recommended)



### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```



### 3. Running the Application:
To run the application locally using Streamlit, simply execute:
```
streamlit run app/main.py
```

This will start the app and open it in your default web browser.

### 4. (Optional) Docker Deployment:
If you prefer to deploy the application in a Docker container, follow these steps:
1. Build the Docker image:
   ```bash
   docker build -t queryvista .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 queryvista
   ```

The app will be accessible at `http://localhost:8501`.

## How to Use:

1. **Input News Articles**: On the homepage, you can input URLs of news articles. The application will scrape and process the content, storing it in the vector database.
   
2. **Query for Summaries**: Once articles are stored, you can enter a query related to any of the stored articles. The app will rank the articles based on their relevance to the query and summarize them for you.

3. **View Summaries**: The summarized content of the top-ranked articles will be displayed on the screen. You can view the original article or the generated summary directly.

## Code Structure:

```
QueryVista/
├── main.py                # Streamlit app interface
├── scraper.py             # Article scraping logic
├── vector_store.py        # Vector store and database operations
├── rag_pipeline.py        # Retrieval-Augmented Generation (RAG) pipeline
├── requirements.txt       # Required Python libraries
├── Dockerfile             # Dockerfile for containerizing the app
├── README.md   
├── LICENSE 
demo/
├── sample_input.txt       # Contains the sample/test case input to the app

```


