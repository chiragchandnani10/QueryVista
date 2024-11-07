from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

def get_prompt_template():
    return """
        You are an expert news summarizer and analyzer. Your task is to summarize and rank the following news articles in response to the user's query. Focus on providing concise, relevant information.

**Instructions:**
1. Read the user query to understand the main focus.
2. Review the provided articles to extract key points related to the query.
3. Generate a clear, brief summary highlighting the most important details relevant to the user's query.
4. Rank the articles by relevance based on their content and the user query.

**User Query:** "{query}"

**Articles:**
{documents}

**Output:**
1. **Top Summary:** Provide a concise summary that best answers the user's query.
2. **Relevant Articles:** List the most relevant articles in descending order of relevance, including a one-sentence summary for each.

**Format:**
- **Top Summary:** [Your detailed summary here]
- **Relevant Articles:**
    1. [Title or identifier of Article 1] - [One-sentence summary]
    2. [Title or identifier of Article 2] - [One-sentence summary]
    ...

Only include information directly relevant to the user query. Avoid unnecessary details.

    """




def init_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        tokenizer=tokenizer,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()}
    )
    return qa_chain

