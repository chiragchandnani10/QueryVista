from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

# from langchain.llms import OpenAI
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-125M"  # Use a generative model for summarization
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_prompt_template_text(documents, query):
    return  """You are an expert news summarizer and analyzer. Your task is to summarize  the following news articles in response to the user's query. Focus on providing concise, relevant information.

Given Context : {documents}

Given User Query: {query}



Output: Give a good concise summary

Only include information directly relevant to the user query. Avoid unnecessary details.

    """

def get_prompt_template():
    return PromptTemplate(
        input_variables=["documents", "query"],
        template = """You are an expert news summarizer and analyzer. Your task is to summarize the following news articles with respect to the user's query. Focus on providing concise, relevant information.

Given Context : {documents}

Given User Query: {query}



Output: Give a good concise summary

Only include information directly relevant to the user query. Avoid unnecessary details.

    """
    )




def format_docs(docs):
    return " ".join(doc for doc in docs)

def generate_summary(input_data):
    # print(type(input_data))
    # documents = input_data['documents']
    # query = input_data['query']
    # input_text = get_prompt_template_text(input_data['documents'],input_data['documents'])
    print("INPUTDATA_",input_data)

    input_data = str(input_data)
    # Tokenize the input
    test_input = "The sky is blue, and the grass is green. The sun shines brightly in the sky."

    tokenizer.pad_token = tokenizer.eos_token  #Padding token set as EOS (End of Sentence)
    print("Len of prompt",len(input_data))
    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True,max_length=2000)
    
    # Generate the summary
    summary_ids = model.generate(
        input_ids = inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens = 200, 
        num_return_sequences=1,
        temperature=0.6,        # Adjust randomness
        # top_k=50,              # Limit next token choices
        num_beams=5,           # Use beam search for better results
        no_repeat_ngram_size=2 # Prevent repetition
    )    
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"SUMMARYTEXT is {summary_text}")
    # Remove any residual parts of the input prompt if necessary
    if input_data in summary_text:
        summary_text = summary_text.split(input_data)[-1].strip()

    print(f"SUMMARYTEXT is {summary_text}")

    return summary_text

def fit_in_prompt(input_str):
    prompt = get_prompt_template()
    print("INPUT STRING FIT IN PROMPT ",input_str)
    return prompt.format(query=input_str['query']['query'],documents = input_str['query']['documents'])


def init_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever()
    prompt_template = get_prompt_template()
    qa_chain = (
    RunnableMap({  # Wrapping the dictionary in RunnableMap
            "documents": RunnablePassthrough() | format_docs,
            "query": RunnablePassthrough(),
        })
    | fit_in_prompt
    | generate_summary
    | StrOutputParser()
)
    return qa_chain

