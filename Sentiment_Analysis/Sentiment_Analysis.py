import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Step 1: Models
llm1 = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation")
model1 = ChatHuggingFace(llm=llm1)
llm2 = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation")
model2 = ChatHuggingFace(llm=llm2)

# Step 2: Parsers
class Pydan(BaseModel):
    response: Literal['positive', 'negative']

def parse_sentiment(x):
    """Parse model output to sentiment with better error handling"""
    content = x.content.strip().lower()
    
    # Check if the content contains the sentiment keywords
    if 'positive' in content:
        return Pydan(response='positive')
    elif 'negative' in content:
        return Pydan(response='negative')
    else:
        # Default to negative, but you might want to log this
        print(f"Warning: Unexpected output '{content}', defaulting to negative")
        return Pydan(response='negative')

parser1 = RunnableLambda(parse_sentiment)

# Convert Pydantic model to dict for next prompt
pydantic_to_dict = RunnableLambda(lambda x: {'response': x.response})

parser2 = StrOutputParser()

# Step 3: Prompts
prompt1 = PromptTemplate(
    input_variables=['user_text'],
    template="Analyze the sentiment of this text and respond with ONLY one word: either 'positive' or 'negative'.\n\nText: {user_text}\n\nSentiment:"
)

prompt2 = PromptTemplate(
    input_variables=['response'],
    template=(
        "The user's sentiment is: {response}. "
        "If positive, give a warm thankful message. "
        "If negative, give a sincere and polite apology."
    )
)

# Step 4: Chain


st.header('AI Based Sentiment Analyzer')
user_input = st.text_input("Enter your tweet: ")



if st.button("Analyze"):
  chain = (
    prompt1 
    | model1 
    | parser1 
    | pydantic_to_dict  # Add this step!
    | prompt2 
    | model2 
    | parser2
)

  response = chain.invoke({'user_text': user_input})
  st.write(response)