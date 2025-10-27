import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from typing import Literal
from dotenv import load_dotenv
import streamlit as st
import os

# 1 - Load Api
os.environ['LANGCHAIN_PROJECT'] = "Youtube Summarizer"
load_dotenv()


#2 - Making a model
embedding_model =  HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
llm1 = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation")
model1 = ChatHuggingFace(llm=llm1)



#3 - Getting Transcript form video
# normal_link = "https://www.youtube.com/watch?v=E2DEHOEbzks"

st.header("Youtube Video Summarizer")
user_input = st.text_input("Provide a Link to Summarize")



if st.button("Analyze"):
# 5 -Fetch transcript
 video_id = user_input.split("v=")[1].split("&")[0]
 try:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    transcript = ''.join([chunk.text for chunk in transcript])

 except TranscriptsDisabled:
   print("No transcript available for video")



#6 -Chunking and FAISS
 splitter = RecursiveCharacterTextSplitter(chunk_size =1000 ,chunk_overlap=50)
 chunks = splitter.create_documents([transcript])


# 7- Retriever and Contwxting
 vector_store = FAISS.from_documents(chunks,embedding_model)
 retriever = vector_store.as_retriever(search_kwargs ={'k':4})

 prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant that summarizes YouTube videos.

    Context:
    {context}

    Question:
    {question}

    Answer in clear, concise paragraphs:
    """
)


# 8 -Paraller Chaining
 parallel_chain = RunnableParallel({
    'context': retriever,
    'question':RunnablePassthrough()})


 final_chain = parallel_chain | prompt | model1 | StrOutputParser()


 response = final_chain.invoke(user_input)
 st.write(response)
