from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import streamlit as st

# 1. Load API Keys
os.environ["LANGCHAIN_PROJECT"] = "Youtube Summarizer"
load_dotenv()

# 2. Creating Models (✅ Fixed Embedding Initialization)
# Load SentenceTransformer manually to avoid meta tensor bug
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation"
)
model1 = ChatHuggingFace(llm=llm1)

# 3. Streamlit UI
st.header("🎥 YouTube Video Summarizer")
normal_link = st.text_input("Enter the YouTube link to summarize")

if st.button("Summarize"):
    if "v=" in normal_link:
        link = normal_link.split("v=")[1].split("&")[0]
    else:
        link = normal_link.strip()

    try:
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(link)
        transcript = " ".join([chunk.text for chunk in transcript_data])
    except TranscriptsDisabled:
        st.error("This video does not have a transcript available.")
        st.stop()
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        st.stop()

    # 4. Splitter & Storing
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    # 5. Prompt Template
    prompt = PromptTemplate(
        template="""You are an intelligent YouTube summarizer.
Context:
{context}

Question:
{question}

Generate a detailed and insightful summary of the video content.
""",
        input_variables=["context", "question"],
    )

    parallel_chain = RunnableParallel({
        'context': retriever,
        'question': RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model1 | StrOutputParser()

    st.write("Generating summary...")
    response = final_chain.invoke("Summarize this YouTube video.")
    st.subheader("Summary:")
    st.write(response)
