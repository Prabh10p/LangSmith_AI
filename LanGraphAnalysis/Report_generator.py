from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import TypedDict
import os
import streamlit as st
from dotenv import load_dotenv
import json


# 1. Load API and Env
os.environ["LANGCHAIN_PROJECT"] = "Report Analyzer and Generator"
load_dotenv()


# 2. Create Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-7b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header("🧠 Report Generator and Evaluator")

#3. User Input

user_input = st.text_input("Enter the topic you want to generate a report on:")

# 4. Define State TypedDict
class Imp(TypedDict, total=False):
    text: str
    essay: str
    depth_feedback: str
    grammar_feedback: str
    structure_feedback: str
    feedback: str
    depth_score: int
    grammar_score: int
    structure_score: int
    avg_score: float

# 5. Feedback Pydantic Models

class FeedbackModel(BaseModel):
    feedback: str
    score: int

class OverallFeedbackModel(BaseModel):
    feedback: str
    evaluation: str

# 5. Define Functions (Nodes)

def essay_generator(state: Imp):
    prompt = f"""
    You are a professional essay writer. 
    Write a well-structured essay on the following topic:
    "{state['text']}".
    Ensure it includes an introduction, body, and conclusion.
    """
    response = model.invoke(prompt)
    return {"essay": response.content}

def depth_feedback(state: Imp):
    prompt = f"""
    Analyze the depth of the following essay based on the topic "{state['text']}".
    Essay: {state['essay']}
    Provide detailed feedback and give a score out of 10 in JSON like:
    {{
        "feedback": "...",
        "score": 8
    }}
    """
    response = model.invoke(prompt)
    parsed = FeedbackModel.parse_raw(response.content)
    return {"depth_feedback": parsed.feedback, "depth_score": parsed.score}

def grammar_feedback(state: Imp):
    prompt = f"""
    Analyze the grammar of the following essay based on the topic "{state['text']}".
    Essay: {state['essay']}
    Provide detailed feedback and give a score out of 10 in JSON like:
    {{
        "feedback": "...",
        "score": 8
    }}
    """
    response = model.invoke(prompt)
    parsed = FeedbackModel.parse_raw(response.content)
    return {"grammar_feedback": parsed.feedback, "grammar_score": parsed.score}

def structure_feedback(state: Imp):
    prompt = f"""
    Analyze the structure and tone of the following essay based on the topic "{state['text']}".
    Essay: {state['essay']}
    Provide detailed feedback and give a score out of 10 in JSON like:
    {{
        "feedback": "...",
        "score": 8
    }}
    """
    response = model.invoke(prompt)
    parsed = FeedbackModel.parse_raw(response.content)
    return {"structure_feedback": parsed.feedback, "structure_score": parsed.score}

def overall_feedback(state: Imp):
    avg_score = round(
        (state['depth_score'] + state['grammar_score'] + state['structure_score']) / 3, 2
    )
    prompt = f"""
    Provide an overall evaluation summary for the essay below in JSON format like:
    {{
        "feedback": "...",
        "evaluation": "approved"  # or "not approved"
    }}

    Topic: {state['text']}
    Essay: {state['essay']}
    Depth Feedback: {state['depth_feedback']}
    Grammar Feedback: {state['grammar_feedback']}
    Structure Feedback: {state['structure_feedback']}
    Average Score: {avg_score}
    """
    response = model.invoke(prompt)
    parsed = OverallFeedbackModel.parse_raw(response.content)
    return {"feedback": parsed.feedback, "avg_score": avg_score}

# 6. Build Graph

graph = StateGraph(Imp)
graph.add_node("essay_generator", essay_generator)
graph.add_node("depth_feedback", depth_feedback)
graph.add_node("grammar_feedback", grammar_feedback)
graph.add_node("structure_feedback", structure_feedback)
graph.add_node("overall_feedback", overall_feedback)

# Connect edges
graph.add_edge(START, "essay_generator")
graph.add_edge("essay_generator", "depth_feedback")
graph.add_edge("depth_feedback", "grammar_feedback")
graph.add_edge("grammar_feedback", "structure_feedback")
graph.add_edge("structure_feedback", "overall_feedback")
graph.add_edge("overall_feedback", END)

workflow = graph.compile()


# 7. Execute Workflow
if st.button("Generate Report"):
    if not user_input.strip():
        st.warning("Please enter a topic to generate a report.")
    else:
        with st.spinner("Generating essay and evaluation... ⏳"):
            response = workflow.invoke({"text": user_input})

        st.subheader("📘 Generated Essay")
        st.write(response["essay"])

        st.subheader("🔍 Feedback Summary")
        st.write(f"**Depth Feedback:** {response['depth_feedback']}")
        st.write(f"**Grammar Feedback:** {response['grammar_feedback']}")
        st.write(f"**Structure Feedback:** {response['structure_feedback']}")

        st.subheader("⭐ Overall Evaluation")
        st.write(f"**Feedback:** {response['feedback']}")
        st.write(f"**Average Score:** {response['avg_score']}/10")
