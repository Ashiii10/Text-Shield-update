import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.7,
        convert_system_message_to_human=True
    )


def run_agentic_task(prompt: str) -> str:
    try:
        llm = get_llm()
        template = PromptTemplate(
            input_variables=["task"],
            template="{task}"
        )
        chain = template | llm
        result = chain.invoke({"task": prompt})
        return result.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def run_rewrite_task(text: str) -> str:
    try:
        llm = get_llm()
        template = PromptTemplate(
            input_variables=["text"],
            template="Rewrite the following text in a formal, professional tone:\n\n{text}"
        )
        chain = template | llm
        result = chain.invoke({"text": text})
        return result.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def run_grammar_task(text: str) -> str:
    try:
        llm = get_llm()
        template = PromptTemplate(
            input_variables=["text"],
            template="Fix the grammar in the following text and briefly explain the corrections made:\n\n{text}"
        )
        chain = template | llm
        result = chain.invoke({"text": text})
        return result.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def run_chat_task(conversation_history: str) -> str:
    try:
        llm = get_llm()
        template = PromptTemplate(
            input_variables=["history"],
            template="{history}\nAssistant:"
        )
        chain = template | llm
        result = chain.invoke({"history": conversation_history})
        return result.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
