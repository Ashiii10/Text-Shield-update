from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os
import streamlit as st

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore(uploaded_file) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def query_rag(question: str, vectorstore: FAISS) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.3,
        convert_system_message_to_human=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]

    pages = list(set([
        doc.metadata.get("page", None)
        for doc in source_docs
        if doc.metadata.get("page") is not None
    ]))
    pages_str = ", ".join([f"p.{p+1}" for p in sorted(pages)])

    if pages_str:
        return f"{answer}\n\n📄 *Sources: {pages_str}*"
    return answer


def analyze_document_rag(vectorstore: FAISS) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.1,
        convert_system_message_to_human=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    result = qa_chain.invoke({
        "query": """Analyze the writing style of this document.
Does it appear to be AI-generated or human-written?
Look for: repetitive sentence structures, overly formal transitions,
lack of personal voice, uniform paragraph lengths.
Give a clear verdict and brief explanation."""
    })

    return result["result"]
