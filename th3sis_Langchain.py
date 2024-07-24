import streamlit as st
import numpy as np
import pandas as pd
import openai
import requests
import json
from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
import bs4


import openai
import hmac
from io import StringIO
import os
import base64
import os
from os.path import exists
import ast
import pdb
import sys
import time
import fitz  # PyMuPDF
import getpass
import os

#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity

prompt=""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN_API_KEY
#conversation_history=[]
llm = ChatOpenAI(model="gpt-4o-mini")






file_paths = "data/dataset/Book1.csv"
loader = CSVLoader(file_path=file_paths)
data = loader.load()

file_paths = "data/dataset/Book2.csv"
loader = CSVLoader(file_path=file_paths)
data2 = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(data)

# Set the maximum batch size
max_batch_size = 5461

# Function to add documents in batches
def add_documents_in_batches(splits, max_batch_size):

    for i in range(0, len(splits), max_batch_size):
        batch = splits[i:i + max_batch_size]
        vectorstore = Chroma.from_documents(documents=batch,embedding=OpenAIEmbeddings())
        vectorstore.add_documents(documents=batch)
    return vectorstore

vectorstore = add_documents_in_batches(splits, max_batch_size)
retriever = vectorstore.as_retriever()



# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#st.text("Index is ready")
        

response = rag_chain.invoke({"input": "What can you say about the document?"})
print(response["answer"])