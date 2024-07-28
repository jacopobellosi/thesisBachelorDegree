import streamlit as st
import numpy as np
import pandas as pd
import openai
import requests
import json
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, \
                             SimpleDirectoryReader, \
                             load_index_from_storage, \
                             Settings, \
                             StorageContext, \
                             Document
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
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity

prompt=""
#conversation_history=[]

def remove_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


st.set_page_config(page_title="Chat with your CV Review Assistant for a better resume",
                   page_icon="📄",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("CV Virtual Assistant")

def check_password():

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets.mypassword):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

st.subheader("Welcome to the AI Virtual Assistant for CV consultant")
st.text("The assistant is based on LLamaIndex and ChatGpt-4o models")
st.text("⚠️Important: you have to upload your CV in order the let the AI\n work properly (better if in English)⚠️")
st.divider()
openai.api_key = st.secrets.openai_key
client = OpenAI(api_key=st.secrets.openai_key,project="proj_TSOrJg10Snkt4UuWMSPHpgpQ")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Start your consultation about your CV.",
        }
   ]

#st.text('Preparing the model...')
persist_directory = './index'
index_files = ['vector_store.json', 'docstore.json', 'index_store.json']
index_exists = all(os.path.exists(os.path.join(persist_directory, file)) for file in index_files)
Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            system_prompt="""You're an expert recruiter. You have received my CV as a JSON, and you need to provide technical advice on how to improve it. 
                     Don't show the JSON file; only show the text!

                     Instructions:
                     - Provide technical and academic advice based on facts.
                     - Do not make up features or information.
                     - Use the provided dataset for references.
                     - Focus on areas such as skills, job experience, education,certification,ecc.
                     - Offer career advice based on current industry trends.
                     - Provide a detailed response with actionable advice.
                     - Use a professional tone and be concise.
                     - Do no alucinate.

                     Example of response format:
                     - Skills: Improve your proficiency in Python by taking advanced courses.
                     - Experience: Highlight your project management experience more prominently.
                     - Education: Consider obtaining a certification in data science.

                     Note: If the CV is missing crucial information, suggest adding it.

                     Dataset reference: The dataset has important information about the carrer path, skills, and job experience of the user. Use this information to provide advice.
                        It contains information about the carrer, as if the job domand is rising or not, use this information to provide advice and data for the user.
                        Show the actual data when you provide advice.

            """,
)

def print_text():
    for message in st.session_state.messages:  # Write message history to UI
        with st.chat_message(message["role"]):
            st.write(message["content"])





def extract_text_from_images(concatenated_image):

    prompt = f"""
            Extract the text into appropriate JSON. Put multiple related items in an array.
            The response must be only in JSON. Must only contain the JSON response, not a word in eccess!
            
            """

    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
            }
    print("elaboro il CV")

    payload = {
        "model": "gpt-4o-mini",
        "temperature":0,
        "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{concatenated_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
            
    response = response['choices'][0]['message']['content']
    #st.write(response)
    print(response)
            
    response = {"role": "assistant", "content": response}
    #st.text('Loading your data...')
    #index.insert_nodes(response["content"])
    #st.text('Preparing the engine...')
    print("MOSTRO LA STORIA \n",st.session_state.messages, " \n lenght:", len(st.session_state.messages),"FINE STORIA \n")
    if len(st.session_state.messages)==1:
        get_first_advice(response["content"])
            
    #conversation_history.append(response)
    #print(conversation_history)
    #st.session_state.messages.append(message)
    #response_stream = st.session_state.chat_engine.stream_chat(prompt)
    #st.write_stream(response_stream.response_gen)
    #message = {"role": "assistant", "content": response_stream.response}
    #st.session_state.messages.append(message)
            
    return response["content"]


def get_first_advice(JSON_CV):
    #st.write(JSON_CV)
    print("ti do ora un primo consiglio")
    #print(response)
    #response = response['choices'][0]['message']['content']
    #st.write(response)

    response_stream = st.session_state.chat_engine.chat(f"""
        Based on the provided CV, suggestsome technical improvements on how to get a better CV in the following areas:
            - Summary
            - Skills
            - Job Experience
            - Education
            - Certifications
            - Career Advice
            - Language
            - Other relevant areas
            Provide specific, actionable advice. For example:
            - "Enhance your proficiency in data analysis by learning advanced Excel functions."
            - "Consider adding more details about your project management responsibilities in your last job."
            - "Is betteer to follow Cloud Computing sector as it's rising rapidly."

            CV: {JSON_CV}
            Use the following dataset for references and ensure your advice is based on current industry trends: {index}.
            Stick to these contexts and do not hallucinate features.""")
    print("messaggi in stream",response_stream.response_gen)
    #st.write_stream(response_stream.response_gen)
    print("messaggio di prima risposta da stamapre", response_stream.response)
    message = {"role": "assistant", "content": response_stream.response}
    st.session_state.messages.append(message)
    print_text()
    st.divider()
    st.text("Example of questions to ask:")
    st.text("How can I improve my CV?")
    st.text("What skills should I improve?")
    st.text("Is my job sector rising?")
    st.text("How can i write a better summary?")
    #remove_files("./CV/"+ uploaded_file.name)
    #for image_path in image_paths:
    #    remove_files(image_path)



@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing all the documentation – hang tight! This should take 1-2 minutes."):
        with st.expander('See process'):
            if not index_exists:
                st.text("Loading new documents...")
                docs = SimpleDirectoryReader(input_dir="./data/dataset").load_data()
                number_of_documents = len(docs)
                st.text(f"{number_of_documents} documents loaded")
                st.text("Preparing the index...")
                index = VectorStoreIndex.from_documents(docs, show_progress=True)
                index.storage_context.persist(persist_dir="persist_directory")
            else:
                st.text("Loading the index...")
                storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
                index = load_index_from_storage(storage_context)
                docs = SimpleDirectoryReader(input_dir="./data/dataset").load_data()

        st.text("Index is ready")
        return index

index = load_data()

#chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True, streaming=True,system_prompt=Settings.llm.system_prompt
    )

#st.text("Index is ready")
        



#if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#    st.session_state.chat_engine = st.session_state.chat_engine.as_chat_engine(
#        chat_mode="condense_question", verbose=True, streaming=True
#    )

#st.text('Ready...')

uploaded_file = st.file_uploader("Upload your CV (PDF only)", type="pdf")
if uploaded_file is not None:
    with st.spinner('Converting PDF to images...'):
        # Convert PDF to images
        print("Ora converto l'immagine")
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join("./CV", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Convert PDF to images
        #images = convert_from_path(temp_file_path,poppler_path='./poppler-24.02.0/Library/bin')
        #image_paths = []
        #for i, image in enumerate(images):
        #    image_path = f'page_{i}.jpg'
        #    image.save(image_path, 'JPEG')
        #    image_paths.append(image_path)
        
        file_path = temp_file_path
        with fitz.open(file_path) as doc:
            #doc = fitz.open(file_path)  # open document
            image_paths = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap()  # render page to an image
                image_path = f'page_{i}.jpg'
                pix.save(f"page_{i}.jpg")
                image_paths.append(image_path)
        
        

        concatenated_image = ""

        for image_path in image_paths:
            with open(image_path, "rb") as img_file:
                st.image(image_path, caption=f'Page {image_paths.index(image_path) + 1}')
                img_data = base64.b64encode(img_file.read()).decode()
                concatenated_image += img_data

         # Now you can use the concatenated_image variable as needed
        # Delete the temporary image files
        for image_path in image_paths:
            os.remove(image_path)

        # Delete the temporary PDF file
        try:
            os.remove(temp_file_path)
        except PermissionError:
            time.sleep(0.5)  # Wait for 0.2 seconds
            os.remove(temp_file_path)
        
    st.success('PDF converted to images successfully!')
    with st.spinner('Analyzing the CV...'):
        st.session_state.JSON_CV = extract_text_from_images(concatenated_image)
        
        #get_first_advice()



if prompt := st.chat_input("Ask a question about your CV"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    #print_text()







#for message in st.session_state.messages:  # Write message history to UI
#    with st.chat_message(message["role"]):
#        st.write(message["content"])




if st.session_state.messages[-1]["role"] != "assistant" and len(st.session_state.messages)>1:
    with st.spinner("Thinking..."):
        response_stream = st.session_state.chat_engine.chat(f"""You are an expert recruiter with a deep understanding of CV review and improvement.
            A user has asked the following question about their CV: "{prompt}".

            Here is the CV content in JSON format: {st.session_state.JSON_CV}.

            Provide a detailed, accurate, and specific response to the user's question.
            Your response should be based on the provided CV and use the information from the dataset referenced below.
            Make sure to:
            - Stick to the context of the user's question.
            - Provide technical and factual advice.
            - Do not make up features or information.
            - Use examples and actionable suggestions where possible.
            - Refer to the dataset for supporting information.

            Conversation history for context: {st.session_state.messages}.

            Dataset reference: {index}.
                                                                        """)
            #st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        st.session_state.messages.append(message)
        print_text()
        st.divider()
        st.text("Example of questions to ask:")
        st.text("How can I improve my CV?")
        st.text("What skills should I improve?")
        st.text("Is my job sector rising?")
        st.text("How can i write a better summary?")
            
    #print(conversation_history)



