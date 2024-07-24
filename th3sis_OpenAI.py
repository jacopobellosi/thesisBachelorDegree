import streamlit as st
import numpy as np
import pandas as pd
import openai
import requests
import json
from openai import OpenAI
import hmac
from io import StringIO
import os
from pdf2image import convert_from_path
import base64
import os
from os.path import exists
import ast
import pdb
import sys
import fitz  # PyMuPDF
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity

prompt=""
conversation_history=[]

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
st.text("The assistant is based on the OpenAI API")
st.text("!Important: you have to upload your CV in order the let the AI work properly")
st.divider()
openai.api_key = st.secrets.openai_key
client = OpenAI(api_key=st.secrets.openai_key,project="proj_TSOrJg10Snkt4UuWMSPHpgpQ")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Start your consultation about your CV. Upload your CV in PDF format and ask a question about it.",
        }
    ]

#st.text('Preparing the model...')


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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
        "model": "gpt-4o",
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
    #print(response)
            
    response = {"role": "assistant", "content": response}
    st.text('Loading your data...')
    context1,context2,context3 = load_data(response["content"])
    st.session_state.context1=context1
    st.session_state.context2=context2
    st.session_state.context3=context3
    #st.text('Preparing the engine...')
    if len(st.session_state.messages)<2:
        get_first_advice(response["content"])
            
    conversation_history.append(response)
    #print(conversation_history)
    #st.session_state.messages.append(message)
    #response_stream = st.session_state.chat_engine.stream_chat(prompt)
    #st.write_stream(response_stream.response_gen)
    #message = {"role": "assistant", "content": response_stream.response}
    #st.session_state.messages.append(message)
            
    return response["content"]

def get_embedding(text,model="text-embedding-3-small"):
    text=text.replace("\n"," ")
    return client.embeddings.create(input=[text],model=model).data[0].embedding

def fn(page_embedding):
    return np.dot(page_embedding,CV_ambedded)

def get_first_advice(JSON_CV):
    #st.write(JSON_CV)
    print("ti do ora un primo consiglio")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
        }
    payload = {
        "model": "gpt-4o-mini",
        "temperature":0.2,
        "messages": [
             {"role":"system", "content": """"You're an expert recruiter, you have receive my CV as a JSON and you will have to 
                                 give me some technical advice on how to improve it. Don't show the JSON file, only show text!
                
                Keep your answers technical, academic 
                languages and based on 
                facts.  do not hallucinate features.
                Give also the references from where you are taking the information, use maninly the dataset i am giving you.
                """},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "How can i improve it?, suggest me some technical advice. Give also at the end some career adive ONLY based on the dataset i am giving you ."
                },
                {
                "type": "text",
                "text": JSON_CV
                }
            ]
            },
            {"role":"assistant","content":f"""
                            Use this information from my datasets to answare the user question and to give more information:
                            In this dataset, you have the columns:Occupation, Skill_Name, Frequency_in_the_market:{st.session_state.context1}.
                            In this dataset, you have the columns: Skills_Label,Growth Category,Mid Scenario:{st.session_state.context2},
                            In this dataset, you have the columns: Skill_Name,Salary Premium,Skill Frequency,Averege job postings duration (days),Difficulty to fill in the market,Difficulty,GROWTH PROJECTION{st.session_state.context3}.
                            Stick to these contexts to answering the question, do not hallucinate features."""},
        ],
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    #print(response)
    response = response['choices'][0]['message']['content']
    #st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    st.divider()
    st.text("Example of prompts:")
    st.text("How can I improve my CV?")
    st.text("What skills should I improve?")
    st.text("Is my job sector rising?")
    #remove_files("./CV/"+ uploaded_file.name)
    #for image_path in image_paths:
    #    remove_files(image_path)



@st.cache
def load_data(JSON_CV):
    print("Carico i dati")
    with st.expander('See process'):
        #st.text("loading....")
        st.text("Loading the first dataset...")
        def fn(page_embedding):
            return np.dot(page_embedding, CV_embedded)
        CV_embedded = get_embedding(JSON_CV)

        df= pd.read_pickle("data/Book1.pkl")
        df["distance"]=df["embedding"].apply(fn)
        df.sort_values('distance',ascending=False,inplace=True)
        distance_series = df['embedding'].apply(fn)
        top_ten= distance_series.sort_values(ascending=False).index[0:10]
        text_series = df.loc[top_ten, ["Occupation", "Skill_Name", "Frequency_in_the_market"]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        context1 = "\n".join(text_series)
        print("Context1",context1)

        st.text("Loading the second dataset...")
        df2= pd.read_pickle("data/Book2.pkl")
        df2["distance"]=df2["embedding"].apply(fn)
        df2.sort_values('distance',ascending=False,inplace=True)
        distance_series = df2['embedding'].apply(fn)
        top_ten= distance_series.sort_values(ascending=False).index[0:10]
        text_series = df2.loc[top_ten, ["Skills_Label","Growth Category","Mid Scenario"]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        context2="\n".join(text_series)
        print("Context2",context2)

        st.text("Loading the third dataset...")
        df3= pd.read_pickle("data/Book3.pkl")
        df3["distance"]=df3["embedding"].apply(fn)
        df3.sort_values('distance',ascending=False,inplace=True)
        distance_series = df3['embedding'].apply(fn)
        top_ten= distance_series.sort_values(ascending=False).index[0:10]
        text_series = df3.loc[top_ten, ["Skill_Name","Salary Premium","Skill Frequency","Averege job postings duration (days)","Difficulty to fill in the market","Difficulty","GROWTH PROJECTION "]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        context3="\n".join(text_series)
        print("Context3",context3)



        st.text("Index is ready")
        
    #return index,docs
    return context1,context2,context3



#if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
#    st.session_state.chat_engine = st.session_state.chat_engine.as_chat_engine(
#        chat_mode="condense_question", verbose=True, streaming=True
#    )

st.text('Ready...')


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
        doc = fitz.open(file_path)  # open document
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
        os.remove(temp_file_path)
    st.success('PDF converted to images successfully!')
    with st.spinner('Analyzing the CV...'):
        st.session_state.JSON_CV = extract_text_from_images(concatenated_image)
        CV_ambedded = get_embedding(st.session_state.JSON_CV)
        
        #get_first_advice()





if prompt := st.chat_input("Ask a question about your CV"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("Ora elaboro ciò che hai scritto: "+prompt)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "temperature":0.3,
        "messages": [
            {"role":"system", "content": """"You're an expert recruiter, you have receive my CV as a JSON and you will have to 
                                 give me some technical advice on how to improve it. Don't show the JSON file, only show text!
                
                Keep your answers technical, academic 
                languages and based on 
                facts.  do not hallucinate features.
                Give also the references from where you are taking the information, use maninly the dataset i am giving you.
                """},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt+", stick to the question, giving me only respose of text and be precise."
                },
                {
                "type": "text",
                "text": st.session_state.JSON_CV
                }
            ]
            },
            {"role":"assistant","content":f"Use this information from my datasets to answare the user question and to give more information:{st.session_state.context1},{st.session_state.context2},{st.session_state.context3}.Stick to these contexts to answering the question, do not hallucinate features."},
            {"role":"assistant","content":f"this is the history: {st.session_state.messages}"}
        ],
        "max_tokens": 1000
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    response = response['choices'][0]['message']['content']
    #st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    conversation_history.append(message)
    #print(conversation_history)

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

