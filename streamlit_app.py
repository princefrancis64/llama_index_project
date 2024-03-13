## Important libraries

import streamlit as st
from llama_index.core import VectorStoreIndex,ServiceContext,Document,SimpleDirectoryReader
# from llama_index.llms import OpenAI
from openai import OpenAI
import openai

## Initialize message history
openai.api_key = st.secrets.openai_key
st.header("Chat  💬 📚")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role":"assistant","content":"Ask me a question about the uploaded document!"}

    ]


## Load and index data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the doc – hang tight! This should take 1-2 minutes."):
        document = SimpleDirectoryReader(input_dir ="./data",recursive=True).load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo",temperature=0.2,system_prompt="You are an expert on the YOLO, the popular object detection algorithm that has revolutionized the field of computer vision and your job is to answer technical questions. Assume that all questions are related to YOLO only. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(document,service_context=service_context)
        return index

index=load_data()

## Creating a chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question",verbose=True)


## Prompt for user input and display message history
if prompt :=st.chat_input("Your Question"):
    st.session_state.messages.append({"role":"user","content":prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


## Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.message[-1]["role"]!="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message={"role":"assistant","content":response.response}
            st.session_state.messages.append(message)