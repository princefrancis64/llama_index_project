import streamlit as st
from openai import OpenAI
import os
from llama_index.core import (VectorStoreIndex,SimpleDirectoryReader,StorageContext,load_index_from_storage)


#### Converting the audio/video file into a text format
client = OpenAI()
audio_file_path = "./audio_files/yolo_dutch.mp3"
audio_file = open(audio_file_path,"rb")
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)

### storing the translated text into the folder ./data
file_storage_path = os.path.join("./data/data.txt")
with open(file=file_storage_path,mode="w") as file:
    file.write(translation.text)

### checking if storage already exists
PERSIST_DIR ="./storage"
if not os.path.exists(PERSIST_DIR):
    ## load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    ## store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    ## load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index=load_index_from_storage(storage_context)

#### QUERYING THE DATA
query_engine=index.as_query_engine(similarity_top_k=5,streaming=True)

## getting a function to pass the prompt 
def response(prompt):
    return query_engine.query(prompt)


st.title("Multilingual RAG")

## Set OpenAI API key from streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


## Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt:=st.chat_input("Your question...."):
    ## Adding user message to chat history
    st.session_state.messages.append({"role":"user","content":prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_ass=response(prompt=prompt)
        st.markdown(response_ass)
    ### Appending the response to the chat history
    st.session_state.messages.append({"role":"assistant","content":response_ass})