import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from load_docs import load_docs
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Llamada OpenAI API
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Cargar archivos
documents = load_docs()
chat_history = []

# Dividimos los datos en trozos
text_splitter = CharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=10
)
docs = text_splitter.split_documents(documents)

# Crear db vectorial chromadb
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)


# Streamlit frontend
st.title("LexConstitucionalBot usado con Langchain")
st.header("Pregunta cualquier consulta sobre derecho... 🤖")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_query():
    input_text = st.chat_input("Haga una pregunta sobre la constitución...")
    return input_text


# Recuperar la entrada del usuario
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i)+ '_user')
        message(st.session_state['generated'][i], key=str(i))
