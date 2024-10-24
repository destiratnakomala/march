# importing necessary libraries
import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
 

# load the environment variables into the python script
load_dotenv() 
# fetching the openai_api_key environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')


# Initialize session states
if 'vectorDB' not in st.session_state:
    st.session_state.vectorDB = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'bot_name' not in st.session_state:
    st.session_state.bot_name = ''
if 'chain' not in st.session_state:
    st.session_state.chain = None


def get_pdf_text(pdf) -> str:
    """ This function extracts the text from the PDF file """
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def get_vectorstore(text_chunks):
    """ This function will create a vector database as well as create and store the embedding of the text chunks into the VectorDB """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_text_chunks(text: str):
    """ This function will split the text into the smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def processing(pdf):
    """This function divides the PDF into smaller chunks and saves these segmented chunks in a vector database. And return the Vector Database"""
    
    # getting all the raw text from the PDF
    raw_text = get_pdf_text(pdf)

    # divinding the raw text into smaller chunks
    text_chunks = get_text_chunks(raw_text)

    # Creating and storing the chunks in vector database
    vectorDB = get_vectorstore(text_chunks)

    return vectorDB

def get_response(query: str):
    """This function will return the output of the user query! """
    
    # getting the context from the database that is similar to the user query
    query_context = st.session_state.vectorDB.similarity_search(query=query,k=4)
    # calling the chain to get the output from the LLM
    response = st.session_state.chain.invoke({'human_input':query,'context':query_context,'name':st.session_state.bot_name})['text']
    # Iterate through each word in the 'response' string after splitting it based on whitespace
    for word in response.split():
        # Yield the current word followed by a space, effectively creating a generator
        yield word + " "
    
        # Pause execution for 0.05 seconds (50 milliseconds) to introduce a delay
        time.sleep(0.05)

def get_conversation_chain(vectorDB):
    """ This function will create and return a LLM-Chain"""
    
    # using OPENAI LLM
    llm = OpenAI(temperature=0.4)

    # creating a template to pass into LLM
    template = """You are a Personalized ChatBot with a name: {name} for a company's customer support system, aiming to enhance the customer experience by providing tailored assistance and information.

    Answer the question as detailed as possible and to the point from the context: {context}\n , and  make sure to provide all the information, if the answer is not in the provided context just say, "answer is not available in the context", do not provide the wrong answer\n\n

    {chat_history}
    Human: {human_input}
    AI: """

    # creating a prompt that is used to format the input of the user
    prompt = PromptTemplate(template = template,input_variables=['chat_history','human_input','name','context'])

    # creating a memory that will store the chat history between chatbot and user
    memory = ConversationBufferWindowMemory(memory_key='chat_history',input_key="human_input",k=5)

    chain = LLMChain(llm=llm,prompt=prompt,memory=memory,verbose=True) 

    return chain



if __name__ =='__main__':
    #setting the config of WebPage
    st.set_page_config(page_title="Personalized ChatBot",page_icon="🤖")
    st.header('Personalized Customer Support Chatbot 🤖',divider='rainbow')
    
    # taking input( bot name and pdf file) from the user
    with st.sidebar:
        st.caption('Please enter the **Bot Name** and Upload **PDF** File!')

        bot_name = st.text_input(label='Bot Name',placeholder='Enter the bot name here....',key="bot_name")
        file = st.file_uploader("Upload a PDF file!",type='pdf')

        # moving forward only when both the inputs are given by the user
        if file and bot_name:
            # the Process File button will process the pdf file and save the chunks into the vector database
            if st.button('Process File'):
                # if there is existing chat history we will delete it 
                if st.session_state.messages != []:
                    st.session_state.messages = []
                
                with st.spinner('Processing.....'):
                    st.session_state['vectorDB'] = processing(file)
                    st.session_state['chain'] = get_conversation_chain(st.session_state['vectorDB'])
                st.write('File Processed')
                

    # if the vector database is ready to use then only show the chatbot interface
    if st.session_state.vectorDB: 
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # taking the input i.e. query from the user (walrus operator)
        if prompt := st.chat_input(f"Message {st.session_state.bot_name}"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.write(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(get_response(prompt))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
