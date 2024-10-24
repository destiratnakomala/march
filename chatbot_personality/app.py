
# Import necessary libraries
import streamlit as st
from langchain.chat_models import ChatOpenAI
# from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from typing_extensions import Concatenate

from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

import os

# Initialize session states
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "bot_details" not in st.session_state:
    st.session_state["bot_details"] = ""
if "bot_name" not in st.session_state:
    st.session_state["bot_name"] = ""
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = ""
if "is_created" not in st.session_state:
    st.session_state["is_created"] = False
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "file" not in st.session_state:
    st.session_state["file"] = None


template = """As a highly intelligent and powerful chatbot, your personality is shaped by the following description:
{description}

Your designated name is {name}. Initiate the conversation with a warm greeting only when the user asks something for the first time. Subsequent interactions can skip the greeting.

Your generated responses should be comprehensive, utilizing pointers when necessary for an enhanced user experience. Provide detailed answers, and if the user's input isn't related to personality, respond politely with an apology, encouraging them to ask questions related to your established personality. Keep your responses concise and accurate, seeking additional information from the user when required.

Incorporate the provided context and any relevant information from the chat history into your responses. If the user's input is related to the context or question, articulate your answer accordingly.

{context}
{chat_history}
Human: {input}
Assistant:"""

st.set_page_config(page_title=' 🤖ChatGPT with Memory🧠', layout='wide')


prompt = PromptTemplate(
    input_variables=["name", "description",
                     "context", "input", "chat_history"],
    template=template
)

chatgpt_chain = LLMChain(
    llm=OpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.7,
               api_key=os.environ.get('OPEN_API_KEY')),
    prompt=prompt,
    verbose="true",
    memory=ConversationBufferWindowMemory(
        memory_key="chat_history", input_key="input", k=5),
)


def genrate_result():
    if len(st.session_state["input"]) > 0:
        db = st.session_state["vectorstore"]
        result = db.similarity_search(st.session_state["input"])
        inputs = {"input": st.session_state["input"],
                  "description": st.session_state["bot_details"],
                  "name": st.session_state["bot_name"],
                  "context": result[0].page_content
                  }
        output = chatgpt_chain.run(inputs)
        st.session_state.past.append(st.session_state["input"])
        st.session_state.generated.append(output)
        st.session_state["input"] = ""


# function to delete the chatbot
def delete_bot():
    """
    Clears session state and starts a new chat.
    """
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["bot_details"] = ""
    st.session_state["bot_name"] = ""
    st.session_state["raw_text"] = ""
    st.session_state["is_created"] = False
    st.session_state["vectorstore"] = None
    st.session_state["is_file_uploded"] = False


#  set up the stram lit user inputs in slider bar
with st.sidebar:
    with st.form("my_form"):
        name = st.text_input('Name', key='name',
                             type="default", placeholder='Bot Name')
        details = st.text_area(
            "Enter Description", placeholder='Bot Description', key='description', height=100)
        file = st.file_uploader('Document', type='pdf')

        submitted = st.form_submit_button("Create Bot")

        if submitted:
            if file and name and details:
                st.session_state["bot_details"] = details
                st.session_state["bot_name"] = name
                loader = PdfReader(file)
                for i, page in enumerate(loader.pages):
                    content = page.extract_text()
                    if content:
                        temp = st.session_state["raw_text"]
                        st.session_state["raw_text"] = temp+content
                text_splitter = CharacterTextSplitter(
                            separator='\n', chunk_size=600, chunk_overlap=150, length_function=len)
                texts = text_splitter.split_text(
                            st.session_state["raw_text"])
                st.session_state["vectorstore"] = Chroma().from_texts(texts, embedding=OpenAIEmbeddings(
                            openai_api_key=os.environ.get('OPEN_API_KEY')))
                st.session_state["is_created"] = True
            else:
                st.warning(
                    'Name ,Description and File are required to create chatbot', icon="⚠️")

    if st.session_state["is_created"] == True:
        st.button('Delete Bot', on_click=delete_bot)


# Set up the Streamlit app layout
st.title("🤖 Personality Chatbot 🧠")
st.subheader(" Powered by Coffeebeans")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        # overflow: auto;
        # max-height: 300px;
       }

       </style>
       
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


if st.session_state["is_created"] == True:
    st.text_input("You: ", st.session_state["input"], key="input",
                  placeholder="Your AI assistant here! Ask me anything ...",
                  on_change=genrate_result(),
                  label_visibility='hidden')

with st.container():
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.success(st.session_state["generated"][i], icon="🤖")
        st.info(st.session_state["past"][i], icon="🧐")
