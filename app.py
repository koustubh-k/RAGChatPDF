import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from dotenv import load_dotenv

# --- Environment Setup ---
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Streamlit App Layout ---
st.set_page_config(page_title="RAG Chatbot with PDF", layout="wide")
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password", value=groq_api_key)

# Check if an API key is provided before proceeding
if not api_key:
    st.warning("Please enter the Groq API Key to proceed.")
    st.stop()

# --- Caching and Initialization ---
@st.cache_resource
def get_llm(key):
    """Initializes and caches the LLM."""
    return ChatGroq(groq_api_key=key, model_name="gemma2-9b-it")

@st.cache_resource
def get_embeddings():
    """Initializes and caches the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def create_vector_embedding(files, _embeddings_model):  # <-- ADDED UNDERSCORE HERE
    """Loads, splits, and creates a FAISS vector store from uploaded PDFs."""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
    
    if not documents:
        st.error("No text could be extracted from the uploaded PDF files. They might be empty or in an unsupported format.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        st.error("No document chunks could be created. Please check the content of the PDF.")
        return None

    return FAISS.from_documents(documents=splits, embedding=_embeddings_model) # <-- USE THE NEW PARAMETER NAME HERE

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Manages chat history for a session."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --- Main App Logic ---
session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        st.session_state.vectorstore = create_vector_embedding(uploaded_files, get_embeddings())
        if st.session_state.vectorstore:
            st.success("Documents processed and ready for Q&A!")
            st.session_state.store[session_id] = ChatMessageHistory()
            st.session_state.chat_messages = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.get("vectorstore"):
    llm = get_llm(api_key)
    retriever = st.session_state.vectorstore.as_retriever()
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise. \n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    q_a_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, q_a_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.chat_input("Ask a question about the document:")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable":{"session_id":session_id}}
                )
                st.write(response['answer'])
                st.session_state.chat_messages.append({"role": "assistant", "content": response['answer']})
else:
    st.info("Please upload PDF files and click 'Process Documents' to start chatting.")
