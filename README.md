# RAGChatPDF
## Conversational RAG With PDF uploads and chat history

This is a Streamlit application that enables a conversational chatbot experience over the content of user-uploaded PDF files. It uses a Retrieval Augmented Generation (RAG) architecture, allowing the chatbot to answer questions based on the uploaded documents while maintaining chat history.

## Features
PDF File Uploader: Users can upload one or more PDF documents to serve as the knowledge base.

Conversational Memory: The chatbot maintains a chat history, allowing it to understand and respond to follow-up questions with context.

Groq LLM Integration: Uses the high-speed Groq API for powerful and low-latency responses.

Local Vector Store: Utilizes ChromaDB to create and manage the vector database from document content on the fly.

Hugging Face Embeddings: Embeds the document content using a Hugging Face model for efficient similarity search.

## Getting Started
Prerequisites
You need to have Python installed. It is recommended to use a virtual environment.

## Installation
Clone the repository or save the code to a local directory.

Set .env and install dependencies from requirements.txt

streamlit run app.py
This will launch the application in your web browser. You can then upload a PDF and start your conversation with the document's content.
