This repo contains code to create a RAG-LLM based chatbot.

Step 1: Prepare the data

    python create.db.py
    This creates the retrieval database using the urls and documents mentioned in the file

Step 2: Install local LLM
    
    Download and install ollama from ollama.com
    
    ollama pull llama3.1
    This installs the llama3.1 model locally

Step 3: Launch the chatbot

    python chat_ui.py
    This invokes a UI chatbot using a simple gradio based interface

Running chat as a server

    python chat_server.py
    This creates a rest API endpoint for chat using flask

    python chat_client.py
    This is a sample chat client which connects to chat_server
