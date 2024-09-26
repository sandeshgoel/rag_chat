import requests
import json
import sys

def chat_with_bot(message, server):
    if server == 'render':
        url = "https://rag-chat-0h9o.onrender.com/chat" 
    elif server == 'local':
        url = "http://127.0.0.1:5678/chat"
    else:
        return f'Error: unknown server {server}'
    
    headers = {"Content-Type": "application/json"}
    data = {"message": message}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    except Exception as e:
        return f'Exception: {e}'
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        
        server = sys.argv[1] if len(sys.argv)>1 else 'local'
        bot_response = chat_with_bot(user_input, server)
        print(f"Bot: {bot_response}")
