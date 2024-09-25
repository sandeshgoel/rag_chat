import requests
import json

def chat_with_bot(message):
    url = "http://127.0.0.1:5000/chat"
    headers = {"Content-Type": "application/json"}
    data = {"message": message}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
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
        
        bot_response = chat_with_bot(user_input)
        print(f"Bot: {bot_response}")
