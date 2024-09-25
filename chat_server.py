from flask import Flask, request, jsonify
import random
from rag_chat import predict

app = Flask(__name__)

# Simple responses for the chatbot
responses = [
    "Hello! How can I help you today?",
    "That's an interesting question.",
    "I'm not sure I understand. Could you please rephrase that?",
    "I'm here to assist you with any questions you may have.",
    "That's a great point!",
]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    
    # In a real chatbot, you would process the message here
    # For this example, we'll just return a random response
    response = predict(message) #random.choice(responses)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)