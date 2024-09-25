from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chat import predict

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "You have reached the clarity chat server."

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'GET':
        return "You have reached the clarity chat end point."
    
    data = request.json
    message = data.get('message', '')
    print(f'Message received: {message}')
    
    response = predict(message)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5678)