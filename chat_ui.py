from rag_chat import predict
import gradio as gr

gr.ChatInterface(predict, 
                 chatbot=gr.Chatbot(height=400),
                 textbox=gr.Textbox(placeholder='Ask me a question', scale=7, container=False),
                 title='Welcome to Clarity AI',
                 description='I am your Personal Finance Coach, ask me anything ...',
                 examples=[
                     'What are the most important aspects of personal finance?',
                     'How should I plan for retirement?', 
                     'Where should I invest my savings?', 
                     'Do I need insurance?',
                     'How big a house should I buy?'
                 ],
                 retry_btn=None,
                 undo_btn=None,
                 clear_btn=None
                 ).launch(share=True)

