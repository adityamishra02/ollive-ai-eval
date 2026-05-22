import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv() 


frontier_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def generate_frontier_response(message, history):
    langchain_messages = []
    
    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
            
    langchain_messages.append(HumanMessage(content=message))
    
    try:
        reply = frontier_llm.invoke(langchain_messages).content
    except Exception as e:
        reply = f"API Error: {str(e)} (Check your .env file)"
        
    return reply
