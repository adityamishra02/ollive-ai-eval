import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# It's best practice to load this from a .env file, but for rapid prototyping:
os.environ["OPENAI_API_KEY"] = "sk-..." # Replace with your actual key

# We'll use gpt-4o-mini to keep latency low and costs minimal during testing. 
# You can easily swap this to 'gpt-4o' for the final evaluation.
frontier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def generate_frontier_response(message, history):
    """
    Translates the Gradio history into LangChain messages, calls OpenAI, 
    and returns the string response.
    """
    langchain_messages = []
    
    # Rebuild the short-term memory from the Gradio history state
    for user_msg, bot_msg in history:
        langchain_messages.append(HumanMessage(content=user_msg))
        if bot_msg:
            langchain_messages.append(AIMessage(content=bot_msg))
            
    # Append the brand new user prompt
    langchain_messages.append(HumanMessage(content=message))
    
    # Generate the response via the OpenAI API
    response = frontier_llm.invoke(langchain_messages)
    
    return response.content