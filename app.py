import gradio as gr
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from oss_agent import workflow
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize Gemini instead of OpenAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyAwhLmmVWQVWmijo74igvSQc3AExtpgw64" 
frontier_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 2. Compile the Qwen LangGraph agent
oss_agent = workflow.compile()

def chat_with_both(message, history_oss, history_frontier):
    # Setup history states
    history_oss.append((message, None))
    history_frontier.append((message, None))
    yield history_oss, history_frontier, ""
    
    # Generate Qwen (OSS) Response
    state_messages = []
    for user_msg, bot_msg in history_oss[:-1]: 
        state_messages.append(HumanMessage(content=user_msg))
        if bot_msg:
            state_messages.append(AIMessage(content=bot_msg))
    state_messages.append(HumanMessage(content=message))
    
    oss_state = oss_agent.invoke({"messages": state_messages, "is_safe": True})
    oss_reply = oss_state["messages"][-1].content
    history_oss[-1] = (message, oss_reply)
    yield history_oss, history_frontier, ""
    
    # Generate OpenAI (Frontier) Response
    langchain_messages = []
    for user_msg, bot_msg in history_frontier[:-1]:
        langchain_messages.append(HumanMessage(content=user_msg))
        if bot_msg:
            langchain_messages.append(AIMessage(content=bot_msg))
    langchain_messages.append(HumanMessage(content=message))
    
    try:
        frontier_reply = frontier_llm.invoke(langchain_messages).content
    except Exception as e:
        frontier_reply = f"OpenAI Error: {str(e)} (Check your API key)"
        
    history_frontier[-1] = (message, frontier_reply)
    yield history_oss, history_frontier, ""

# 3. Gradio Dual-Pane Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Assistant Evaluation Arena")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Qwen 2.5 (Open Source)")
            chatbot_oss = gr.Chatbot(height=500, type="tuples")
        with gr.Column():
            gr.Markdown("### GPT-4o-mini (Frontier)")
            chatbot_frontier = gr.Chatbot(height=500, type="tuples")
            
    with gr.Row():
        msg = gr.Textbox(placeholder="Enter prompt here...", scale=4, show_label=False)
        submit_btn = gr.Button("Send", scale=1, variant="primary")
        
    with gr.Row():
        clear_btn = gr.ClearButton([chatbot_oss, chatbot_frontier, msg], value="Clear History")

    # Wire up the events
    submit_event = msg.submit(
        chat_with_both, 
        inputs=[msg, chatbot_oss, chatbot_frontier], 
        outputs=[chatbot_oss, chatbot_frontier, msg],
        api_name=False  # <-- Add this!
    )
    submit_btn.click(
        chat_with_both, 
        inputs=[msg, chatbot_oss, chatbot_frontier], 
        outputs=[chatbot_oss, chatbot_frontier, msg],
        api_name=False  # <-- Add this!
    )

if __name__ == "__main__":
    demo.launch() # (Keep this as demo.launch() to use the stable local link)
