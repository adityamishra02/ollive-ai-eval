import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from oss_agent import workflow
from frontier_agent import generate_frontier_response

oss_agent = workflow.compile()

def chat_with_both(message, history_oss, history_frontier):
    history_oss.append({"role": "user", "content": message})
    history_frontier.append({"role": "user", "content": message})
    yield history_oss, history_frontier, ""
    
    state_messages = []
    for msg in history_oss[:-1]: 
        if msg["role"] == "user":
            state_messages.append(HumanMessage(content=msg["content"]))
        else:
            state_messages.append(AIMessage(content=msg["content"]))
    state_messages.append(HumanMessage(content=message))
    
    oss_state = oss_agent.invoke({"messages": state_messages, "is_safe": True})
    oss_reply = oss_state["messages"][-1].content
    
    history_oss.append({"role": "assistant", "content": oss_reply})
    yield history_oss, history_frontier, ""
    
    frontier_reply = generate_frontier_response(message, history_frontier[:-1])
    history_frontier.append({"role": "assistant", "content": frontier_reply})
    yield history_oss, history_frontier, ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Assistant Evaluation Arena")
    gr.Markdown("Comparing Qwen2.5-0.5B (Local OSS) against Gemini 2.5 Flash (Hosted Frontier).")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Qwen 2.5 (Open Source)")
            chatbot_oss = gr.Chatbot(height=500)
        with gr.Column():
            gr.Markdown("### Gemini 2.5 Flash (Frontier)")
            chatbot_frontier = gr.Chatbot(height=500)
            
    with gr.Row():
        msg = gr.Textbox(placeholder="Enter prompt here...", scale=4, show_label=False)
        submit_btn = gr.Button("Send", scale=1, variant="primary")
        
    with gr.Row():
        clear_btn = gr.ClearButton([chatbot_oss, chatbot_frontier, msg], value="Clear History")

    submit_event = msg.submit(
        chat_with_both, 
        inputs=[msg, chatbot_oss, chatbot_frontier], 
        outputs=[chatbot_oss, chatbot_frontier, msg],
        api_name=False
    )
    submit_btn.click(
        chat_with_both, 
        inputs=[msg, chatbot_oss, chatbot_frontier], 
        outputs=[chatbot_oss, chatbot_frontier, msg],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch()
