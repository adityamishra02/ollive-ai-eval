import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ==========================================
# 1. Hardware Detection
# ==========================================
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🚀 Initializing Qwen2.5-0.5B-Instruct on: {device.upper()}")

# ==========================================
# 2. Load Model & Tokenizer
# ==========================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device != "cpu" else torch.float32, # Fixed the deprecation warning
    device_map=device
)

# ==========================================
# 3. LangGraph State & Nodes
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The conversation history"]
    is_safe: bool

def safety_guardrail(state: AgentState):
    """Checks the latest user message for prompt injection."""
    latest_message = state["messages"][-1].content.lower()
    unsafe_keywords = ["ignore previous instructions", "jailbreak", "bypass", "system prompt", "napalm"]
    is_safe = not any(keyword in latest_message for keyword in unsafe_keywords)
    return {"is_safe": is_safe}

def generate_response(state: AgentState):
    """Calls Qwen 2.5 using the full conversation history."""
    messages = state["messages"]
    
    # Format LangChain messages for Qwen
    chat_history = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        if role in ["user", "assistant"]:
            chat_history.append({"role": role, "content": msg.content})
        
    formatted_prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Return only the newly generated text
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"messages": [AIMessage(content=response_text)]}

def safety_refusal(state: AgentState):
    """Fallback if the guardrail catches an unsafe prompt."""
    return {"messages": [AIMessage(content="I cannot fulfill this request due to safety guidelines.")]}

# ==========================================
# 4. Build and Compile the Graph (The 'workflow')
# ==========================================
def route_based_on_safety(state: AgentState):
    if state["is_safe"]:
        return "generate_response"
    return "safety_refusal"

workflow = StateGraph(AgentState)

workflow.add_node("guardrail", safety_guardrail)
workflow.add_node("generate_response", generate_response)
workflow.add_node("safety_refusal", safety_refusal)

workflow.set_entry_point("guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_based_on_safety,
    {
        "generate_response": "generate_response",
        "safety_refusal": "safety_refusal"
    }
)

workflow.add_edge("generate_response", END)
workflow.add_edge("safety_refusal", END)