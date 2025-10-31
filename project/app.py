# app.py
"""
Main application that launches the Gradio user interface
to chat with the RAG agent.

Run with: python app.py
"""
import uuid
import gradio as gr
from langchain_core.messages import HumanMessage

# Import the compiled agent graph
from agent import agent_graph
# Import the directory initializer
from components import initialize_directories

def create_thread_id():
    """Generate a unique thread ID for each conversation"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def clear_session():
    """Clear thread for new conversation and clean up checkpointer state"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat_with_agent(message, history):
    """
    Handle chat with human-in-the-loop support.
    Returns: response text
    """
    print("Running new query...")
    current_state = agent_graph.get_state(config)    
    if current_state.next:        
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:        
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]}, config)
    return result['messages'][-1].content

config = create_thread_id()

# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        height=600,
        placeholder="<strong>Ask me anything!</strong><br><em>I'll search, reason, and act to give you the best answer :)</em>"
    )
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat_with_agent, type="messages", chatbot=chatbot)

print("\nLaunching application...")
demo.launch()