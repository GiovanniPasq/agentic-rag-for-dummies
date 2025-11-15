from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *

def create_agent_graph(llm, tools_list):
    print("Compiling agent graph...")
    
    llm_with_tools = llm.bind_tools(tools_list)
    
    checkpointer = InMemorySaver()
    graph_builder = StateGraph(State)
    
    tool_node = ToolNode(tools_list)
    graph_builder.add_node("summarize", partial(analyze_chat_and_summarize, llm=llm))
    graph_builder.add_node("analyze_rewrite", partial(analyze_and_rewrite_query, llm=llm))
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]
    )
    print("✓ Agent graph compiled successfully.")
    return agent_graph