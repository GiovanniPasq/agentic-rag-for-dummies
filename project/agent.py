# agent.py
"""
Defines the retrieval tools, system prompt,
and builds the agent LangGraph.
"""

import os
import json
from typing import List, Literal
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import config
from pydantic import BaseModel, Field
# Import initialized components
from components import llm, child_vector_store

# --- 1. Define Retrieval Tools ---

@tool
def search_child_chunks(query: str, k: int = 5) -> List[dict]:
    """
    Search for the top K most relevant child chunks.

    Args:
        query: Search query string
        k: Number of results to return

    Returns:
        List of dicts with content, parent_id, and source
    """
    try:
        results = child_vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "parent_id": doc.metadata.get("parent_id", ""),
                "source": doc.metadata.get("source", "")
            }
            for doc in results
        ]
    except Exception as e:
        print(f"Error searching child chunks: {e}")
        return []

@tool
def retrieve_parent_chunks(parent_ids: List[str]) -> List[dict]:
    """
    Retrieve full parent chunks by their IDs.

    Args:
        parent_ids: List of parent chunk IDs to retrieve

    Returns:
        List of dicts with content, parent_id, and metadata
    """
    unique_ids = sorted(list(set(parent_ids)))
    results = []

    for parent_id in unique_ids:
        file_path = os.path.join(config.PARENT_STORE_PATH, parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc_dict = json.load(f)
                    results.append({
                        "content": doc_dict["page_content"],
                        "parent_id": parent_id,
                        "metadata": doc_dict["metadata"]
                    })
            except Exception as e:
                print(f"Error loading parent chunk {parent_id}: {e}")

    return results

# --- 2. Bind Tools to LLM ---
tools = [search_child_chunks, retrieve_parent_chunks]
llm_with_tools = llm.bind_tools(tools)

# --- 3. Define Agent System Prompt ---
AGENT_SYSTEM_PROMPT = """
You are an intelligent assistant that MUST use the available tools to answer questions.

**MANDATORY WORKFLOW - Follow these steps for EVERY question:**

1. **ALWAYS start by calling `search_child_chunks`** with the user's query
   - Choose appropriate K value (default: 5)
   
2. **Read the retrieved chunks** and check if they answer the question
   
3. **If chunks are incomplete**, call `retrieve_parent_chunks` with parent_id values
   
4. **Answer using ONLY the retrieved information**
   - Cite source files from metadata
   
5. **If no relevant information found after searching**, try rephrasing the query once more

**CRITICAL**: You MUST call tools before answering. Never respond without first using `search_child_chunks`.
"""

agent_system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)

# --- 4. State Definition ---

class State(MessagesState):
    questionIsClear: bool
    conversation_summary: str = ""

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )


# --- 4. Define Graph Nodes ---

def analyze_chat_and_summarize(state: State):
    """
    Analyzes chat history and summarizes key points for context.
    """
    if len(state["messages"]) < 4:  # Need some history to summarize
        return {"conversation_summary": ""}
    
    # Extract relevant messages (excluding current query and system messages)
    relevant_msgs = [
        msg for msg in state["messages"][:-1]  # Exclude current query
        if isinstance(msg, (HumanMessage, AIMessage)) 
        and not getattr(msg, "tool_calls", None)
    ]
    
    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    summary_prompt = "Summarize the key topics and context from this conversation concisely (2-3 sentences max). Discard irrelevant information, such as misunderstandings or off-topic queries/responses. If there are no key topics, return an empty string:\n\n"
    for msg in relevant_msgs[-6:]:  # Last 6 messages for context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        summary_prompt += f"{role}: {msg.content}\n"
    
    summary_prompt += "\nBrief Summary:"    
    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=summary_prompt)])
    return {"conversation_summary": summary_response.content}

def analyze_and_rewrite_query(state: State):
    """
    Analyzes user query and rewrites it for clarity, optionally using conversation context.
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"**Conversation Context:**\n{conversation_summary}" 
        if conversation_summary.strip() 
        else "**Conversation Context:**\n[First query in conversation]"
    )
    
    # Create analysis prompt
    prompt = f"""Rewrite the user's query to be clear and optimized for information retrieval.
                **User Query:**
                `"{last_message.content}"`

                `{context_section}`

                **Instructions:**

                1.  **If this is a follow-up** (uses pronouns, references previous topics): Resolve all references using the context to make it a self-contained query.
                2.  **If it's a new independent query:** Ensure it's already clear and specific.
                3.  **If it contains multiple distinct questions:** Split them into a list of separate, focused questions.
                4.  **Use specific, keyword-rich language:** Remove vague terms and conversational filler (e.g., "tell me about" -> "facts about").
                5.  **Mark as unclear** if you cannot determine a clear user intent for information retrieval.
                    * This includes queries that are **nonsense/gibberish**, **insults**, or statements without an apparent question.
                """
    
    llm_with_structure = llm.with_config(temperature=0.2).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=prompt)])
    
    if response.is_clear:
        # Remove all non-system messages
        delete_all = [
            RemoveMessage(id=m.id) 
            for m in state["messages"] 
            if not isinstance(m, SystemMessage)
        ]
        
        # Format rewritten query
        rewritten = (
            "\n".join([f"{i+1}. {q}" for i, q in enumerate(response.questions)])
            if len(response.questions) > 1
            else response.questions[0]
        )
        return {
            "questionIsClear": True,
            "messages": delete_all + [HumanMessage(content=rewritten)]
        }
    else:
        clarification = response.clarification_needed or "I need more information to understand your question."        
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }

def human_input_node(state: State):
    """Placeholder node for human-in-the-loop interruption"""
    return {}

def route_after_rewrite(state: State) -> Literal["agent", "human_input"]:
    """Route to agent if question is clear, otherwise wait for human input"""
    return "agent" if state.get("questionIsClear", False) else "human_input"

def agent_node(state: State):
    """Main agent node that processes queries using tools"""
    messages = [SystemMessage(content=agent_system_message.content)] + state["messages"]
    response = llm_with_tools.with_config(temperature=0.1).invoke(messages)
    return {"messages": [response]}

# The ToolNode handles the execution of the tools
tool_node = ToolNode(tools)

# --- 5. Build the Graph ---
print("Compiling agent graph...")
# Initialize checkpointer
checkpointer = InMemorySaver()

# Create graph builder
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("summarize", analyze_chat_and_summarize)
graph_builder.add_node("analyze_rewrite", analyze_and_rewrite_query)
graph_builder.add_node("human_input", human_input_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "analyze_rewrite")
graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
graph_builder.add_edge("human_input", "analyze_rewrite")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

# Compile graph
agent_graph = graph_builder.compile(
    checkpointer=checkpointer, 
    interrupt_before=["human_input"]
)
print("✓ Agent graph compiled successfully.")