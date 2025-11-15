from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from .graph_state import State
from .schemas import QueryAnalysis
from .prompts import *

def analyze_chat_and_summarize(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    summary_prompt = get_conversation_summary_prompt(relevant_msgs)
    summary_response = llm.with_config(temperature=0.3).invoke([SystemMessage(content=summary_prompt)])
    return {"conversation_summary": summary_response.content}

def analyze_and_rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    prompt = get_query_analysis_prompt(last_message.content, conversation_summary)

    llm_with_structure = llm.with_config(temperature=0.3).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=prompt)])

    if response.is_clear:
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]

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
    return {}

def agent_node(state: State, llm_with_tools):
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}