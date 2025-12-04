from typing import Literal
from langgraph.types import Send
from .graph_state import State

def route_after_rewrite(state: State) -> Literal["human_input", "process_question"]:
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        return [
                Send("process_question", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]