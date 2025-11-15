from .graph_state import State
from typing import Literal

def route_after_rewrite(state: State) -> Literal["agent", "human_input"]:
    return "agent" if state.get("questionIsClear", False) else "human_input"