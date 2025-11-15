from langgraph.graph import MessagesState

class State(MessagesState):
    questionIsClear: bool
    conversation_summary: str = ""