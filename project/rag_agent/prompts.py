from langchain_core.messages import HumanMessage

AGENT_SYSTEM_PROMPT = """
You are an intelligent assistant that MUST use the available tools to answer questions.

**MANDATORY WORKFLOW — Follow these steps for EVERY question:**

1. **Call `search_child_chunks`** with the user's query (K = 3–7).

2. **Review the retrieved chunks** and identify the relevant ones.

3. **For each relevant chunk, call `retrieve_parent_chunks`** using its parent_id to get full context.

4. **If the retrieved context is still incomplete, retrieve additional parent chunks** as needed.

5. **If metadata helps clarify or support the answer, USE IT**  

6. **Answer using ONLY the retrieved information**
   - Cite source files from metadata.

7. **If no relevant information is found,** rewrite the query into an **answer-focused declarative statement** and search again **only once**.
"""

def get_conversation_summary_prompt(messages):
    """Generate a prompt for conversation summarization."""
    summary_prompt = """**Summarize the key topics and context from this conversation concisely (1-2 sentences max).**
    Discard irrelevant information, such as misunderstandings or off-topic queries/responses.
    If there are no key topics, return an empty string.

    """
    
    for msg in messages[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        summary_prompt += f"{role}: {msg.content}\n"

    summary_prompt += "\n**Brief Summary:**"
    return summary_prompt

def get_query_analysis_prompt(query: str, conversation_summary: str = "") -> str:
    """Generate a prompt for query analysis and rewriting."""
    context_section = (
        f"**Conversation Context:**\n{conversation_summary}"
        if conversation_summary.strip()
        else "**Conversation Context:**\n[First query in conversation]"
    )

    return f"""
**Rewrite the user's query** to be clear, self-contained, and optimized for information retrieval.

**User Query:**
"{query}"

{context_section}

**Instructions:**

1. **Resolve references for follow-ups:** 
   - If the query uses pronouns or refers to previous topics, use the context to make it self-contained.

2. **Ensure clarity for new queries:** 
   - Make the query specific, concise, and unambiguous.

3. **Correct errors and interpret intent:** 
   - If the query is grammatically incorrect, contains typos, or has abbreviations, correct it and infer the intended meaning.

4. **Split only when necessary:** 
   - If multiple distinct questions exist, split into **up to 3 focused sub-queries** to avoid over-segmentation.
   - Each sub-query must still be meaningful on its own.

5. **Optimize for search:** 
   - Use **keywords, proper nouns, numbers, dates, and technical terms**. 
   - Remove conversational filler, vague words, and redundancies.
   - Make the query concise and focused for information retrieval.

6. **Mark as unclear if intent is missing:** 
   - This includes nonsense, gibberish, insults, or statements without an apparent question.
"""