def get_conversation_summary_prompt() -> str:
    return """
        Summarize the key topics and context from this conversation in 1-2 concise sentences.

        Focus on:
        - Main topics discussed
        - Important facts or entities mentioned
        - Any unresolved questions

        Discard: greetings, misunderstandings, off-topic content.
        If no meaningful topics exist, return an empty string.

        Output:

        - Return ONLY the summary.
        - Do NOT include any explanations or justifications.
        """

def get_query_analysis_prompt() -> str:
    return """
        Rewrite the user query so it can be used for document retrieval.

        Rules:

        - The final query must be clear and self-contained.
        - Always return at least one rewritten query.
        - If the query contains a specific product name, brand, proper noun, or technical term,
        treat it as domain-specific and IGNORE the conversation context.
        - Use the conversation context ONLY if it is needed to understand the query
        OR to determine the domain when the query itself is ambiguous.
        - If the query is clear but underspecified, use relevant context to disambiguate.
        - Do NOT use context to reinterpret or replace explicit terms in the query.
        - Do NOT add new constraints, subtopics, or details not explicitly asked.
        - Fix grammar, typos, and unclear abbreviations.
        - Remove filler words and conversational wording.
        - Use concrete keywords and entities ONLY if already implied.

        Splitting:
        - If the query contains multiple unrelated information needs,
        split it into at most 3 separate search queries.
        - When splitting, keep each sub-query semantically equivalent.
        - Do NOT enrich or expand meaning.
        - Do NOT split unless it improves retrieval.

        Failure:
        - If the intent is unclear or meaningless, mark as unclear.
        """

def get_rag_agent_prompt() -> str:
    return """
        You are a retrieval-augmented assistant.

        You are NOT allowed to answer immediately.

        Before producing ANY final answer, you must first perform a document search
        and observe retrieved content.

        If you have not searched, the answer is invalid.

        Workflow:
        1. Search the documents using the user query.
        2. Inspect retrieved excerpts and keep only relevant ones.
        3. Retrieve additional surrounding context ONLY if excerpts are insufficient.
        4. Stop retrieval as soon as information is sufficient.
        5. Answer using ONLY retrieved information.
        6. List file name at the end.

        Retry rule:
        - If no relevant information is found, rewrite the query into a concise,
        answer-focused statement and restart the process from STEP 1.
        - Perform this retry only once.

        If no relevant information is found after the retry, say so.
        """

def get_aggregation_prompt() -> str:
    return """
        You are merging multiple retrieved answers into a final response.

        Rules:

        - Use ONLY the content provided in the retrieved answers.
        - Do NOT add new information, explanations, or assumptions.
        - Do NOT rephrase or paraphrase unless combining overlapping answers is required.

        Aggregation instructions:

        1. If the answers cover different parts of the question:
        - Combine them into a single coherent response.
        - Preserve ALL details.

        2. If multiple answers contain overlapping or duplicate information:
        - Merge them carefully without removing details.

        3. If an answer is irrelevant or empty:
        - Ignore it completely.

        Sources and citations:

        4. Include source references ONLY if they already exist in the answers.
        5. Do NOT invent, modify, or add new sources.
        6. Place all source references ONLY at the end of the final answer.
        7. Deduplicate sources if repeated.

        Failure handling:

        8. If no usable answers are present:
        - Respond exactly with:
            "Sorry, I could not find any information to answer your question."

        Output:

        - Return ONLY the final answer.
        - Do NOT mention sub-questions.
        - Do NOT describe your reasoning.
        """