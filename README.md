<p align="center">
<img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="300px">
</p>

<h1 align="center">Agentic RAG for Dummies</h1>

<p align="center">
  <strong>Build a production-ready Agentic RAG system with LangGraph, conversation memory, and human-in-the-loop query clarification</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#llm-provider-configuration">LLM Providers</a> •
  <a href="#implementation">Implementation</a> •
  <a href="#installation--usage">Installation & Usage</a> •
  <a href="#upcoming-features">Upcoming Features</a>
</p>

<p align="center">
  <strong>Quickstart here 👉</strong> 
  <a href="https://colab.research.google.com/gist/GiovanniPasq/3fc8f09cd88170fe69c2eb62f08c505e/agentic_rag_for_dummies.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <img alt="Agentic RAG Demo" src="assets/newDemo.gif" width="650px">
</p>

<p align="center">
  <strong>If you like this project, a star ⭐️ would mean a lot :)</strong>
</p>

<p align="center" style="line-height: 1.6;">
  <em>✨ <strong>New:</strong> Comprehensive PDF → Markdown conversion guide, including tool comparisons and VLM-based approaches.</em><br>
  <em>🚀 <strong>Coming end of November:</strong> End-to-end Gradio interface for an interactive RAG pipeline.</em>
</p>

---

## Why This Repo?

While many RAG examples exist, most are simple pipelines that force a trade-off: use small chunks for search precision *or* large chunks for answer context. They lack the intelligence to adapt and understand conversation context.

This repository provides a production-ready **Agentic RAG** system that combines:

- **Hierarchical (Parent/Child) indexing** for precision and context
- **Conversation memory** to understand follow-up questions
- **Human-in-the-loop query clarification** to resolve ambiguous queries
- **LangGraph-powered reasoning** that evaluates, searches, and self-corrects
- **Provider-agnostic design** - develop with Ollama, deploy with any cloud LLM

---

## Overview

This repository demonstrates how to build an **Agentic RAG (Retrieval-Augmented Generation)** system using LangGraph with minimal code. It implements:

- 🔍 **Hierarchical Indexing**: Search small, specific chunks (Child) for precision, retrieve larger Parent chunks for context
- 💬 **Conversation Memory**: Maintains context across multiple questions for natural dialogue
- 🔄 **Query Clarification**: Automatically rewrites ambiguous queries or asks for clarification
- 🧠 **Intelligent Evaluation**: Assesses relevance at the granular chunk level
- 🤖 **Agent Orchestration**: Uses LangGraph to coordinate the entire workflow
- ✅ **Self-Correction**: Re-queries if initial results are insufficient

This approach combines the **precision of small chunks** with the **contextual richness of large chunks**, while understanding conversation flow and resolving unclear queries.

---

## How It Works

The system uses a **four-stage intelligent workflow**:

```
User Query → Conversation Analysis → Query Clarification →
Agent Reasoning → Search Child Chunks → Evaluate Relevance →
(If needed) → Retrieve Parent Chunks → Generate Answer → Return Response
```

### Stage 1: Conversation Understanding

Before processing any query, the system:
1. **Analyzes recent conversation history** to extract key topics and context
2. **Summarizes relevant information** from previous exchanges (2-3 sentences max)
3. **Maintains conversational continuity** across multiple questions

### Stage 2: Query Clarification

The system intelligently processes the user's query:
1. **Resolves references** - Converts "How do I update it?" → "How do I update SQL?"
2. **Splits complex questions** - Breaks multi-part questions into focused sub-queries
3. **Detects unclear queries** - Identifies nonsense, insults, or vague questions
4. **Requests clarification** - Uses human-in-the-loop to pause and ask for details
5. **Rewrites for retrieval** - Optimizes query with specific, keyword-rich language

### Stage 3: Hierarchical Indexing

Documents are split twice:
- **Parent Chunks**: Large sections based on Markdown headers (H1, H2, H3)
- **Child Chunks**: Small, fixed-size pieces (500 chars) derived from parents

Storage:
- **Child Chunks** → Qdrant vector database (hybrid dense + sparse embeddings)
- **Parent Chunks** → JSON file store (retrieved by ID)

### Stage 4: Intelligent Retrieval

1. Agent searches child chunks for precision
2. Evaluates if results are sufficient
3. Fetches parent chunks for context if needed
4. Generates answer from complete information
5. Self-corrects and re-queries if insufficient (max 3 attempts)

---

## LLM Provider Configuration

This system is **provider-agnostic** - you can use any LLM supported by LangChain. Choose the option that best fits your needs:

### Option 1: Ollama (Local - Recommended for Development)

**Install Ollama and download the model:**

```bash
# Install Ollama from https://ollama.com
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

**Python code:**

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0.1)
```

---

### Option 2: Google Gemini (Cloud - Recommended for Production)

**Install the package:**

```bash
pip install -qU langchain-google-genai
```

**Python code:**

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.1)
```

---

### Option 3: OpenAI (Cloud)

**Install the package:**

```bash
pip install -qU langchain-openai
```

**Python code:**

```python
import os
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
```

---

### Option 4: Anthropic Claude (Cloud)

**Install the package:**

```bash
pip install -qU langchain-anthropic
```

**Python code:**

```python
import os
from langchain_anthropic import ChatAnthropic

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
```

---

### Important Notes

- **Temperature 0.1-0.2** is used for consistent outputs while allowing slight reasoning flexibility
- **All providers** work with the exact same code - only the LLM initialization changes
- **API keys** should be stored securely using environment variables
- **Cost considerations:** Cloud providers charge per token, while Ollama is free but requires local compute

**💡 Recommendation:** Start with Ollama for development, then switch to Google Gemini or OpenAI for production.

---

## Implementation

### Step 1: Initial Setup and Configuration

Define paths and initialize core components.

```python
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

# Configuration
DOCS_DIR = "docs"  # Directory containing your .md files
PARENT_STORE_PATH = "parent_store"  # Directory for parent chunk JSON files
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0.1)

# Dense embeddings for semantic understanding
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Sparse embeddings for keyword matching
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# Qdrant client (local file-based storage)
client = QdrantClient(path="qdrant_db")
```

---

### Step 2: Configure Vector Database

Set up Qdrant to store child chunks with hybrid search capabilities.

```python
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

# Get embedding dimension
embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    """Create Qdrant collection if it doesn't exist"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"✓ Collection already exists: {collection_name}")

ensure_collection(CHILD_COLLECTION)

# Initialize vector store for child chunks
child_vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)
```

---

### Step 3: Hierarchical Document Indexing

Process documents with the Parent/Child splitting strategy.

```python
import glob
import json
from langchain_text_splitters import (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter)

def index_documents():
    """Index documents using hierarchical Parent/Child strategy"""
    
    # Parent splitter: by Markdown headers
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter( headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Child splitter: by character count
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    all_child_chunks = []
    all_parent_pairs = []
    
    # Process each document
    md_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not md_files:
        return
    
    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        
        with open(doc_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        
        # Split into parent chunks
        parent_chunks = parent_splitter.split_text(md_text)
        
        for i, p_chunk in enumerate(parent_chunks):
            p_chunk.metadata["source"] = str(doc_path)
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata["parent_id"] = parent_id
            
            all_parent_pairs.append((parent_id, p_chunk))
            child_chunks = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(child_chunks)
    
    # Save child chunks to Qdrant
    if all_child_chunks:
        child_vector_store.add_documents(all_child_chunks)
    
    # Save parent chunks to JSON files
    if all_parent_pairs:
        for item in os.listdir(PARENT_STORE_PATH):
            os.remove(os.path.join(PARENT_STORE_PATH, item))
        
        for parent_id, doc in all_parent_pairs:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            file_path = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
            
# Run indexing
index_documents()
```

---

### Step 4: Define Agent Tools

Create the retrieval tools the agent will use.

```python
import json
from typing import List
from langchain_core.tools import tool

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
        file_path = os.path.join(PARENT_STORE_PATH, parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json")
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

# Bind tools to LLM
llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])
```

---

### Step 5: Define State and Data Models

Create the state structure for conversation tracking.

```python
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List

class State(MessagesState):
    """Extended state with conversation tracking"""
    questionIsClear: bool
    conversation_summary: str = ""

class QueryAnalysis(BaseModel):
    """Structured output for query analysis"""
    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable")
    questions: List[str] = Field(description="List of rewritten, self-contained questions")
    clarification_needed: str = Field(description="Explanation if the question is unclear")
```

---

### Step 6: Define Agent System Prompt

Create the prompt that guides the agent's reasoning.

```python
from langchain_core.messages import SystemMessage

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
```

---

### Step 7: Build Graph Node Functions

Create the processing nodes for the LangGraph workflow.

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from typing import Literal

def analyze_chat_and_summarize(state: State):
    """
    Analyzes chat history and summarizes key points for context.
    """
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    # Extract relevant messages (excluding current query and system messages)
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) 
        and not getattr(msg, "tool_calls", None)
    ]
    
    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    summary_prompt = """Summarize the key topics and context from this conversation concisely (2-3 sentences max). Discard irrelevant information, such as misunderstandings or off-topic queries/responses. If there are no key topics, return an empty string:\n\n"""
    for msg in relevant_msgs[-6:]:  # Last 6 messages for context
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        summary_prompt += f"{role}: {msg.content}\n"
    
    summary_prompt += "\nBrief Summary:"
    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=summary_prompt)])
    return {"conversation_summary": summary_response.content}

def analyze_and_rewrite_query(state: State):
    """
    Analyzes user query and rewrites it for clarity, using conversation context.
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"**Conversation Context:**\n{conversation_summary}" 
        if conversation_summary.strip() 
        else "**Conversation Context:**\n[First query in conversation]"
    )
    
    prompt = f"""Rewrite the user's query to be clear and optimized for information retrieval.

              **User Query:**
              `"{last_message.content}"`
              
              `{context_section}`
              
              **Instructions:**
              
              1. **If this is a follow-up** (uses pronouns, references previous topics): Resolve all references using the context to make it a self-contained query.
              2. **If it's a new independent query:** Ensure it's already clear and specific.
              3. **If it contains multiple distinct questions:** Split them into a list of separate, focused questions.
              4. **Use specific, keyword-rich language:** Remove vague terms and conversational filler.
              5. **Mark as unclear** if you cannot determine a clear user intent for information retrieval.
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
```

**Why this architecture?**
- **Summarization** maintains conversational context without overwhelming the LLM
- **Query rewriting** ensures search queries are precise and unambiguous
- **Human-in-the-loop** catches unclear queries before wasting retrieval resources
- **Routing logic** determines whether clarification is needed

---

### Step 8: Build the LangGraph Agent

Assemble the complete workflow graph with conversation memory.

```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# Initialize checkpointer for conversation memory
checkpointer = InMemorySaver()

# Create graph builder
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("summarize", analyze_chat_and_summarize)
graph_builder.add_node("analyze_rewrite", analyze_and_rewrite_query)
graph_builder.add_node("human_input", human_input_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))

# Define edges
graph_builder.add_edge(START, "summarize")
graph_builder.add_edge("summarize", "analyze_rewrite")
graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
graph_builder.add_edge("human_input", "analyze_rewrite")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

# Compile graph with checkpointer
agent_graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_input"]
)
```

**Graph flow:**
1. **START** → `summarize` (analyze conversation history)
2. `summarize` → `analyze_rewrite` (rewrite query with context)
3. `analyze_rewrite` → `agent` (if clear) OR `human_input` (if unclear)
4. `human_input` → `analyze_rewrite` (after user clarifies)
5. `agent` → `tools` (if needs retrieval) OR END (if answer ready)
6. `tools` → `agent` (return retrieved data)

---

### Step 9: Create Chat Interface

Build a Gradio interface with conversation persistence.

```python
import gradio as gr
import uuid

def create_thread_id():
    """Generate a unique thread ID for each conversation"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def clear_session():
    """Clear thread for new conversation"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat_with_agent(message, history):
    current_state = agent_graph.get_state(config)
    
    if current_state.next:
        # Resume interrupted conversation
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:
        # Start new query
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]},config)
    
    return result['messages'][-1].content

# Initialize thread configuration
config = create_thread_id()

# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        height=600,
        placeholder="<strong>Ask me anything!</strong><br><em>I'll search, reason, and act to give you the best answer :)</em>"
    )
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat_with_agent, type="messages", chatbot=chatbot)

demo.launch()
```

**You're done!** You now have a fully functional Agentic RAG system with conversation memory and query clarification.

---

## Installation & Usage

### Option 1: Quickstart Notebook (Recommended for Testing)

The easiest way to get started:

**Running in Google Colab:**
1. Click the **Open in Colab** badge at the top of this README
2. Create a `docs/` folder in the file browser
3. Upload your `.md` files to the `docs/` folder (or use sample files from the `sample_files/` folder)
4. Run all cells from top to bottom
5. The chat interface will appear at the end

**Running Locally (Jupyter/VSCode):**
1. Install dependencies first (available inside the project folder): `pip install -r requirements.txt`
2. Open the notebook in your preferred environment
3. Add your `.md` files to the `docs/` folder (sample files available in `sample_files/`)
4. Run all cells from top to bottom
5. The chat interface will appear at the end

### Option 2: Full Python Project (Recommended for Development)

#### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install packages
cd project
pip install -r requirements.txt
```

#### 2. Add Your Documents

Place your `.md` files into the `docs/` directory (sample files available in `sample_files/`).

#### 3. Index Your Documents

```bash
python indexing.py
```

This processes all files from `docs/` and populates the vector database.

#### 4. Run the Application

```bash
python app.py
```

#### 5. Ask Questions

Open the local URL (e.g., `http://127.0.0.1:7860`) to start chatting.

### Example Conversations

**With Conversation Memory:**
```
User: "How do I install SQL?"
Agent: [Provides installation steps from documentation]

User: "How do I update it?"
Agent: [Understands "it" = SQL, provides update instructions]
```

**With Query Clarification:**
```
User: "Tell me about that thing"
Agent: "I need more information. What specific topic are you asking about?"

User: "The installation process for PostgreSQL"
Agent: [Retrieves and answers with specific information]
```

### Converting PDFs to Markdown

Need to convert PDFs? Use this companion notebook:

📘 **[PDF to Markdown Converter](https://colab.research.google.com/gist/GiovanniPasq/a5f749f9f9f03f0ca90f8b480ec952ac/pdf_to_md.ipynb)**

---

## Upcoming Features

| Feature | Release | Description | Status |
|---------|---------|-------------|--------|
| 📄 **Enhanced PDF Notebook** | Released on 4 Nov 2025 | Additional guidance with library comparisons and useful repositories | ✅ Implemented |
| 🎯 **End-to-End Gradio Interface** | End of Nov 2025 | Fully automated pipeline | ⌛ In Progress |
| 🤖 **Multi-Agent Map-Reduce** | End of Dec 2025 | Parallel processing architecture | 🛠️ Planned |

---

## License

MIT License - Feel free to use this for learning and building your own projects!

---

## Contributing

Contributions are welcome! Open an issue or submit a pull request.
