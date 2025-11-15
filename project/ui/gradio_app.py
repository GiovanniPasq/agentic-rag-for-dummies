import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 No documents in knowledge base"
        return "\n".join([f"📄 {f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(f"✅ Added: {added} | Skipped: {skipped}")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(f"🗑️ Removed all documents")
        return format_file_list()
    
    def chat_handler(msg, hist):
        return chat_interface.chat(msg, hist)
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    custom_css = """
    .gradio-container { 
        max-width: 1000px !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    #doc-management-tab {
        max-width: 500px !important;
        margin: 0 auto !important;
    }
    """
    
    with gr.Blocks(title="RAG Assistant", css=custom_css, theme=gr.themes.Citrus()) as demo:
        
        with gr.Tab("📚 Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add Documents")
            gr.Markdown("Upload PDF files. Duplicates will be skipped automatically.")
            
            files_input = gr.File(
                label="Drop PDF files here",
                file_count="multiple",
                type="filepath",
                height=200
            )
            
            add_btn = gr.Button("➕ Add Documents", variant="primary")
            
            gr.Markdown("## Current Documents")
            
            file_list = gr.Textbox(
                label="Knowledge Base",
                value=format_file_list(),
                interactive=False,
                lines = 7,
                max_lines=10 
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
            
            add_btn.click(
                upload_handler, 
                [files_input], 
                [files_input, file_list], 
                show_progress="corner"
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                height=600, 
                placeholder="Ask me anything about your documents!"
            )
            chatbot.clear(clear_chat_handler)
            
            gr.ChatInterface(fn=chat_handler, type="messages", chatbot=chatbot)
    
    return demo