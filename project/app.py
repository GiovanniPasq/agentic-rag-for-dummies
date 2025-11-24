import gradio as gr
from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant...")
    demo.launch(css=custom_css)