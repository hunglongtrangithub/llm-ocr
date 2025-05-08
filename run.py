import gradio as gr
from dotenv import load_dotenv

from src.baseProcessor import baseProcessor
from src.ragProcessor import ragProcessor

load_dotenv()

processor = baseProcessor()


def upload_report_interface(pdf_file):
    if pdf_file is None:
        return "Please upload a file first."
    return processor.upload_report(pdf_file)


def ask_question_interface(question):
    return processor.ask_question(question)


def clear_current_report():
    processor.current_report_text = ""
    return "Current report cleared.", None, None


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Pathology Report RAG Summarization")

    with gr.Row():
        upload = gr.File(label="Upload Pathology Report", type="filepath")
        result = gr.Textbox(label="Answer", lines=10)

    with gr.Row():
        query = gr.Textbox(
            label="Ask a Question about the Report",
            placeholder="Type your question and press Enter...",
        )

    with gr.Row():
        clear_button = gr.Button("Clear Current Report")

    # Process document automatically when uploaded
    upload.change(fn=upload_report_interface, inputs=[upload], outputs=[result])

    # Process question when user presses Enter
    query.submit(fn=ask_question_interface, inputs=[query], outputs=[result])

    clear_button.click(fn=clear_current_report, outputs=[result, upload, query])

if __name__ == "__main__":
    demo.launch()

