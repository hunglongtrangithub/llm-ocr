import gradio as gr
from src.main import ReportProcessor

processor = ReportProcessor()


def upload_report_interface(pdf_file):
    return processor.upload_report(pdf_file)


def ask_question_interface(query):
    return processor.ask_question(query)


def clear_current_report():
    processor.current_report_text = ""
    return "Current report cleared.", None, None


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Pathology Report RAG Summarization")
    with gr.Row():
        upload = gr.File(label="Upload Pathology Report", type="filepath")
        upload_button = gr.Button("Upload")
    with gr.Row():
        query = gr.Textbox(label="Ask a Question about the Report")
        query_button = gr.Button("Ask")
    with gr.Row():
        result = gr.Textbox(label="Answer", lines=10)
        clear_button = gr.Button("Clear Current Report")

    upload_button.click(upload_report_interface, inputs=[upload], outputs=[result])
    query_button.click(ask_question_interface, inputs=[query], outputs=[result])
    clear_button.click(clear_current_report, outputs=[result, upload, query])

if __name__ == "__main__":
    demo.launch()
    