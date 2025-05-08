import gradio as gr
import pymupdf
from dotenv import load_dotenv
from loguru import logger

from src.chat.rag_chat import RAGChat

load_dotenv()


def main():
    processor = RAGChat()
    current_report_text = ""

    def clear_current_report():
        nonlocal current_report_text
        current_report_text = ""
        return "Current report cleared.", None, None

    def upload_report_interface(pdf_file):
        if pdf_file is None:
            return "Please upload a file first."
        with pymupdf.open(pdf_file) as report:
            # Collect all pages' text in bytes and separate with b'\f' (form feed in bytes)
            text_content = b"\f".join(
                page.get_text().encode("utf-8") for page in report.pages()
            )
        # Decode the concatenated bytes into a single string
        nonlocal current_report_text
        current_report_text = text_content.decode("utf-8")
        logger.info("Uploaded report content processed successfully.")
        return "Report uploaded and processed successfully."

    def summarize_report():
        if not current_report_text:
            return "Please enter a question first."
        answer, _ = processor.generate(current_report_text)
        return answer

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("# Pathology Report RAG Summarization")

        with gr.Row():
            upload = gr.File(label="Upload Pathology Report", type="filepath")
            result = gr.Textbox(label="Answer", lines=10)

        with gr.Row():
            summarize_button = gr.Button("Summarize Report")
            clear_button = gr.Button("Clear Current Report")

        # Process document automatically when uploaded
        upload.change(fn=upload_report_interface, inputs=[upload], outputs=[result])

        # Process question when user presses Enter
        summarize_button.click(fn=summarize_report, outputs=[result])

        clear_button.click(fn=clear_current_report, outputs=[result, upload])

    demo.launch()


if __name__ == "__main__":
    main()
