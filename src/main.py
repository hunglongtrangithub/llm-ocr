import pymupdf
import lancedb
from src.index.lancedb import DB_URI, TABLE_NAME


class ReportProcessor:
    def __init__(self):
        self.k = 2
        self.current_report_text = ""
        self.db = lancedb.connect(DB_URI)
        self.table_name = TABLE_NAME

    def upload_report(self, report_file: str):
        with pymupdf.open(report_file) as report:
            # Collect all pages' text in bytes and separate with b'\f' (form feed in bytes)
            text_content = b"\f".join(
                page.get_text().encode("utf-8") for page in report.pages()
            )
        # Decode the concatenated bytes into a single string
        self.current_report_text = text_content.decode("utf-8")
        print(self.current_report_text)
        return "Report uploaded and processed successfully."

    def ask_question(self, query: str):
        if not self.current_report_text:
            return "Please upload a report first."

        # Search LanceDB for relevant documents
        table = self.db.open_table(self.table_name)
        search_result = table.search(query).limit(self.k).to_polars()
        print(search_result)

        # Use search results in a mock RAG process
        # For now, concatenate relevant text as context
        relevant_texts = "\n".join(search_result["text"])  # Simplified mock-up
        llm_response = f"Q: {query}\nContext: {relevant_texts}\nA: [LLM would generate an answer here]"
        return llm_response
