import os
from openai import OpenAI
import pymupdf
import lancedb
from src.index.lancedb import DB_URI, TABLE_NAME
from loguru import logger

PROMPT_TEMPLATE = """\
The user has uploaded a pathology report. The following is context retrieved from a database:

{relevant_texts}

Based on this information, please answer the following question:
{query}

If the information provided is insufficient, respond with "I need more details to answer this question accurately.
"""

SYSTEM_PROMPT = """\
You are an AI assistant specializing in providing accurate and concise answers to questions related to oncology and clinical guidelines. You are given two inputs:

A context containing information retrieved from the NCCN Guidelines document.

A user question, which may reference a pathology report or general clinical information.

Your task is to use only the information provided in the context to answer the user's question. If the context does not contain relevant or sufficient information to address the question, respond with:
'I need more details or additional context to answer this question accurately.'

Guidelines for your response:

Use a professional and factual tone.

Clearly distinguish between information from the context and any external assumptions.

Avoid introducing information not present in the provided context.

Focus on aligning your response with the authoritative guidance of the NCCN Guidelines whenever possible.
"""


class ReportProcessor:
    def __init__(self):
        self.k = 2  # Number of top results to retrieve
        self.current_report_text = ""  # Stores text of the currently uploaded report
        self.table_name = TABLE_NAME
        self.db = lancedb.connect(DB_URI)  # Connect to LanceDB
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def upload_report(self, report_file: str) -> str:
        """
        Process the uploaded PDF report, extracting its content and storing it as a string.
        """
        with pymupdf.open(report_file) as report:
            # Collect all pages' text in bytes and separate with b'\f' (form feed in bytes)
            text_content = b"\f".join(
                page.get_text().encode("utf-8") for page in report.pages()
            )
        # Decode the concatenated bytes into a single string
        self.current_report_text = text_content.decode("utf-8")
        logger.info("Uploaded report content processed successfully.")
        return "Report uploaded and processed successfully."

    def query_llm(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Query the OpenAI GPT model with a constructed prompt.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return "There was an error processing your query."

    def ask_question(self, query: str) -> str:
        """
        Process a user query about the uploaded report, searching LanceDB for context
        and querying the OpenAI GPT model for an answer.
        """
        if not self.current_report_text:
            return "Please upload a report first."

        # Search LanceDB for relevant documents
        try:
            table = self.db.open_table(self.table_name)
            search_result = table.search(query).limit(self.k).to_polars()
            logger.info("LanceDB search result:\n" + str(search_result))
        except Exception as e:
            logger.error(f"Error searching LanceDB: {e}")
            return "There was an error retrieving information from the database."

        # Concatenate relevant texts as context
        try:
            relevant_texts = "\n".join(search_result["text"])  # Simplified mock-up
        except KeyError:
            logger.error("Search result does not contain expected 'text' column.")
            return "No relevant context could be found in the database."

        # Construct the prompt
        prompt = PROMPT_TEMPLATE.format(relevant_texts=relevant_texts, query=query)

        # Query the LLM
        llm_response = self.query_llm(prompt)
        logger.info(f"LLM response: {llm_response}")
        return llm_response
