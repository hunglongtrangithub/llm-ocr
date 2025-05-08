import os

import pymupdf
from loguru import logger
from openai import OpenAI

import lancedb
from src.evaluate import evaluate
from src.index.lancedb import DB_URI, TABLE_NAME

PROMPT_TEMPLATE = """\
Reading the pathology report and summarizing your key findings from it. If the information provided is insufficient, respond with "I need more details to answer this question accurately".
From the following text, produce a concise bullet-point summary that covers:
• Patient information and clinical background
• What specimen was examined
• Main diagnostic findings
• Tumor measurements and grade
• Overall stage and lymph node evaluation\n\n
"{report}\n\n"
Summary:\n
"""


class baseProcessor:
    def __init__(self):
        # self.k = 10  # Number of top results to retrieve
        self.current_report_text = ""  # Stores text of the currently uploaded report
        # self.table_name = TABLE_NAME
        # self.db = lancedb.connect(DB_URI)  # Connect to LanceDB
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set in the environment variables.")
        self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client

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
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return "There was an error processing your query."

    def ask_question(self, user_prompt: str) -> str:
        """
        Process a user query about the uploaded report, searching LanceDB for context
        and querying the OpenAI GPT model for an answer.
        """
        if not self.current_report_text:
            return "Please upload a report first."

        # Construct the prompt
        prompt = PROMPT_TEMPLATE.format(
            report=self.current_report_text,
        )

        # Query the LLM
        llm_response = self.query_llm(prompt)
        logger.info(f"LLM response: {llm_response}")

        # Compute contextual relevancy
        cr = evaluate(
            query=user_prompt,
            actual_output=llm_response,
        )
        logger.info(f"Contextual Relevancy: {cr[0]}")
        logger.info(f"Contextual Relevancy Reason: {cr[1]}")
        return llm_response
