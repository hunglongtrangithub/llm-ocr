import os
from openai import OpenAI
import pymupdf
import lancedb
from src.index.lancedb import DB_URI, TABLE_NAME
from src.evaluate import evaluate
from loguru import logger

PROMPT_TEMPLATE = """\
The user has uploaded this pathology report:

{report}

Below is context retrieved from the NCCN guideline database:

{relevant_texts}

Based on **both** the report _and_ the guideline context, please answer the question:

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


class ragProcessor:
    def __init__(self):
        self.k = 10  # Number of top results to retrieve
        self.current_report_text = ""  # Stores text of the currently uploaded report
        self.current_retrieval_context = ""  # Stores retrieved context from LanceDB
        self.table_name = TABLE_NAME
        self.db = lancedb.connect(DB_URI)  # Connect to LanceDB
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

    def ask_question(self, user_prompt: str) -> str:
        """
        Process a user query about the uploaded report, searching LanceDB for context
        and querying the OpenAI GPT model for an answer.
        """
        if not self.current_report_text:
            return "Please upload a report first."

        # Search LanceDB for relevant documents
        try:
            table = self.db.open_table(self.table_name)
            search_result = table.search(user_prompt).limit(self.k).to_polars()
            logger.info(
                "LanceDB search result:\n" + str(search_result["text"].to_list())
            )
            self.current_retrieval_context = search_result["text"].to_list()
        except Exception as e:
            logger.error(f"Error searching LanceDB: {e}")
            return "There was an error retrieving information from the database."

        # Concatenate relevant texts as context
        try:
            relevant_texts = "\n".join(self.current_retrieval_context)  # Simplified mock-up
        except KeyError:
            logger.error("Search result does not contain expected 'text' column.")
            return "No relevant context could be found in the database."

        # Construct the prompt
        prompt = PROMPT_TEMPLATE.format(
            report=self.current_report_text,
            relevant_texts=relevant_texts,
            query=user_prompt,
        )

        # Query the LLM
        llm_response = self.query_llm(prompt)
        logger.info(f"LLM response: {llm_response}")

        # Compute contextual relevancy
        cr = evaluate(
            query=user_prompt,
            actual_output=llm_response,
            retrieval_context=self.current_retrieval_context
        )
        logger.info(f"Contextual Relevancy: {cr[0]}")
        logger.info(f"Contextual Relevancy Reason: {cr[1]}")
        return llm_response
        
