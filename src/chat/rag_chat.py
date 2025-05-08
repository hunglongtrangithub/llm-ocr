import os

from loguru import logger
from openai import OpenAI

import lancedb
from src.chat.prompt_templates import RAG_SYSTEM_TEMPLATE, USER_TEMPLATE
from src.index.index_lancedb import DB_URI, TABLE_NAME


class RAGChat:
    def __init__(self, k: int = 10):
        self.k = k
        db = lancedb.connect(DB_URI)
        self.table = db.open_table(TABLE_NAME)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set in the environment variables.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, report: str, model: str = "gpt-3.5-turbo") -> tuple[str, list[str]]:
        """
        Process a user query about the report, searching LanceDB for context
        and querying the OpenAI GPT model for an summarization.
        """

        # Search LanceDB for relevant documents
        try:
            search_result = self.table.search(report).limit(self.k).to_polars()
            retrieved_texts = search_result["text"].to_list()
        except Exception as e:
            logger.error(f"Error searching LanceDB: {e}")
            return "There was an error retrieving information from the database.", []
        
        # Format chunks with doc tags and IDs
        formatted_texts = [
            f"<doc id='{i}'>\n{text}\n</doc>"
            for i, text in enumerate(retrieved_texts)
        ]
        retrieval_context = "\n".join(formatted_texts)

        # Query the LLM
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_TEMPLATE.format(context=retrieval_context)},
                {"role": "user", "content": USER_TEMPLATE.format(report=report)},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        llm_response = (response.choices[0].message.content or "").strip()

        return llm_response, retrieved_texts
