import os

from loguru import logger
from openai import OpenAI

from .prompt_templates import BASE_SYSTEM_TEMPLATE, USER_TEMPLATE


class BaseChat:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set in the environment variables.")
        self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client

    def generate(self, report: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Process a user query about the uploaded report, searching LanceDB for context
        and querying the OpenAI GPT model for an answer.
        """
        # Query the LLM
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BASE_SYSTEM_TEMPLATE},
                {"role": "user", "content": USER_TEMPLATE.format(report=report)},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        llm_response = (response.choices[0].message.content or "").strip()

        return llm_response
