# rag_processor.py

from llama_github.config.config import Config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.llm_integration.llm_handler import LLMManager, LLMHandler
from llama_github.logger import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class RAGProcessor:
    def __init__(self, github_api_handler: GitHubAPIHandler):
        self.llm_manager = LLMManager()
        self.llm_handler = LLMHandler(llm_manager=self.llm_manager)
        self.github_api_handler = github_api_handler

    async def first_genenral_answer(self, query: str):
        """
        generate an answer for user's question, no matter whether there is enough context. This answer will be used
        for RAG cosine distance caculation with context
        
        Args:
            query (str): user's initial question.
        
        Returns:
            str: the answer of question.
        """
        prompt = Config().get("always_answer_prompt")
        response = await self.llm_handler.invoke(human_question=query, prompt=prompt)
        return response
    
    class _GitHubCodeSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub code search, each following GitHub's search syntax.",
            example=["language:python NumPy Array", "language:python Pandas DataFrame"],
            min_items=1,
            max_items=4
        )

    async def get_code_search_criteria(self, query: str):
        """
        generate Github search criteria based on user's question
        
        Args:
            query (str): user's initial question.
        
        Returns:
            str[]: the search criteria for Github code search.
        """
        prompt = Config().get("code_search_criteria_prompt")
        response = await self.llm_handler.invoke(human_question=query, prompt=prompt, output_structure=self._GitHubCodeSearchCriteria)
        return response
