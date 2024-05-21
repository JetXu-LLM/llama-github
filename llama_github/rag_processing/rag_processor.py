# rag_processor.py
from llama_github.config.config import config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository
from llama_github.llm_integration.llm_handler import LLMManager, LLMHandler
from llama_github.logger import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import json
import random


class RAGProcessor:
    def __init__(self, github_api_handler: GitHubAPIHandler, llm_manager: LLMManager = None, llm_handler: LLMHandler = None):
        if llm_manager:
            self.llm_manager = llm_manager
        else:
            self.llm_manager = LLMManager()

        if llm_handler:
            self.llm_handler = llm_handler
        else:
            self.llm_handler = LLMHandler(llm_manager=self.llm_manager)
            
        self.github_api_handler = github_api_handler

    class _LLMFirstGenenralAnswer(BaseModel):
        question: str = Field(
            ...,
            description="The user's question, could be abstraction if the question is too long.",
            example="How to create a NumPy array in Python?"
        )
        answer: str = Field(
            ...,
            description="The answer to the user's question.",
            example="You can use the `numpy.array` function to create a NumPy array in Python. The sample code is as follows:\n\n```python\nimport numpy as np\n\narray = np.array([1, 2, 3])\nprint(array)\n```"
        )

    async def first_genenral_answer(self, query: str):
        """
        generate an answer for user's question, no matter whether there is enough context. This answer will be used
        for RAG cosine distance caculation with context

        Args:
            query (str): user's initial question.

        Returns:
            str: the answer of question.
        """
        logger.debug(f"Generating first general answer for question: {query}")
        prompt = config.get("always_answer_prompt")
        response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._LLMFirstGenenralAnswer)
        return response.question+"\n"+response.answer

    class _GitHubCodeSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub code search, each following GitHub's search syntax.",
            example=["NumPy Array language:python",
                     "log4j LoggingUtil language:java"],
            min_items=1,
            max_items=4
        )

    async def get_code_search_criteria(self, query: str) -> List[str]:
        """
        generate Github search criteria based on user's question

        Args:
            query (str): user's initial question.

        Returns:
            str[]: the search criteria for Github code search.
        """
        logger.debug(f"Generating code search criteria for question: {query}")
        prompt = config.get("code_search_criteria_prompt")
        response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._GitHubCodeSearchCriteria)
        logger.debug(f"{random.random()} For {query}, the search_criterias for code search is: {response.search_criteria}")
        return response.search_criteria

    class _GitHubRepoSearchCriteria(BaseModel):
        necessity_score: int = Field(
            ...,
            description="In case there is already Github Code search and issue search for question related context retrieve. How necessity do you think seperate repo search in Github will be required. 0-59:no necessity, 60-79:medium necessity, 80-100:high necessity",
            example=65
        )
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub repository search, each following GitHub's search syntax. The sorting of the list should be based on the necessity of the search criteria.",
            example=["NumPy Array language:python",
                     "spring-boot log4j language:java"],
            min_items=1,
            max_items=4
        )

    async def get_repo_search_criteria(self, query: str) -> List[str]:
        """
        generate Github search criteria based on user's question

        Args:
            query (str): user's initial question.

        Returns:
            str[]: the search criteria for Github code search.
        """
        search_criteria = []
        try:
            prompt = config.get("repo_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._GitHubRepoSearchCriteria)
            if response.necessity_score >= 80:
                search_criteria = response.search_criteria
            elif response.necessity_score >= 60:
                search_criteria = response.search_criteria[:2]
        except Exception as e:
            logger.error(f"Error in get_repo_search_criteria: {e}")
            return search_criteria
        return search_criteria

    def get_repo_simple_structure(self, repo: Repository):
        """
        get a simple structure of a repository, only contains first 3 levels of repo folder/file structure.

        Args:
            repo (Repository): the repository object.

        Returns:
            json: the simple structure of the repository.
        """
        full_structure = repo.get_structure()

        def simplify_tree(tree, level=1):
            """
            Simplify the tree structure to keep only three levels deep.
            """
            if level > 3:
                return '...'

            simplified_tree = {}
            for key, value in tree.items():
                if 'children' in value:
                    simplified_tree[key] = {
                        'children': simplify_tree(value['children'], level + 1)
                    }
                else:
                    simplified_tree[key] = value
            return simplified_tree

        simplified_structure = simplify_tree(full_structure)
        return json.dumps(simplified_structure, indent=4)