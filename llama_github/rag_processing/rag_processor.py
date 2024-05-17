# rag_processor.py
from llama_github.config.config import config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository
from llama_github.llm_integration.llm_handler import LLMManager, LLMHandler
from llama_github.logger import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import json

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
        prompt = config.get("always_answer_prompt")
        response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt)
        return response.content
    
    class _GitHubCodeSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub code search, each following GitHub's search syntax.",
            example=["NumPy Array language:python", "log4j LoggingUtil language:java"],
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
        prompt = config.get("code_search_criteria_prompt")
        response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._GitHubCodeSearchCriteria)
        return response.search_criteria
    
    class _GitHubRepoSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub repository search, each following GitHub's search syntax.",
            example=["NumPy Array language:python", "spring-boot log4j language:java"],
            min_items=1,
            max_items=4
        )

    async def get_repo_search_criteria(self, query: str):
        """
        generate Github search criteria based on user's question
        
        Args:
            query (str): user's initial question.
        
        Returns:
            str[]: the search criteria for Github code search.
        """
        prompt = config.get("repo_search_criteria_prompt")
        response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._GitHubRepoSearchCriteria)
        return response.search_criteria

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
    



