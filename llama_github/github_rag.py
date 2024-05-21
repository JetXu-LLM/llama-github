from llama_github.logger import logger
from llama_github.config.config import config
from typing import Optional, Any
from dataclasses import dataclass
import json

from llama_github.llm_integration.initial_load import LLMManager
from llama_github.llm_integration.llm_handler import LLMHandler
from llama_github.rag_processing.rag_processor import RAGProcessor

from llama_github.github_integration.github_auth_manager import GitHubAuthManager
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository, RepositoryPool

import asyncio
from IPython import get_ipython

@dataclass
class GitHubAppCredentials:
    app_id: int
    private_key: str
    installation_id: int

class GithubRAG:
    rag_processor: RAGProcessor = None

    def __init__(self, github_access_token: Optional[str] = None,
                 github_app_credentials: Optional[GitHubAppCredentials] = None,
                 openai_api_key: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 open_source_models_hg_dir: Optional[str] = None,
                 embedding_model: Optional[str] =config.get("default_embedding"),
                 rerank_model: Optional[str] =config.get("default_reranker"),
                 llm: Any = None,
                 **kwargs) -> None:
        """
        Initialize the GithubRAG with the provided credentials and configuration.

        Parameters:
        - github_access_token (Optional[str]): GitHub access token for authentication.
        - github_app_credentials (Optional[GitHubAppCredentials]): Credentials for GitHub App authentication.
        - openai_api_key (Optional[str]): API key for OpenAI services -- recommend to use, GPT-4o will be used.
        - huggingface_token (Optional[str]): Token for Hugging Face services -- recommend to fill.
        - open_source_models_hg_dir (Optional[str]): Name of open-source models from Hugging Face to replace OpenAI.
        - embedding_model (Optional[str]): Name of Embedding model from Hugging Face, if you have preferred embedding model to be used.
        - rerank_model (Optional[str]): Name of Rerank model from Hugging Face, if you have preferred rerank model to be used.
        - llm (Any): Any kind of LangChain llm chat object - to replace OpenAI or open-source models from Hugging Face.
        - **kwargs:
            :param repo_cleanup_interval (Optional[int]): How often to run repo cleanup in seconds within RepositoryPool.
            :param repo_max_idle_time (Optional[int]): Keep a repo in cache until max idle time if not used.

        Returns:
        - None
        """
        try:
            logger.info("Initializing GithubRAG...")
            logger.debug("Initializing Github Instance...")
            self.loop = asyncio.get_event_loop()

            self.auth_manager = GitHubAuthManager()
            if github_access_token:
                self.github_instance = self.auth_manager.authenticate_with_token(github_access_token)
            elif github_app_credentials:
                self.github_instance = self.auth_manager.authenticate_with_app(github_app_credentials.app_id, github_app_credentials.private_key, github_app_credentials.installation_id)
            else:
                logger.error("GitHub credentials not provided.")
            logger.debug("Github Instance Initialized.")

            logger.debug("Initializing Repository Pool...")
            param_mapping = {
                "repo_cleanup_interval": "cleanup_interval",
                "repo_max_idle_time": "max_idle_time"
            }
            repo_pool_kwargs = {param_mapping[k]: v for k, v in kwargs.items() if k in param_mapping}
            self.RepositoryPool = RepositoryPool(self.github_instance, **repo_pool_kwargs)
            self.github_api_handler = GitHubAPIHandler(self.github_instance)
            logger.debug("Repository Pool Initialized.")

            logger.debug("Initializing llm manager, embedding model & reranker model...")
            self.llm_manager = LLMManager(openai_api_key, huggingface_token, open_source_models_hg_dir, embedding_model, rerank_model, llm)
            logger.debug("LLM Manager, Embedding model & Reranker model Initialized.")

            self.rag_processor = RAGProcessor(self.github_api_handler, self.llm_manager)
        except Exception as e:
            logger.error(f"Error initializing GithubRAG: {e}")
            raise e

    async def async_retrieve_context(self, query):
        """
        Retrieve context from GitHub code, issue and repo search based on the input query.

        Args:
            query (str): The query or question to retrieve context for.

        Returns:
            List[str]: A list of context strings retrieved from the specified GitHub repositories.
        """
        # Implementation of the context retrieval process
        # This will involve using the GitHub API to search for relevant information,
        # augmenting the retrieved data through the RAG methodology, and
        # enhancing it with LLM capabilities.

        context_list = []  # This will be the list of context strings
        try:
            logger.info("Retrieving context...")
            # generate draft answer
            task_draft_answer = self.rag_processor.first_genenral_answer(query)
            # code search from GitHub
            task_code_search = self.code_search_retrieval(query)
            # issue search from GitHub
            task_issue_search = self.issue_search_retrieval(query)
            # repo search from GitHub
            task_repo_search = self.repo_search_retrieval(query)
            
            # wait for all tasks to complete
            await asyncio.gather(task_draft_answer, task_code_search, task_issue_search, task_repo_search)

            logger.info("Context retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise e
        return context_list
    
    def retrieve_context(self, query):
        ipython = get_ipython()
        if ipython and ipython.has_trait('kernel'):
            logger.debug("Running in Jupyter notebook, nest_asyncio applied.")
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.async_retrieve_context(query))
        
        if self.loop.is_running():
            return asyncio.ensure_future(self.async_retrieve_context(query))
        else:
            return asyncio.run(self.async_retrieve_context(query))
    
    async def code_search_retrieval(self, query):
        result = []
        try:
            logger.info("Retrieving code search...")
            search_criterias = await self.rag_processor.get_code_search_criteria(query)
            logger.debug(f"For {query}, the search_criterias for code search is: {search_criterias}")
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_code(search_criteria)
                for d in single_search_result:
                    result.append(d)
            #deduplicate results
            seen = set()
            unique_list = []
            for d in result:
                value = d["url"]
                if value not in seen:
                    seen.add(value)
                    unique_list.append(d)
            result = unique_list
            logger.debug(f"search results: {json.dumps(result, indent=4)}")
            logger.info("Code search retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving code search: {e}")
            raise e
        return result
    
    async def issue_search_retrieval(self, query):
        pass

    async def repo_search_retrieval(self, query):
        pass