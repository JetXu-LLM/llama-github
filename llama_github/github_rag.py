from llama_github.logger import logger
from llama_github.config.config import config
from typing import Optional, Any
from dataclasses import dataclass

from llama_github.llm_integration.initial_load import LLMManager
from llama_github.llm_integration.llm_handler import LLMHandler
from llama_github.rag_processing.rag_processor import RAGProcessor

from llama_github.github_integration.github_auth_manager import GitHubAuthManager
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository, RepositoryPool

@dataclass
class GitHubAppCredentials:
    app_id: int
    private_key: str
    installation_id: int

class GitHubRAG:
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
        Initialize the GitHubRAG with the provided credentials and configuration.

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
        
        logger.info("Initializing GitHubRAG...")
        logger.info("Initializing Github Instance...")
        self.auth_manager = GitHubAuthManager()
        if github_access_token:
            self.github_instance = self.auth_manager.authenticate_with_token(github_access_token)
        elif github_app_credentials:
            self.github_instance = self.auth_manager.authenticate_with_app(github_app_credentials.app_id, github_app_credentials.private_key, github_app_credentials.installation_id)
        else:
            logger.error("GitHub credentials not provided.")
        logger.info("Github Instance Initialized.")

        logger.info("Initializing Repository Pool...")
        param_mapping = {
            "repo_cleanup_interval": "cleanup_interval",
            "repo_max_idle_time": "max_idle_time"
        }
        repo_pool_kwargs = {param_mapping[k]: v for k, v in kwargs.items() if k in param_mapping}
        self.RepositoryPool = RepositoryPool(self.github_instance, **repo_pool_kwargs)
        self.github_api_handler = GitHubAPIHandler(self.github_instance)
        logger.info("Repository Pool Initialized.")

        logger.info("Initializing llm manager, embedding model & reranker model...")
        self.llm_manager = LLMManager(openai_api_key, huggingface_token, open_source_models_hg_dir, embedding_model, rerank_model, llm)
        logger.info("LLM Manager, Embedding model & Reranker model Initialized.")