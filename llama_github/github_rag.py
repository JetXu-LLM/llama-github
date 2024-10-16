from llama_github.logger import logger
from llama_github.config.config import config
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from pprint import pformat

from llama_github.llm_integration.initial_load import LLMManager
from llama_github.rag_processing.rag_processor import RAGProcessor

from llama_github.github_integration.github_auth_manager import GitHubAuthManager
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository, RepositoryPool
from llama_github.utils import AsyncHTTPClient

import asyncio
from IPython import get_ipython
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote


@dataclass
class GitHubAppCredentials:
    app_id: int
    private_key: str
    installation_id: int


class GithubRAG:
    rag_processor: RAGProcessor = None
    simple_mode: bool = False

    def __init__(self,
                 github_access_token: Optional[str] = None,
                 github_app_credentials: Optional[GitHubAppCredentials] = None,
                 openai_api_key: Optional[str] = None,
                 mistral_api_key: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 jina_api_key: Optional[str] = None,
                 open_source_models_hg_dir: Optional[str] = None,
                 embedding_model: Optional[str] = config.get(
                     "default_embedding"),
                 rerank_model: Optional[str] = config.get("default_reranker"),
                 llm: Any = None,
                 simple_mode: bool = False,
                 **kwargs) -> None:
        """
        Initialize the GithubRAG with the provided credentials and configuration.

        This constructor sets up the necessary components for GitHub integration,
        RAG processing, and LLM capabilities. It handles authentication, initializes
        the repository pool, and sets up the LLM manager.

        Args:
            github_access_token (Optional[str]): GitHub access token for authentication.
            github_app_credentials (Optional[GitHubAppCredentials]): Credentials for GitHub App authentication.
            openai_api_key (Optional[str]): API key for OpenAI services (GPT-4-turbo will be used).
            mistral_api_key (Optional[str]): API key for Mistral AI services.
            huggingface_token (Optional[str]): Token for Hugging Face services (recommended).
            jina_api_key (Optional[str]): API key for Jina AI services (s.jina.ai API will be used).
            open_source_models_hg_dir (Optional[str]): Directory for open-source models from Hugging Face.
            embedding_model (Optional[str]): Name of the preferred embedding model from Hugging Face.
            rerank_model (Optional[str]): Name of the preferred rerank model from Hugging Face.
            llm (Any): Any LangChain LLM chat object to replace OpenAI or open-source models.
            simple_mode (bool): If True, skip embedding and rerank model initialization in LLMManager.
            **kwargs: Additional keyword arguments for repository pool configuration.

        Raises:
            Exception: If there's an error during initialization.
        """
        try:
            logger.info("Initializing GithubRAG...")
            logger.debug("Initializing Github Instance...")

            self.simple_mode = simple_mode

            self.auth_manager = GitHubAuthManager()
            if github_access_token:
                self.github_instance = self.auth_manager.authenticate_with_token(
                    github_access_token)
            elif github_app_credentials:
                self.github_instance = self.auth_manager.authenticate_with_app(
                    github_app_credentials.app_id, github_app_credentials.private_key, github_app_credentials.installation_id)
            else:
                logger.error("GitHub credentials not provided.")
            logger.debug("Github Instance Initialized.")

            logger.debug("Initializing Repository Pool...")
            param_mapping = {
                "repo_cleanup_interval": "cleanup_interval",
                "repo_max_idle_time": "max_idle_time"
            }
            repo_pool_kwargs = {
                param_mapping[k]: v for k, v in kwargs.items() if k in param_mapping}
            self.RepositoryPool = RepositoryPool(
                self.github_instance, **repo_pool_kwargs)
            self.github_api_handler = GitHubAPIHandler(self.github_instance)
            logger.debug("Repository Pool Initialized.")

            self.jina_api_key = jina_api_key

            logger.debug(
                "Initializing llm manager, embedding model & reranker model...")
            self.llm_manager = LLMManager(
                openai_api_key, mistral_api_key, huggingface_token, open_source_models_hg_dir, embedding_model, rerank_model, llm, simple_mode=self.simple_mode)
            logger.debug(
                "LLM Manager, Embedding model & Reranker model Initialized.")

            self.rag_processor = RAGProcessor(
                self.github_api_handler, self.llm_manager)
            logger.info("GithubRAG initialization completed.")
        except Exception as e:
            logger.error(f"Error during GithubRAG initialization: {str(e)}")
            raise

    async def async_retrieve_context(self, query, simple_mode: Optional[bool] = None) -> List[str]:
        """
        Asynchronously retrieve context based on the given query.

        This method orchestrates the context retrieval process, including Google search,
        code search, issue search, and repository search. It uses the RAG processor to
        analyze the query and retrieve the most relevant contexts.

        Args:
            query (str): The query to retrieve context for.
            simple_mode (Optional[bool]): If provided, overrides the instance's simple_mode setting.

        Returns:
            List[str]: A list of the most relevant context strings.

        Raises:
            Exception: If there's an error during context retrieval.
        """

        if simple_mode is None:
            simple_mode = self.simple_mode

        topn_contexts = []  # This will be the list of context strings
        try:
            logger.info("Retrieving context...")
            if simple_mode:
                # In simple mode, only a Google search will be conducted based on the user's question.
                # This model is not suitable for long questions (e.g., questions with more than 20 words).
                task_google_search = asyncio.create_task(
                    self.google_search_retrieval(query=query))
                await asyncio.gather(task_google_search)
                logger.debug(
                    f"Google search: {str(len(task_google_search.result()))}")
                context_list = self.rag_processor.arrange_context(
                    google_search_result=task_google_search.result())
                if len(context_list) > 0:
                    topn_contexts = await self.rag_processor.retrieve_topn_contexts(
                        context_list=context_list, query=query, top_n=config.get("top_n_contexts"))
            else:
                # Analyzing question and generating strategy
                analyze_strategy = asyncio.create_task(
                    self.rag_processor.analyze_question(query))
                # wait for generate analyze strategy
                await analyze_strategy
                analyzed_strategy = analyze_strategy.result()
                logger.debug(f"Analyze strategy: {analyzed_strategy}")

                # google search from GitHub
                tokens = self.llm_manager.get_tokenizer().encode(query)
                query_tokens = len(tokens)
                task_google_search = asyncio.create_task(
                    self.google_search_retrieval(query=query if query_tokens < 20 else analyzed_strategy[0]))
                # code search from GitHub
                task_code_search = asyncio.create_task(
                    self.code_search_retrieval(query=analyzed_strategy[0], draft_answer=analyzed_strategy[1]+"\n\n"+analyzed_strategy[2]))
                # issue search from GitHub
                task_issue_search = asyncio.create_task(
                    self.issue_search_retrieval(query=analyzed_strategy[0], draft_answer=analyzed_strategy[1]+"\n\n"+analyzed_strategy[3]))
                # repo search from GitHub
                task_repo_search = asyncio.create_task(
                    self.repo_search_retrieval(query=analyzed_strategy[0]))

                # wait for all tasks to complete
                await asyncio.gather(task_google_search, task_code_search, task_issue_search, task_repo_search)

                logger.debug(
                    f"Google search: {str(len(task_google_search.result()))}")
                logger.debug(
                    f"Code search: {str(len(task_code_search.result()))}")
                logger.debug(
                    f"Issue search: {str(len(task_issue_search.result()))}")
                logger.debug(
                    f"Repo search: {str(len(task_repo_search.result()))}")

                context_list = self.rag_processor.arrange_context(
                    code_search_result=task_code_search.result(),
                    issue_search_result=task_issue_search.result(),
                    repo_search_result=task_repo_search.result(),
                    google_search_result=task_google_search.result())

                if len(context_list) > 0:
                    topn_contexts = await self.rag_processor.retrieve_topn_contexts(
                        context_list=context_list, query=query, answer=analyzed_strategy[1], top_n=config.get("top_n_contexts"))

            logger.info("Context retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise e
        return topn_contexts

    def retrieve_context(self, query, simple_mode: Optional[bool] = None) -> List[str]:
        """
        Retrieve context from GitHub code, issue and repo search based on the input query.

        This method serves as a wrapper for the async_retrieve_context method,
        handling the asynchronous call in different runtime environments (e.g., Jupyter notebook,
        asyncio event loop).

        Args:
            query (str): The query or question to retrieve context for.
            simple_mode (Optional[bool]): If provided, overrides the instance's simple_mode setting.

        Returns:
            List[str]: A list of context strings retrieved from the specified GitHub repositories.
        """
        effective_simple_mode = self.simple_mode if simple_mode is None else simple_mode

        self.loop = asyncio.get_event_loop()
        ipython = get_ipython()
        if ipython and ipython.has_trait('kernel'):
            logger.debug("Running in Jupyter notebook, nest_asyncio applied.")
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.async_retrieve_context(query, simple_mode=effective_simple_mode))
        if self.loop.is_running():
            return asyncio.ensure_future(self.async_retrieve_context(query, simple_mode=effective_simple_mode))
        return self.loop.run_until_complete(self.async_retrieve_context(query, simple_mode=effective_simple_mode))

    async def code_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Perform a code search on GitHub based on the given query and draft answer.

        This method uses the RAG processor to generate search criteria, then performs
        a code search using the GitHub API. It filters and deduplicates the results
        based on star count and relevance.

        Args:
            query (str): The main query for the code search.
            draft_answer (Optional[str]): A draft answer to refine the search criteria.

        Returns:
            list: A list of unique, relevant code search results.

        Raises:
            Exception: If there's an error during the code search retrieval.
        """
        
        result = []
        try:
            logger.info("Retrieving code search...")
            search_criterias = await self.rag_processor.get_code_search_criteria(query, draft_answer)
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_code(
                    search_criteria.replace('"', ''))
                for d in single_search_result:
                    result.append(d)
            # deduplicate results
            seen = set()
            unique_list = []
            for d in result:
                value = d["url"]
                if value not in seen and d["stargazers_count"] + config.get("code_search_max_hits") - d["index"] >= config.get("min_stars_to_keep_result"):
                    seen.add(value)
                    unique_list.append(d)
            result = unique_list

            logger.info("Code search retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving code search: {e}")

        return result

    async def issue_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Perform an issue search on GitHub based on the given query and draft answer.

        This method uses the RAG processor to generate search criteria, then performs
        an issue search using the GitHub API. It deduplicates the results and transforms
        the API URLs to official GitHub issue webpage URLs.

        Args:
            query (str): The main query for the issue search.
            draft_answer (Optional[str]): A draft answer to refine the search criteria.

        Returns:
            list: A list of unique, relevant issue search results with transformed URLs.

        Raises:
            Exception: If there's an error during the issue search retrieval.
        """

        result = []
        try:
            logger.info("Retrieving issue search...")
            search_criterias = await self.rag_processor.get_issue_search_criteria(query, draft_answer)
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_issues(
                    search_criteria.replace('"', ''))
                for d in single_search_result:
                    result.append(d)
            # deduplicate results
            seen = set()
            unique_list = []
            for d in result:
                api_url = d["url"]
                if api_url not in seen:
                    seen.add(api_url)
                    # Transform the API URL to the official GitHub issue webpage URL
                    html_url = api_url.replace(
                        'api.github.com/repos', 'github.com').replace('issues/', 'issues/')
                    d["url"] = html_url
                    unique_list.append(d)
            result = unique_list

            logger.info("Issue search retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving issue search: {e}")

        return result

    async def google_search_retrieval(self, query):
        """
        Perform a Google search for GitHub-related content based on the given query.

        This method uses the Jina AI search API to perform a Google search limited to
        GitHub.com. It then retrieves the content of the resulting GitHub URLs using
        the GitHub API.

        Args:
            query (str): The query to search for on Google.

        Returns:
            list: A list of dictionaries containing the GitHub URL and its content.

        Raises:
            Exception: If there's an error during the Google search retrieval.
        """

        result = []
        try:
            logger.info("Retrieving google search...")
            encoded_query = quote("site:github.com "+query)
            url = f"https://s.jina.ai/{encoded_query}"
            if self.jina_api_key is not None and self.jina_api_key != "":
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.jina_api_key}"
                }
                retry_delay = 1
            else:
                headers = {
                    "Accept": "application/json"
                }
                retry_delay = 20

            response = await AsyncHTTPClient.request(url, headers=headers, retry_count=2, retry_delay=retry_delay)
            urls = []
            urls = [item["url"] for item in response["data"] if "url" in item]

            for github_url in urls:
                content = self.github_api_handler.get_github_url_content(
                    github_url)
                if content and content != "":
                    result.append({
                        'url': github_url,
                        'content': content
                    })
            logger.info(f"Google search retrieved successfully:{urls}")
        except Exception as e:
            logger.error(f"Error retrieving google search: {e}")
        return result

    def _get_repository_rag_info(self, repository: Repository):
        """
        Retrieve RAG-related information for a given repository.

        This helper method fetches the README content and a simple structure
        of the repository using the Repository object.

        Args:
            repository (Repository): The Repository object to get information from.

        Returns:
            tuple: A tuple containing the repository's README content and simple structure.
        """

        return repository.get_readme(), self.rag_processor.get_repo_simple_structure(repository)

    async def repo_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Perform a repository search on GitHub based on the given query and draft answer.

        This method uses the RAG processor to generate search criteria, then performs
        a repository search using the GitHub API. It retrieves README content and
        simple structure for each relevant repository concurrently.

        Args:
            query (str): The main query for the repository search.
            draft_answer (Optional[str]): A draft answer to refine the search criteria.

        Returns:
            list: A list of dictionaries containing repository information and content.

        Raises:
            Exception: If there's an error during the repository search retrieval.
        """

        result = []
        results_with_index = []
        try:
            logger.info("Retrieving repo search...")
            search_criterias = await self.rag_processor.get_repo_search_criteria(query, draft_answer)
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_repositories(
                    search_criteria.replace('"', ''))
                for d in single_search_result:
                    result.append(d)
            # deduplicate results
            seen = set()
            unique_list = []
            for d in result:
                value = d.full_name
                if value not in seen:
                    seen.add(value)
                    unique_list.append(d)
            repositories = unique_list

            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                # Concurrently fetch the file content for each code search result
                future_to_index = {executor.submit(
                    self._get_repository_rag_info, repository): index for index, repository in enumerate(repositories)}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    repository = repositories[index]
                    try:
                        repo_readme, repo_simple_structure = future.result()
                        if repo_readme is None or repository.description is None or repository.description == "" or repo_simple_structure == "{}":
                            continue
                        if repo_readme:
                            results_with_index.append({
                                'index': index,
                                'full_name': repository.full_name,
                                'url': repository.html_url,
                                'content': repo_readme,
                            })
                        # if repo_simple_structure:
                        #     results_with_index.append({
                        #         'index': index,
                        #         'full_name': repository.full_name,
                        #         'content': "The repository "+repository.full_name+" with description:" + repository.description+" has below repo simple structure:\n"+repo_simple_structure,
                        #     })
                    except Exception as e:
                        logger.error(
                            f"Error getting repository info: {e}")

            logger.info("Repo search retrieved successfully.")
        except Exception as e:
            logger.error(f"Error retrieving repos search: {e}")
        return results_with_index

    def answer_with_context(self, query: str, contexts: Optional[List[Dict[str, Any]]] = None, simple_mode=False) -> str:
        """
        Generate an answer based on the given query and optional contexts.

        This method serves as a wrapper for the async_answer_with_context method,
        handling the asynchronous call in different runtime environments (e.g., Jupyter notebook,
        asyncio event loop).

        Args:
            query (str): The user's query.
            contexts (Optional[List[Dict[str, Any]]]): Optional list of context dictionaries.
                Each dictionary should contain 'content' and 'url' keys.
            simple_mode (bool): Whether to use simple mode for context retrieval.

        Returns:
            str: The generated answer.
        """

        self.loop = asyncio.get_event_loop()
        ipython = get_ipython()
        if ipython and ipython.has_trait('kernel'):
            logger.debug("Running in Jupyter notebook, nest_asyncio applied.")
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.async_answer_with_context(query, contexts, simple_mode))
        if self.loop.is_running():
            return asyncio.ensure_future(self.async_answer_with_context(query, contexts, simple_mode))
        return self.loop.run_until_complete(self.async_answer_with_context(query, contexts, simple_mode))

    async def async_answer_with_context(self, query: str, contexts: Optional[List[Dict[str, Any]]] = None, simple_mode=False) -> str:
        """
        Asynchronously generate an answer based on the given query and optional contexts.

        This method retrieves contexts if not provided, extracts relevant information,
        and uses the RAG processor's LLM handler to generate an answer.

        Args:
            query (str): The user's query.
            contexts (Optional[List[Dict[str, Any]]]): Optional list of context dictionaries.
                Each dictionary should contain 'content' and 'url' keys.
            simple_mode (bool): Whether to use simple mode for context retrieval.

        Returns:
            str: The generated answer.
        """
        
        if contexts is None:
            contexts = await self.async_retrieve_context(query, simple_mode)
            logger.debug(f"Retrieved contexts: {contexts}")
        context_contents = [context['context'] for context in contexts]
        context_urls = [context['url'] for context in contexts]

        answer = await self.rag_processor.llm_handler.ainvoke(
            human_question=query,
            context=context_contents,
            # context_urls=context_urls
        )

        return answer
