from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from llama_github.config.config import config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository, RepositoryPool
from llama_github.github_integration.github_auth_manager import GitHubAuthManager
from llama_github.llm_integration.initial_load import LLMManager
from llama_github.logger import logger
from llama_github.rag_processing.rag_processor import RAGProcessor
from llama_github.utils import AsyncHTTPClient


@dataclass
class GitHubAppCredentials:
    """Credentials required for GitHub App authentication."""

    app_id: int
    private_key: str
    installation_id: int


class GithubRAG:
    """
    Main high-level facade for GitHub retrieval and answer generation.

    The class wires together GitHub authentication, repository caching,
    retrieval orchestration, and LLM-backed ranking/answering utilities.
    """

    rag_processor: Optional[RAGProcessor] = None
    simple_mode: bool = False

    def __init__(
        self,
        github_access_token: Optional[str] = None,
        github_app_credentials: Optional[GitHubAppCredentials] = None,
        openai_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        open_source_models_hg_dir: Optional[str] = None,
        embedding_model: Optional[str] = config.get("default_embedding"),
        rerank_model: Optional[str] = config.get("default_reranker"),
        llm: Any = None,
        simple_mode: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize GithubRAG with GitHub credentials, LLM configuration, and cache settings.

        Args:
            github_access_token: Personal access token for GitHub API access.
            github_app_credentials: GitHub App credentials used to mint installation tokens.
            openai_api_key: Optional OpenAI API key for chat-model-backed retrieval.
            mistral_api_key: Optional Mistral API key for chat-model-backed retrieval.
            huggingface_token: Reserved for future provider support.
            jina_api_key: Optional Jina search API key for the Google-style GitHub search path.
            open_source_models_hg_dir: Reserved placeholder for future native open-source chat support.
            embedding_model: Embedding model identifier used in professional mode.
            rerank_model: Reranker model identifier used in professional mode.
            llm: Optional injected LangChain-compatible chat model.
            simple_mode: When True, skip heavyweight embedding/reranker loading.
            **kwargs: Optional repository-pool settings such as cleanup interval and max idle time.
        """
        try:
            logger.info("Initializing GithubRAG...")
            self.simple_mode = simple_mode
            self.github_instance = None
            self.auth_manager = GitHubAuthManager()

            if github_access_token:
                self.github_instance = self.auth_manager.authenticate_with_token(
                    github_access_token
                )
            elif github_app_credentials:
                self.github_instance = self.auth_manager.authenticate_with_app(
                    github_app_credentials.app_id,
                    github_app_credentials.private_key,
                    github_app_credentials.installation_id,
                )
            else:
                logger.debug(
                    "GitHub credentials not provided during initialization. "
                    "Methods that actively retrieve from GitHub will require credentials."
                )

            param_mapping = {
                "repo_cleanup_interval": "cleanup_interval",
                "repo_max_idle_time": "max_idle_time",
            }
            repo_pool_kwargs = {
                param_mapping[key]: value
                for key, value in kwargs.items()
                if key in param_mapping
            }
            self.RepositoryPool = RepositoryPool(self.github_instance, **repo_pool_kwargs)
            self.github_api_handler = GitHubAPIHandler(
                self.github_instance,
                pool=self.RepositoryPool,
            )
            self.jina_api_key = jina_api_key
            self.llm_manager = LLMManager(
                openai_api_key=openai_api_key,
                mistral_api_key=mistral_api_key,
                huggingface_token=huggingface_token,
                open_source_models_hg_dir=open_source_models_hg_dir,
                embedding_model=embedding_model,
                rerank_model=rerank_model,
                llm=llm,
                simple_mode=self.simple_mode,
            )
            self.rag_processor = RAGProcessor(
                self.github_api_handler,
                self.llm_manager,
            )
            logger.info("GithubRAG initialization completed.")
        except Exception as exc:
            logger.error("Error during GithubRAG initialization: %s", exc)
            raise

    def _run_async(self, coroutine):
        """
        Execute an async coroutine in plain Python, IPython, or already-running loops.

        Returns the final value in synchronous contexts, or an `asyncio.Task` when
        called from an already-running non-notebook loop to preserve compatibility.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        try:
            from IPython import get_ipython
        except ImportError:
            get_ipython = None

        ipython = get_ipython() if get_ipython else None
        if ipython and hasattr(ipython, "has_trait") and ipython.has_trait("kernel"):
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coroutine)

        return asyncio.create_task(coroutine)

    def _query_token_count(self, query: str) -> int:
        """Estimate token count using the configured tokenizer when available."""
        tokenizer = self.llm_manager.get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(query))
        return len(query.split())

    async def async_retrieve_context(
        self, query, simple_mode: Optional[bool] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieve ranked GitHub context blocks for a coding question.

        In `simple_mode`, the method uses the lighter Google/Jina search path plus
        deterministic fallback ranking. In professional mode, it first asks the LLM
        to analyze the question and then combines Google, code, issue, and repository
        retrieval before final ranking.
        """
        if simple_mode is None:
            simple_mode = self.simple_mode

        topn_contexts: List[Dict[str, str]] = []
        try:
            logger.info("Retrieving context...")
            if simple_mode:
                google_search_result = await self.google_search_retrieval(query=query)
                context_list = self.rag_processor.arrange_context(
                    google_search_result=google_search_result
                )
                if context_list:
                    topn_contexts = await self.rag_processor.retrieve_topn_contexts(
                        context_list=context_list,
                        query=query,
                        top_n=config.get("top_n_contexts"),
                    )
                return topn_contexts

            if self.llm_manager.get_llm() is None:
                raise RuntimeError(
                    "Professional mode requires a configured chat model or provider API key."
                )

            analyzed_strategy = await self.rag_processor.analyze_question(query)
            logger.debug("Analyze strategy: %s", analyzed_strategy)
            query_tokens = self._query_token_count(query)

            task_google_search = asyncio.create_task(
                self.google_search_retrieval(
                    query=query if query_tokens < 20 else analyzed_strategy[0]
                )
            )
            task_code_search = asyncio.create_task(
                self.code_search_retrieval(
                    query=analyzed_strategy[0],
                    draft_answer=analyzed_strategy[1] + "\n\n" + analyzed_strategy[2],
                )
            )
            task_issue_search = asyncio.create_task(
                self.issue_search_retrieval(
                    query=analyzed_strategy[0],
                    draft_answer=analyzed_strategy[1] + "\n\n" + analyzed_strategy[3],
                )
            )
            task_repo_search = asyncio.create_task(
                self.repo_search_retrieval(query=analyzed_strategy[0])
            )

            await asyncio.gather(
                task_google_search,
                task_code_search,
                task_issue_search,
                task_repo_search,
            )

            context_list = self.rag_processor.arrange_context(
                code_search_result=task_code_search.result(),
                issue_search_result=task_issue_search.result(),
                repo_search_result=task_repo_search.result(),
                google_search_result=task_google_search.result(),
            )
            if context_list:
                topn_contexts = await self.rag_processor.retrieve_topn_contexts(
                    context_list=context_list,
                    query=query,
                    answer=analyzed_strategy[1],
                    top_n=config.get("top_n_contexts"),
                )
            logger.info("Context retrieved successfully.")
        except Exception as exc:
            logger.error("Error retrieving context: %s", exc)
            raise
        return topn_contexts

    def retrieve_context(
        self, query, simple_mode: Optional[bool] = None
    ) -> List[Dict[str, str]]:
        """
        Synchronous wrapper around `async_retrieve_context`.

        Returns a list of dictionaries where each item contains at least `context`
        and `url`.
        """
        effective_simple_mode = self.simple_mode if simple_mode is None else simple_mode
        return self._run_async(
            self.async_retrieve_context(query, simple_mode=effective_simple_mode)
        )

    async def code_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Run GitHub code search using LLM-generated search criteria and rank-preserving deduplication.
        """
        result = []
        try:
            logger.info("Retrieving code search...")
            search_criterias = await self.rag_processor.get_code_search_criteria(
                query,
                draft_answer,
            )
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_code(
                    search_criteria.replace('"', "")
                )
                result.extend(single_search_result)

            seen = set()
            unique_list = []
            for item in result:
                value = item["url"]
                score = (
                    item["stargazers_count"]
                    + config.get("code_search_max_hits")
                    - item["index"]
                )
                if value not in seen and score >= config.get("min_stars_to_keep_result"):
                    seen.add(value)
                    unique_list.append(item)
            result = unique_list
            logger.info("Code search retrieved successfully.")
        except Exception as exc:
            logger.error("Error retrieving code search: %s", exc)
        return result

    async def issue_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Run GitHub issue search using LLM-generated search criteria and deduplicate by URL.
        """
        result = []
        try:
            logger.info("Retrieving issue search...")
            search_criterias = await self.rag_processor.get_issue_search_criteria(
                query,
                draft_answer,
            )
            for search_criteria in search_criterias:
                single_search_result = self.github_api_handler.search_issues(
                    search_criteria.replace('"', "")
                )
                result.extend(single_search_result)

            seen = set()
            unique_list = []
            for item in result:
                api_url = item["url"]
                if api_url not in seen:
                    seen.add(api_url)
                    item["url"] = api_url.replace(
                        "https://api.github.com/repos/",
                        "https://github.com/",
                    )
                    unique_list.append(item)
            result = unique_list
            logger.info("Issue search retrieved successfully.")
        except Exception as exc:
            logger.error("Error retrieving issue search: %s", exc)
        return result

    async def google_search_retrieval(self, query):
        """
        Query the Jina-backed GitHub web-search path and expand the returned URLs into content.
        """
        result = []
        try:
            logger.info("Retrieving google search...")
            encoded_query = quote("site:github.com " + query)
            url = f"https://s.jina.ai/{encoded_query}"
            if self.jina_api_key:
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.jina_api_key}",
                }
                retry_delay = 1
            else:
                headers = {"Accept": "application/json"}
                retry_delay = 20

            response = await AsyncHTTPClient.request(
                url,
                headers=headers,
                retry_count=2,
                retry_delay=retry_delay,
            )
            data = response.get("data", []) if isinstance(response, dict) else []
            urls = [item["url"] for item in data if "url" in item]

            for github_url in urls:
                content = self.github_api_handler.get_github_url_content(github_url)
                if content:
                    result.append({"url": github_url, "content": content})
            logger.info("Google search retrieved successfully: %s", urls)
        except Exception as exc:
            logger.error("Error retrieving google search: %s", exc)
        return result

    def _get_repository_rag_info(self, repository: Repository):
        """Fetch the lightweight repository artifacts used during repository retrieval."""
        return repository.get_readme(), self.rag_processor.get_repo_simple_structure(
            repository
        )

    async def repo_search_retrieval(self, query, draft_answer: Optional[str] = None):
        """
        Search repositories and collect README-driven repository context blocks.
        """
        result = []
        results_with_index = []
        try:
            logger.info("Retrieving repo search...")
            search_criterias = await self.rag_processor.get_repo_search_criteria(
                query,
                draft_answer,
            )
            for search_criteria in search_criterias:
                result.extend(
                    self.github_api_handler.search_repositories(
                        search_criteria.replace('"', "")
                    )
                )

            seen = set()
            repositories = []
            for repository in result:
                if repository.full_name not in seen:
                    seen.add(repository.full_name)
                    repositories.append(repository)

            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                future_to_index = {
                    executor.submit(self._get_repository_rag_info, repository): index
                    for index, repository in enumerate(repositories)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    repository = repositories[index]
                    try:
                        repo_readme, repo_simple_structure = future.result()
                        if (
                            repo_readme is None
                            or not repository.description
                            or repo_simple_structure == "{}"
                        ):
                            continue
                        results_with_index.append(
                            {
                                "index": index,
                                "full_name": repository.full_name,
                                "url": repository.html_url,
                                "content": repo_readme,
                            }
                        )
                    except Exception as exc:
                        logger.error("Error getting repository info: %s", exc)

            logger.info("Repo search retrieved successfully.")
        except Exception as exc:
            logger.error("Error retrieving repos search: %s", exc)
        return results_with_index

    def answer_with_context(
        self,
        query: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
        simple_mode=False,
    ) -> str:
        """Synchronous wrapper around `async_answer_with_context`."""
        return self._run_async(self.async_answer_with_context(query, contexts, simple_mode))

    async def async_answer_with_context(
        self,
        query: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
        simple_mode=False,
    ) -> str:
        """
        Generate an answer using injected contexts or newly retrieved contexts.

        Context items may use either `context` or the older `content` key.
        """
        if contexts is None:
            contexts = await self.async_retrieve_context(query, simple_mode)
            logger.debug("Retrieved contexts: %s", contexts)

        context_contents = []
        for context in contexts:
            if "context" in context:
                context_contents.append(context["context"])
            elif "content" in context:
                context_contents.append(context["content"])

        return await self.rag_processor.llm_handler.ainvoke(
            human_question=query,
            context=context_contents,
        )
