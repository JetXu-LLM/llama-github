from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel, Field

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from llama_github.config.config import config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository
from llama_github.llm_integration.initial_load import LLMManager
from llama_github.llm_integration.llm_handler import LLMHandler
from llama_github.logger import logger


class RAGProcessor:
    """
    Retrieval-side processor responsible for question analysis, chunking, and ranking.

    It sits between the raw GitHub retrieval layer and the high-level `GithubRAG`
    facade so the ranking logic stays reusable and testable.
    """

    def __init__(
        self,
        github_api_handler: GitHubAPIHandler,
        llm_manager: Optional[LLMManager] = None,
        llm_handler: Optional[LLMHandler] = None,
    ):
        """Create a processor backed by one GitHub API handler and one LLM stack."""
        self.llm_manager = llm_manager if llm_manager else LLMManager()
        self.llm_handler = (
            llm_handler if llm_handler else LLMHandler(llm_manager=self.llm_manager)
        )
        self.github_api_handler = github_api_handler

    class _LLMFirstGeneralAnswer(BaseModel):
        question: str = Field(
            ...,
            description="The abstraction of the user's question, only one sentence no more than 20 words.",
            examples=["How to create a NumPy array in Python?"],
        )
        answer: str = Field(
            ...,
            description="The answer to the user's question, preferably with sample code.",
        )
        code_search_logic: str = Field(
            ...,
            description="High-level logic for searching related GitHub code.",
        )
        issue_search_logic: str = Field(
            ...,
            description="High-level logic for searching related GitHub issues.",
        )

    async def analyze_question(self, query: str) -> List[str]:
        """
        Ask the chat model to summarize the question and produce retrieval guidance.

        Returns a four-item list:
        1. short normalized question
        2. draft answer
        3. code-search logic
        4. issue-search logic
        """
        try:
            logger.debug("Analyzing question and generating strategy")
            prompt = config.get("always_answer_prompt")
            response = await self.llm_handler.ainvoke(
                human_question=query,
                prompt=prompt,
                output_structure=self._LLMFirstGeneralAnswer,
            )
            return [
                response.question,
                response.answer,
                response.code_search_logic,
                response.issue_search_logic,
            ]
        except Exception as exc:
            logger.error("Question analysis failed error_type=%s", type(exc).__name__)
            return [query, "", "", ""]

    class _GitHubCodeSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of GitHub code search criteria strings.",
            min_length=1,
            max_length=2,
        )

    async def get_code_search_criteria(
        self, query: str, draft_answer: Optional[str] = None
    ) -> List[str]:
        """Generate GitHub code-search queries from the user question and optional draft answer."""
        try:
            logger.debug(
                "Generating code search criteria query_length=%s", len(query or "")
            )
            prompt = config.get("code_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(
                human_question=query,
                prompt=prompt,
                context=[draft_answer] if draft_answer is not None else None,
                output_structure=self._GitHubCodeSearchCriteria,
            )
            logger.debug(
                "Generated code search criteria count=%s",
                len(response.search_criteria),
            )
            return response.search_criteria
        except Exception as exc:
            logger.error(
                "Code search criteria generation failed error_type=%s",
                type(exc).__name__,
            )
            return []

    class _GitHubRepoSearchCriteria(BaseModel):
        necessity_score: int = Field(
            ...,
            description="0-100 necessity score for separate repository search.",
        )
        search_criteria: List[str] = Field(
            ...,
            description="A list of GitHub repository search criteria strings.",
            min_length=0,
            max_length=2,
        )

    async def get_repo_search_criteria(
        self, query: str, draft_answer: Optional[str] = None
    ) -> List[str]:
        """Generate GitHub repository-search queries and apply the returned necessity score."""
        search_criteria: List[str] = []
        try:
            logger.debug(
                "Generating repository search criteria query_length=%s",
                len(query or ""),
            )
            prompt = config.get("repo_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(
                human_question=query,
                prompt=prompt,
                context=[draft_answer] if draft_answer is not None else None,
                output_structure=self._GitHubRepoSearchCriteria,
            )
            if response.necessity_score >= 80:
                search_criteria = response.search_criteria
            elif response.necessity_score >= 60 and response.search_criteria:
                search_criteria = response.search_criteria[:1]
            logger.debug(
                "Generated repository search criteria count=%s necessity_score=%s",
                len(search_criteria),
                response.necessity_score,
            )
        except Exception as exc:
            logger.error(
                "Repository search criteria generation failed error_type=%s",
                type(exc).__name__,
            )
        return search_criteria

    def get_repo_simple_structure(self, repo: Repository) -> str:
        """Return a simplified JSON representation of the first three levels of a repository tree."""
        full_structure = repo.get_structure()

        if not full_structure:
            return json.dumps({})

        def simplify_tree(tree, level=1):
            if level > 3:
                return "..."

            simplified_tree = {}
            for key, value in tree.items():
                if "children" in value:
                    simplified_tree[key] = {
                        "children": simplify_tree(value["children"], level + 1)
                    }
                else:
                    simplified_tree[key] = value
            return simplified_tree

        return json.dumps(simplify_tree(full_structure), indent=4)

    class _GitHubIssueSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of GitHub issue search criteria strings.",
            min_length=1,
            max_length=2,
        )

    async def get_issue_search_criteria(
        self, query: str, draft_answer: Optional[str] = None
    ) -> List[str]:
        """Generate GitHub issue-search queries from the question and optional draft answer."""
        try:
            logger.debug(
                "Generating issue search criteria query_length=%s", len(query or "")
            )
            prompt = config.get("issue_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(
                human_question=query,
                prompt=prompt,
                context=[draft_answer] if draft_answer is not None else None,
                output_structure=self._GitHubIssueSearchCriteria,
            )
            logger.debug(
                "Generated issue search criteria count=%s",
                len(response.search_criteria),
            )
            return response.search_criteria
        except Exception as exc:
            logger.error(
                "Issue search criteria generation failed error_type=%s",
                type(exc).__name__,
            )
            return []

    def _arrange_code_search_result(
        self, code_search_result: List[Dict]
    ) -> List[Dict[str, str]]:
        """Convert raw code-search hits into context dictionaries enriched with repository metadata."""
        arranged_results = []

        for result in code_search_result:
            content = result.get("content")
            if not content:
                continue

            url = result.get("url", "")
            chunks = self._split_content_into_chunks(
                content,
                language=result.get("language"),
            )

            for chunk in chunks:
                updated_at = result.get("updated_at")
                if hasattr(updated_at, "strftime"):
                    updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S %Z")

                chunk_text = (
                    f"Sample code from repository: {result.get('repository_full_name', 'None')}\n"
                    f"repository description: {result.get('description', 'None')}\n"
                    f"repository stars: {result.get('stargazers_count', 'None')}\n"
                    f"repository last updated: {updated_at or 'None'}\n"
                    f"code path in repository: {result.get('path', 'None')}\n"
                    f"programming language is: {result.get('language', 'None')}\n\n"
                    f"{chunk}"
                )
                arranged_results.append({"context": chunk_text, "url": url})

        return arranged_results

    def _split_content_into_chunks(
        self,
        content: str,
        language: Optional[str] = None,
        max_tokens: Optional[int] = config.get("chunk_size"),
    ) -> List[str]:
        """
        Split long code or text into retrieval-sized chunks.

        Language-aware splitting is used when the language is supported and helpful;
        otherwise a generic newline-aware splitter is used. In simple mode the
        tokenizer is intentionally skipped and the generic splitter falls back to
        character-based chunk sizing.
        """
        chunk_overlap = math.ceil(max_tokens * 0.15 / 100) * 100

        supported_languages = {lang.value for lang in Language}
        if (
            language is None
            or language.lower() not in supported_languages
            or language.lower() in {"markdown", "html", "c", "perl"}
        ):
            tokenizer = self.llm_manager.get_tokenizer()
            if tokenizer is not None:
                splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    separators=["\n\n", "\n", "\r\n"],
                    chunk_size=max_tokens,
                    chunk_overlap=chunk_overlap,
                    tokenizer=tokenizer,
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", "\r\n"],
                    chunk_size=max_tokens,
                    chunk_overlap=chunk_overlap,
                )
        else:
            max_tokens = max_tokens * 5
            chunk_overlap = chunk_overlap * 5
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language[language.upper()],
                chunk_size=max_tokens,
                chunk_overlap=chunk_overlap,
            )

        return splitter.split_text(content)

    def _arrange_issue_search_result(
        self, issue_search_result: List[Dict]
    ) -> List[Dict[str, str]]:
        """Convert normalized issue-search results into chunked context dictionaries."""
        arranged_results = []
        for result in issue_search_result:
            content = result.get("issue_content")
            if not content:
                continue
            url = result.get("url", "")
            chunks = self._split_content_into_chunks(
                content,
                max_tokens=config.get("issue_chunk_size"),
            )
            for chunk in chunks:
                arranged_results.append({"context": chunk, "url": url})
        return arranged_results

    def _arrange_repo_search_result(
        self, repo_search_result: List[Dict]
    ) -> List[Dict[str, str]]:
        """Convert repository README results into chunked context dictionaries."""
        arranged_results = []
        for result in repo_search_result:
            content = result.get("content")
            if not content:
                continue
            url = result.get("url", "")
            chunks = self._split_content_into_chunks(
                content,
                max_tokens=config.get("repo_chunk_size"),
            )
            for chunk in chunks:
                arranged_results.append({"context": chunk, "url": url})
        return arranged_results

    def _arrange_google_search_result(
        self, google_search_result: List[Dict]
    ) -> List[Dict[str, str]]:
        """Convert expanded GitHub web-search results into chunked context dictionaries."""
        arranged_results = []
        for result in google_search_result:
            content = result.get("content")
            if not content:
                continue
            url = result.get("url", "")
            chunks = self._split_content_into_chunks(
                content,
                max_tokens=config.get("google_chunk_size"),
            )
            for chunk in chunks:
                arranged_results.append({"context": chunk, "url": url})
        return arranged_results

    def arrange_context(
        self,
        code_search_result: Optional[List[Dict]] = None,
        issue_search_result: Optional[List[Dict]] = None,
        repo_search_result: Optional[List[Dict]] = None,
        google_search_result: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Merge all enabled retrieval sources into one flat list of context dictionaries."""
        context: List[Dict[str, str]] = []
        if code_search_result:
            context.extend(self._arrange_code_search_result(code_search_result))
        if issue_search_result:
            context.extend(self._arrange_issue_search_result(issue_search_result))
        if repo_search_result:
            context.extend(self._arrange_repo_search_result(repo_search_result))
        if google_search_result:
            context.extend(self._arrange_google_search_result(google_search_result))
        return context

    def _fallback_rank_contexts(
        self, context_list: List[Dict[str, str]], query: str, top_n: int
    ) -> List[Dict[str, str]]:
        """Cheap deterministic fallback ranking used when rerankers or embeddings are unavailable."""
        query_terms = [term for term in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(term) > 2]

        def score(item_with_index):
            index, item = item_with_index
            text = item["context"].lower()
            overlap = sum(text.count(term) for term in query_terms)
            return overlap, -len(text), -index

        ranked = sorted(enumerate(context_list), key=score, reverse=True)
        return [item for _, item in ranked[: min(top_n, len(ranked))]]

    async def retrieve_topn_contexts(
        self,
        context_list: List[Dict[str, str]],
        query: str,
        answer: Optional[str] = None,
        top_n: Optional[int] = 5,
    ) -> List[Dict[str, str]]:
        """
        Rank and return the top-N contexts.

        Ranking strategy:
        - fallback lexical ranking when rerankers are unavailable
        - reranker-only ranking when embeddings are unavailable
        - reranker + embeddings + optional simple-LLM scoring when all components exist
        """
        top_n = top_n or 5
        try:
            reranker = self.llm_manager.get_rerank_model()
            if reranker is None or not hasattr(reranker, "compute_score"):
                return self._fallback_rank_contexts(context_list, query, top_n)

            contexts = [context_item["context"] for context_item in context_list]
            sentence_pairs = [[query, doc] for doc in contexts]
            rerank_scores = reranker.compute_score(sentence_pairs)
            scored_contexts = list(zip(rerank_scores, context_list))
            sorted_scored_contexts = sorted(
                scored_contexts,
                key=lambda item: item[0],
                reverse=True,
            )
            selected_contexts = [
                context
                for _, context in sorted_scored_contexts[
                    : min(top_n * 3, len(sorted_scored_contexts))
                ]
            ]

            embedding_model = self.llm_manager.get_embedding_model()
            if embedding_model is None or len(selected_contexts) < top_n * 2:
                return selected_contexts[: min(top_n, len(selected_contexts))]

            logger.debug("Embedding start...")
            query_text = query if answer is None else f"{query}\n{answer}"
            query_embedding = embedding_model.encode(query_text)
            context_embeddings = [
                embedding_model.encode(context_item["context"])
                for context_item in selected_contexts
            ]

            cos_similarities = [
                (query_embedding @ context_embedding.T)
                / (norm(query_embedding) * norm(context_embedding))
                for context_embedding in context_embeddings
            ]

            top_indices = np.argsort(cos_similarities)[-(top_n * 2) :][::-1]
            top_contexts = [selected_contexts[i] for i in top_indices]
            top_cos_similarities = [cos_similarities[i] for i in top_indices]
            top_rerank_scores = [
                rerank_scores[contexts.index(context_item["context"])]
                for context_item in top_contexts
            ]

            if self.llm_manager.get_llm_simple() is None:
                combined = list(
                    zip(
                        top_contexts,
                        [cos * rerank for cos, rerank in zip(top_cos_similarities, top_rerank_scores)],
                    )
                )
            else:
                llm_scores = await asyncio.gather(
                    *[
                        self.get_context_relevance_score(query, context_item["context"])
                        for context_item in top_contexts
                    ]
                )
                combined = list(
                    zip(
                        top_contexts,
                        [
                            llm_score * cos_sim * rerank_score
                            for llm_score, cos_sim, rerank_score in zip(
                                llm_scores,
                                top_cos_similarities,
                                top_rerank_scores,
                            )
                        ],
                    )
                )

            combined.sort(key=lambda item: item[1], reverse=True)
            return [context for context, _ in combined[:top_n]]
        except Exception as exc:
            logger.error("Context ranking failed error_type=%s", type(exc).__name__)
            return self._fallback_rank_contexts(context_list, query, top_n)

    class _ContextRelevanceScore(BaseModel):
        score: int = Field(
            ...,
            description="0-100 relevance score showing how well a context supports the question.",
        )

    async def get_context_relevance_score(self, query: str, context: str) -> int:
        """Ask the lightweight model to score how relevant a context block is to the question."""
        try:
            if self.llm_manager.get_llm_simple() is None:
                return 1
            prompt = config.get("scoring_context_prompt")
            response = await self.llm_handler.ainvoke(
                human_question=query,
                prompt=prompt,
                context=[context],
                output_structure=self._ContextRelevanceScore,
                simple_llm=True,
            )
            logger.debug(
                "Context relevance scoring completed context_length=%s score=%s",
                len(context or ""),
                response.score,
            )
            return response.score
        except Exception as exc:
            logger.error(
                "Context relevance scoring failed error_type=%s",
                type(exc).__name__,
            )
            return 1
