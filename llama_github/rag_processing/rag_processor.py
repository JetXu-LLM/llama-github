# rag_processor.py
from llama_github.config.config import config
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository
from llama_github.llm_integration.llm_handler import LLMManager, LLMHandler
from llama_github.logger import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import json
import math
from numpy.linalg import norm
import numpy as np
import asyncio


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
            description="The abstraction of user's question, only one sentence no more than 20 words.",
            example="How to create a NumPy array in Python?"
        )
        answer: str = Field(
            ...,
            description="The answer to the user's question, better with sample code.",
            example="You can use the `numpy.array` function to create a NumPy array in Python. The sample code is as follows:\n\n```python\nimport numpy as np\n\narray = np.array([1, 2, 3])\nprint(array)\n```"
        )
        code_search_logic: str = Field(
            ...,
            description="Simple logic analyze on how to search for Github code related to the user's question without detail search criteria nor keywords.",
        )
        issue_search_logic: str = Field(
            ...,
            description="Simple logic analyze on how to search for Github issues related to the user's question without detail search criteria nor keywords.",
        )

    async def analyze_question(self, query: str) -> List[str]:
        """
        analyze user's question and generate strategy for code search and issue search

        Args:
            query (str): user's initial question.

        Returns:
            str: the answer of question.
        """
        try:
            logger.debug(
                f"Analyzing question and generating strategy")
            prompt = config.get("always_answer_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, output_structure=self._LLMFirstGenenralAnswer)
            return [response.question, response.answer, response.code_search_logic, response.issue_search_logic]
        except Exception as e:
            logger.error(f"Error in analyzing question: {e}")
            return [response.question, ""]

    class _GitHubCodeSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub code search, each following GitHub's search syntax.",
            example=["NumPy Array language:python",
                     "log4j LoggingUtil language:java"],
            min_items=1,
            max_items=2
        )

    async def get_code_search_criteria(self, query: str, draft_answer: Optional[str] = None) -> List[str]:
        """
        generate Github search criteria based on user's question

        Args:
            query (str): user's initial question.

        Returns:
            str[]: the search criteria for Github code search.
        """
        try:
            logger.debug(
                f"Generating code search criteria for question: {query}")
            prompt = config.get("code_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, context=[draft_answer] if draft_answer is not None else None, output_structure=self._GitHubCodeSearchCriteria)
            logger.debug(
                f"For {query}, the search_criterias for code search is: {response.search_criteria}")
            return response.search_criteria
        except Exception as e:
            logger.error(f"Error in get_code_search_criteria: {e}")
            return []

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
            min_items=0,
            max_items=2
        )

    async def get_repo_search_criteria(self, query: str, draft_answer: Optional[str] = None) -> List[str]:
        """
        generate Github search criteria based on user's question

        Args:
            query (str): user's initial question.

        Returns:
            str[]: the search criteria for Github code search.
        """
        search_criteria = []
        try:
            logger.debug(
                f"Generating repo search criteria for question: {query}")
            prompt = config.get("repo_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, context=[draft_answer] if draft_answer is not None else None, output_structure=self._GitHubRepoSearchCriteria)
            if response.necessity_score >= 80:
                search_criteria = response.search_criteria
            elif response.necessity_score >= 60 and len(response.search_criteria) >= 1:
                search_criteria = response.search_criteria[:1]
            logger.debug(
                f"For {query}, the search_criterias for repo search is: {search_criteria} and repo search necessity score is {response.necessity_score}")
        except Exception as e:
            logger.error(f"Error in get_repo_search_criteria: {e}")
            return search_criteria
        return search_criteria

    def get_repo_simple_structure(self, repo: Repository) -> str:
        """
        get a simple structure of a repository, only contains first 3 levels of repo folder/file structure.

        Args:
            repo (Repository): the repository object.

        Returns:
            json: the simple structure of the repository.
        """
        full_structure = repo.get_structure()

        if not full_structure:
            return json.dumps({})

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

    class _GitHubIssueSearchCriteria(BaseModel):
        search_criteria: List[str] = Field(
            ...,
            description="A list of search criteria strings for GitHub issue search, each following GitHub's search syntax.",
            example=["is:open label:bug Numpy Array",
                     "is:closed label:documentation langchain ollama"],
            min_items=1,
            max_items=2
        )

    async def get_issue_search_criteria(self, query: str, draft_answer: Optional[str] = None) -> List[str]:
        """
        generate Github search criteria based on user's question

        Args:
            query (str): user's initial question.

        Returns:
            str[]: the search criteria for Github issue search.
        """
        try:
            logger.debug(
                f"Generating issue search criteria for question: {query}")
            prompt = config.get("issue_search_criteria_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, context=[draft_answer] if draft_answer is not None else None, output_structure=self._GitHubIssueSearchCriteria)
            logger.debug(
                f"For {query}, the search_criterias for issue search is: {response.search_criteria}")
            return response.search_criteria
        except Exception as e:
            logger.error(f"Error in get_issue_search_criteria: {e}")
            return []

    def _arrange_code_search_result(self, code_search_result: List[Dict]) -> List[Dict[str, str]]:
        """
        Arrange the result of Code search with metadata.

        Args:
            _arrange_code_search_result (dict): The result of Code search.

        Returns:
            List[Dict[str, str]]: The arranged result of Code search with metadata.
        """
        arranged_results = []

        for result in code_search_result:
            content = result['content']
            url = result.get('url', '')

            # Split content into chunks
            chunks = self._split_content_into_chunks(
                content, language=result['language'] if 'language' in result else None)

            for chunk in chunks:
                repository_full_name = result.get(
                    'repository_full_name', 'None')
                description = result.get('description', 'None')
                stargazers_count = result.get('stargazers_count', 'None')
                updated_at = result.get('updated_at', 'None')
                path = result.get('path', 'None')
                language = result.get('language', 'None')

                if updated_at != 'None':
                    updated_at = updated_at.strftime('%Y-%m-%d %H:%M:%S %Z')

                chunk_text = (
                    f"Sample code from repository: {repository_full_name}\n"
                    f"repository description: {description}\n"
                    f"repository stars: {stargazers_count}\n"
                    f"repository last updated: {updated_at}\n"
                    f"code path in repository: {path}\n"
                    f"programming language is: {language}\n\n"
                    f"{chunk}"
                )
                arranged_results.append({'context': chunk_text, 'url': url})

        return arranged_results

    def _split_content_into_chunks(self, content: str, language: Optional[str] = None, max_tokens: Optional[int] = config.get('chunk_size')) -> List[str]:
        """
        Split the content into chunks of maximum token length using LangChain's RecursiveCharacterTextSplitter.

        Args:
            content (str): The content to be split.
            language (Optional[str]): The programming language of the code. Defaults to None.

        Returns:
            list: A list of content chunks.
        """
        chunk_overlap = math.ceil(max_tokens * 0.15 / 100) * 100

        if language is None or language.lower() not in [e.value for e in Language] or language.lower() in ['markdown', 'html', 'c', 'perl']:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                separators=[
                    "\n\n",
                    "\n",
                    "\r\n",
                ],
                chunk_size=max_tokens,
                chunk_overlap=chunk_overlap,
                tokenizer=self.llm_manager.tokenizer
            )
        else:
            max_tokens = max_tokens * 5
            chunk_overlap = chunk_overlap * 5
            language_enum = Language[language.upper()]
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language_enum,
                chunk_size=max_tokens,
                chunk_overlap=chunk_overlap,
            )

        chunks = splitter.split_text(content)
        return chunks

    def _arrange_issue_search_result(self, issue_search_result: dict) -> List[Dict[str, str]]:
        """
        Arrange the result of Issue search with metadata.

        Args:
            _arrange_issue_search_result (dict): The result of Issue search.

        Returns:
            List[Dict[str, str]]: The arranged result of Issue search with metadata.
        """
        arranged_results = []

        for result in issue_search_result:
            content = result['issue_content']
            url = result.get('url', '')

            # Split content into chunks
            chunks = self._split_content_into_chunks(
                content, max_tokens=config.get('issue_chunk_size'))

            for chunk in chunks:
                arranged_results.append({'context': chunk, 'url': url})

        return arranged_results

    def _arrange_repo_search_result(self, repo_search_result: dict) -> List[Dict[str, str]]:
        """
        Arrange the result of Repo search with metadata.

        Args:
            _arrange_repo_search_result (dict): The result of Repo search.

        Returns:
            List[Dict[str, str]]: The arranged result of Repo search with metadata.
        """
        arranged_results = []

        for result in repo_search_result:
            content = result['content']
            url = result.get('url', '')

            # Split content into chunks
            chunks = self._split_content_into_chunks(
                content, max_tokens=config.get('repo_chunk_size'))

            for chunk in chunks:
                arranged_results.append({'context': chunk, 'url': url})

        return arranged_results

    def _arrange_google_search_result(self, google_search_result: dict) -> List[Dict[str, str]]:
        """
        Arrange the result of Google search with metadata.

        Args:
            google_search_result (dict): The result of Google search.

        Returns:
            List[Dict[str, str]]: The arranged result of Google search with metadata.
        """
        arranged_results = []

        for result in google_search_result:
            content = result['content']
            url = result.get('url', '')  # Extract the URL if available

            # Split content into chunks
            chunks = self._split_content_into_chunks(content, max_tokens=config.get('google_chunk_size'))

            for chunk in chunks:
                arranged_results.append({'context': chunk, 'url': url})

        return arranged_results

    def arrange_context(self, code_search_result: Optional[dict] = None, issue_search_result: Optional[dict] = None,
                        repo_search_result: Optional[dict] = None, google_search_result: Optional[dict] = None) -> List[
        Dict[str, str]]:
        """
        Arrange the context before RAG with metadata.

        Args:
            code_search_result (dict, optional): The result of code search. Defaults to None.
            issue_search_result (dict, optional): The result of issue search. Defaults to None.
            repo_search_result (dict, optional): The result of repo search. Defaults to None.
            google_search_result (dict, optional): The result of Google search. Defaults to None.

        Returns:
            List[Dict[str, str]]: The arranged context with metadata.
        """
        context = []
        if code_search_result:
            context.extend(self._arrange_code_search_result(code_search_result))
        if issue_search_result:
            context.extend(self._arrange_issue_search_result(issue_search_result))
        if repo_search_result:
            context.extend(self._arrange_repo_search_result(repo_search_result))
        if google_search_result:
            context.extend(self._arrange_google_search_result(google_search_result))
        return context

    async def retrieve_topn_contexts(self, context_list: List[Dict[str, str]], query: str, answer: Optional[str] = None,
                                     top_n: Optional[int] = 5) -> List[Dict[str, str]]:
        """
        Retrieve top n context dictionaries from the context list.

        Args:
            context_list (List[Dict[str, str]]): List of context dictionaries to retrieve top n from.
                Each dictionary should have at least 'context' and 'url' keys.
            query (str): The query string.
            answer (Optional[str]): The answer string (optional).
            top_n (Optional[int]): Number of top context strings to retrieve (default: 5).

        Returns:
            List[Dict[str, str]]: A list of top n context dictionaries.
        """
        top_contexts = []
        try:
            reranker = self.llm_manager.get_rerank_model()

            # Extract contexts from the dictionaries
            contexts = [context_item['context'] for context_item in context_list]

            # Create sentence pairs for reranking
            sentence_pairs = [[query, doc] for doc in contexts]
            rerank_scores = reranker.compute_score(sentence_pairs)

            # Zip scores with context dictionaries
            scored_contexts = list(zip(rerank_scores, context_list))
            sorted_scored_contexts = sorted(
                scored_contexts, key=lambda x: x[0], reverse=True)

            # Extract top 3*top_n context dictionaries after rerank
            selected_contexts = [context for score, context in
                                 sorted_scored_contexts[:min(top_n * 3, len(sorted_scored_contexts))]]

            # If there are too few contexts, skip embedding comparison step
            if len(selected_contexts) < top_n * 2:
                return selected_contexts[:min(top_n, len(selected_contexts))]

            # Calculate embeddings to select top 2*top_n
            logger.debug("Embedding start...")
            embedding_model = self.llm_manager.get_embedding_model()
            query_embedding = embedding_model.encode(query + "\n" + answer if answer is not None else "")
            context_embeddings = [embedding_model.encode(context_item['context']) for context_item in selected_contexts]

            # Calculate cosine similarities
            cos_similarities = [
                (query_embedding @ context_embedding.T) / (norm(query_embedding) * norm(context_embedding))
                for context_embedding in context_embeddings]

            # Get top indices based on cosine similarities
            top_indices = np.argsort(cos_similarities)[-(top_n * 2):][::-1]
            top_contexts = [selected_contexts[i] for i in top_indices]
            top_cos_similarities = [cos_similarities[i] for i in top_indices]
            top_rerank_scores = [rerank_scores[contexts.index(context_item['context'])] for context_item in
                                 top_contexts]

            # Use simple LLM to calculate context relevance scores
            llm_scores = await asyncio.gather(
                *[self.get_context_relevance_score(query, context_item['context']) for context_item in top_contexts])
            logger.debug(f"Simple LLM scores: {llm_scores}")

            # Combine scores for final ranking
            combined_scores = [llm_score * cos_sim * rerank_score
                               for llm_score, cos_sim, rerank_score in
                               zip(llm_scores, top_cos_similarities, top_rerank_scores)]

            combined_context_scores = list(zip(top_contexts, combined_scores))
            sorted_combined_context_scores = sorted(
                combined_context_scores, key=lambda x: x[1], reverse=True)
            logger.debug(f"Combined sorted context scores: {sorted_combined_context_scores}")

            # Extract top n context dictionaries
            top_contexts = [context for context, _ in sorted_combined_context_scores[:top_n]]
            logger.debug(f"Final top contexts: {top_contexts}")

        except Exception as e:
            logger.error(f"Error retrieving top n context: {e}")

        return top_contexts

    class _ContextRelevanceScore(BaseModel):
        score: int = Field(
            ...,
            description="This is a Context Relevance Score, ranging from 0 to 100, indicates how well a given coding-related context supports answering a specific question, with higher scores signifying greater relevance."
        )

    async def get_context_relevance_score(self, query: str, context: str) -> int:
        """
        generate context relevance score based on user's question and provided context

        Args:
            query (str): user's initial question.
            context (str): context fetched from Github

        Returns:
            int: context relevance score, from 0-100.
        """
        try:
            prompt = config.get("scoring_context_prompt")
            response = await self.llm_handler.ainvoke(human_question=query, prompt=prompt, context=[context], output_structure=self._ContextRelevanceScore, simple_llm=True)
            logger.debug(
                f"For {context[:20]}, the context relevance score is: {response.score}")
            return response.score
        except Exception as e:
            logger.error(f"Error in get_context_relevance_score: {e}")
            return -1
