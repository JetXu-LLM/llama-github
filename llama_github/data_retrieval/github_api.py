from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from github import GithubException

from llama_github.config.config import config
from llama_github.github_integration.github_auth_manager import (
    ExtendedGithub,
    RetrievalOutcome,
    RetrievalResult,
)
from llama_github.logger import logger

from .github_entities import RepositoryPool


class GitHubAPIHandler:
    """Higher-level retrieval wrapper built on top of `ExtendedGithub` and `RepositoryPool`."""

    def __init__(
        self,
        github_instance: Optional[ExtendedGithub],
        pool: Optional[RepositoryPool] = None,
    ):
        """Create a retrieval helper bound to one GitHub client and one repository pool."""
        self._github = github_instance
        self._owns_pool = pool is None
        self.pool = pool if pool is not None else RepositoryPool(github_instance)

    def close(self) -> None:
        """Close the internally-created pool; injected pools remain caller-owned."""
        if self._owns_pool:
            self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _require_github(self) -> ExtendedGithub:
        """Return the configured GitHub client or raise a clear runtime error."""
        if self._github is None:
            raise RuntimeError("GitHub credentials are required for repository retrieval.")
        return self._github

    def search_repositories(self, query, sort="best match", order="desc"):
        """Search repositories and normalize them into pooled `Repository` objects."""
        try:
            github = self._require_github()
            if sort not in ["stars", "forks", "updated"]:
                repositories = github.search_repositories(query=query, order=order)
            else:
                repositories = github.search_repositories(query=query, sort=sort, order=order)

            result = []
            for index, repo in enumerate(repositories):
                if index >= config.get("repo_search_max_hits"):
                    break
                result.append(
                    self.pool.get_repository(
                        repo.full_name,
                        github_instance=github,
                        id=repo.id,
                        name=repo.name,
                        description=repo.description,
                        html_url=repo.html_url,
                        stargazers_count=repo.stargazers_count,
                        language=repo.language,
                        default_branch=repo.default_branch,
                        updated_at=repo.updated_at,
                    )
                )
            return result
        except Exception as exc:
            logger.error(
                "Repository search failed query_length=%s error_type=%s",
                len(query or ""),
                type(exc).__name__,
            )
            return []

    def get_repository(self, full_repo_name, github_instance=None):
        """Return a pooled `Repository` instance by full repository name."""
        return self.pool.get_repository(full_repo_name, github_instance=github_instance)

    def _get_file_content_through_repository(self, code_search_result):
        """Resolve a single code-search hit into repository metadata plus file content."""
        repository_obj = self.get_repository(
            code_search_result["repository"]["full_name"],
            github_instance=self._github,
        )
        file_content = repository_obj.get_file_content(code_search_result["path"])
        return repository_obj, file_content

    def search_code_with_status(self, query, repo_full_name=None) -> RetrievalResult:
        """Search code and retain fetch status through content expansion."""
        try:
            github = self._require_github()
            logger.debug("Searching code query_length=%s", len(query or ""))
            if repo_full_name:
                query = f"{query} repo:{repo_full_name}"

            status_search = getattr(github, "search_code_with_status", None)
            raw_result = (
                status_search(
                    query=query,
                    per_page=config.get("code_search_max_hits"),
                )
                if callable(status_search)
                else None
            )
            if not isinstance(raw_result, RetrievalResult):
                legacy_items = github.search_code(
                    query=query,
                    per_page=config.get("code_search_max_hits"),
                )
                raw_result = RetrievalResult(
                    items=list(legacy_items or []),
                    outcome=(
                        RetrievalOutcome.OK
                        if legacy_items
                        else RetrievalOutcome.NO_HIT
                    ),
                    pages_fetched=1,
                )
            code_results = raw_result.items

            results_with_index = []
            content_failures = 0
            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                future_to_index = {
                    executor.submit(self._get_file_content_through_repository, code_result): index
                    for index, code_result in enumerate(code_results)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    code_result = code_results[index]
                    try:
                        repository_obj, file_content = future.result()
                        if repository_obj and file_content:
                            results_with_index.append(
                                {
                                    "index": index,
                                    "name": code_result["name"],
                                    "path": code_result["path"],
                                    "repository_full_name": code_result["repository"]["full_name"],
                                    "url": code_result["html_url"],
                                    "content": file_content,
                                    "stargazers_count": repository_obj.stargazers_count,
                                    "watchers_count": repository_obj.watchers_count,
                                    "language": repository_obj.language,
                                    "description": repository_obj.description,
                                    "updated_at": repository_obj.updated_at,
                                }
                            )
                    except Exception:
                        content_failures += 1
                        logger.error(
                            "Code-search content expansion failed result_index=%s",
                            index,
                        )

            items = sorted(results_with_index, key=lambda item: item["index"])
            missing_content = len(code_results) - len(items)
            incomplete = raw_result.outcome in {
                RetrievalOutcome.PARTIAL,
                RetrievalOutcome.ERROR,
            } or missing_content > 0
            if incomplete:
                outcome = (
                    RetrievalOutcome.ERROR
                    if raw_result.outcome is RetrievalOutcome.ERROR and not items
                    else RetrievalOutcome.PARTIAL
                )
            elif items:
                outcome = RetrievalOutcome.OK
            else:
                outcome = RetrievalOutcome.NO_HIT
            return RetrievalResult(
                items=items,
                outcome=outcome,
                pages_fetched=raw_result.pages_fetched,
                truncated=raw_result.truncated,
                status_code=raw_result.status_code,
                error_type=(
                    "content_fetch_incomplete"
                    if missing_content or content_failures
                    else raw_result.error_type
                ),
            )
        except Exception as exc:
            logger.error(
                "Code search failed query_length=%s error_type=%s",
                len(query or ""),
                type(exc).__name__,
            )
            return RetrievalResult(
                outcome=RetrievalOutcome.ERROR,
                error_type=type(exc).__name__,
            )

    def search_code(self, query, repo_full_name=None):
        """Compatibility API returning expanded code-search items only."""
        return self.search_code_with_status(query, repo_full_name).items

    def _get_issue_content_through_repository(self, issue):
        """Resolve an issue search hit into the rendered issue-content text block."""
        issue_url = issue["url"]
        match = re.search(r"https://api.github.com/repos/([^/]+/[^/]+)/issues/(\d+)", issue_url)
        if not match:
            logger.warning(
                "Failed to extract repository and issue number from issue URL",
            )
            return None

        repo_full_name = match.group(1)
        issue_number = int(match.group(2))
        repository_obj = self.get_repository(repo_full_name, github_instance=self._github)
        return repository_obj.get_issue_content(number=issue_number, issue=issue)

    def search_issues_with_status(self, query, repo_full_name=None) -> RetrievalResult:
        """Search issues and retain fetch status through content expansion."""
        try:
            github = self._require_github()
            logger.debug("Searching issues query_length=%s", len(query or ""))
            if repo_full_name:
                query = f"{query} repo:{repo_full_name}"

            status_search = getattr(github, "search_issues_with_status", None)
            raw_result = (
                status_search(
                    query=query,
                    per_page=config.get("issue_search_max_hits"),
                )
                if callable(status_search)
                else None
            )
            if not isinstance(raw_result, RetrievalResult):
                legacy_items = github.search_issues(
                    query=query,
                    per_page=config.get("issue_search_max_hits"),
                )
                raw_result = RetrievalResult(
                    items=list(legacy_items or []),
                    outcome=(
                        RetrievalOutcome.OK
                        if legacy_items
                        else RetrievalOutcome.NO_HIT
                    ),
                    pages_fetched=1,
                )
            raw_issue_count = len(raw_result.items)
            issue_results = raw_result.items
            issue_results = [
                issue
                for issue in issue_results
                if issue.get("body") not in {None, "null"}
            ]

            results_with_index = []
            content_failures = 0
            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                future_to_index = {
                    executor.submit(self._get_issue_content_through_repository, issue): index
                    for index, issue in enumerate(issue_results)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    issue_result = issue_results[index]
                    try:
                        issue_content = future.result()
                        if issue_content:
                            results_with_index.append(
                                {
                                    "index": index,
                                    "url": issue_result["url"],
                                    "created_at": issue_result["created_at"],
                                    "updated_at": issue_result["updated_at"],
                                    "issue_content": issue_content,
                                }
                            )
                    except Exception:
                        content_failures += 1
                        logger.error(
                            "Issue-search content expansion failed result_index=%s",
                            index,
                        )

            items = sorted(results_with_index, key=lambda item: item["index"])
            missing_content = raw_issue_count - len(items)
            incomplete = raw_result.outcome in {
                RetrievalOutcome.PARTIAL,
                RetrievalOutcome.ERROR,
            } or missing_content > 0
            if incomplete:
                outcome = (
                    RetrievalOutcome.ERROR
                    if raw_result.outcome is RetrievalOutcome.ERROR and not items
                    else RetrievalOutcome.PARTIAL
                )
            elif items:
                outcome = RetrievalOutcome.OK
            else:
                outcome = RetrievalOutcome.NO_HIT
            return RetrievalResult(
                items=items,
                outcome=outcome,
                pages_fetched=raw_result.pages_fetched,
                truncated=raw_result.truncated,
                status_code=raw_result.status_code,
                error_type=(
                    "content_fetch_incomplete"
                    if missing_content or content_failures
                    else raw_result.error_type
                ),
            )
        except Exception as exc:
            logger.error(
                "Issue search failed query_length=%s error_type=%s",
                len(query or ""),
                type(exc).__name__,
            )
            return RetrievalResult(
                outcome=RetrievalOutcome.ERROR,
                error_type=type(exc).__name__,
            )

    def search_issues(self, query, repo_full_name=None):
        """Compatibility API returning expanded issue-search items only."""
        return self.search_issues_with_status(query, repo_full_name).items

    @staticmethod
    def _categorize_github_url(url):
        """Classify a GitHub URL into repository, issue, file, README, or other."""
        repo_pattern = r"^https://github\.com/[^/]+/[^/]+$"
        issue_pattern = r"^https://github\.com/[^/]+/[^/]+/issues/\d+$"
        repo_file_pattern = r"^https://github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/.+$"
        readme_pattern = r"^https://github\.com/[^/]+/[^/]+#readme$"

        if re.match(repo_pattern, url):
            return "repo"
        if re.match(issue_pattern, url):
            return "issue"
        if re.match(repo_file_pattern, url):
            return "file"
        if re.match(readme_pattern, url):
            return "readme"
        return "other"

    def get_github_url_content(self, url):
        """Expand a GitHub repository / issue / file URL into text content for retrieval."""
        try:
            self._require_github()
            logger.debug("Retrieving content from categorized GitHub URL")
            content = None
            category = GitHubAPIHandler._categorize_github_url(url)
            if category == "repo":
                match = re.search(r"https://github\.com/([^/]+/[^/]+)", url)
                if match:
                    repository_obj = self.get_repository(match.group(1), github_instance=self._github)
                    content = repository_obj.get_readme()
            elif category == "issue":
                match = re.search(r"https://github\.com/([^/]+/[^/]+)/issues/(\d+)", url)
                if match:
                    repository_obj = self.get_repository(match.group(1), github_instance=self._github)
                    content = repository_obj.get_issue_content(number=int(match.group(2)))
            elif category == "file":
                match = re.search(
                    r"https://github\.com/([^/]+/[^/]+)/(?:blob|tree)/([^/]+)/(.+)",
                    url,
                )
                if match:
                    repository_obj = self.get_repository(match.group(1), github_instance=self._github)
                    content = repository_obj.get_file_content(match.group(3))
            elif category == "readme":
                match = re.search(r"https://github\.com/([^/]+/[^/]+)#readme", url)
                if match:
                    repository_obj = self.get_repository(match.group(1), github_instance=self._github)
                    content = repository_obj.get_readme()
            else:
                logger.warning("Unsupported GitHub URL category: %s", category)
            return content
        except (GithubException, RuntimeError) as exc:
            logger.error(
                "GitHub URL retrieval failed error_type=%s",
                type(exc).__name__,
            )
            return None
