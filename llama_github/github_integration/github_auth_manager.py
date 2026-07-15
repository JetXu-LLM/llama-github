from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional

import requests
from github import Auth, Github, GithubIntegration
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException
from urllib3.util.retry import Retry

from llama_github.logger import logger


class RetrievalOutcome(str, Enum):
    """Truthful outcome for a bounded GitHub retrieval operation."""

    OK = "ok"
    NO_HIT = "no_hit"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True)
class RetrievalResult:
    """Items plus enough fetch metadata to distinguish absence from failure."""

    items: list = field(default_factory=list)
    outcome: RetrievalOutcome = RetrievalOutcome.NO_HIT
    pages_fetched: int = 0
    truncated: bool = False
    status_code: Optional[int] = None
    error_type: Optional[str] = None

    def to_meta(self) -> dict:
        """Return a JSON-serializable summary without retrieved content."""
        return {
            "outcome": self.outcome.value,
            "item_count": len(self.items),
            "pages_fetched": self.pages_fetched,
            "truncated": self.truncated,
            "status_code": self.status_code,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class _JSONResponse:
    data: Any = None
    headers: dict = field(default_factory=dict)
    status_code: Optional[int] = None
    error_type: Optional[str] = None


@dataclass(frozen=True)
class _BoundedBytesResponse:
    """Private transport result for a size-capped raw GitHub response."""

    data: Optional[bytes] = None
    bytes_read: int = 0
    declared_size_bytes: Optional[int] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    oversize: bool = False


class GitHubAuthManager:
    """Manage personal-token and GitHub-App authentication for `ExtendedGithub` clients."""

    def __init__(self):
        self.github_instance: Optional[ExtendedGithub] = None
        self.access_token: Optional[str] = None
        self.app_id: Optional[int] = None
        self.private_key: Optional[str] = None
        self.installation_id: Optional[int] = None
        self._installation_token_expires_at: Optional[datetime] = None

    def authenticate_with_token(self, access_token: str):
        """Authenticate with a personal access token or OAuth token."""
        self.access_token = access_token
        self.github_instance = ExtendedGithub(access_token=access_token)
        return self.github_instance

    def _refresh_installation_token(self, force: bool = False) -> Optional[str]:
        """
        Refresh the GitHub App installation token when it is missing or close to expiry.
        """
        if not all([self.app_id, self.private_key, self.installation_id]):
            return self.access_token

        if (
            not force
            and self.access_token
            and self._installation_token_expires_at is not None
            and datetime.now(timezone.utc)
            < self._installation_token_expires_at - timedelta(minutes=1)
        ):
            return self.access_token

        app_auth = Auth.AppAuth(self.app_id, self.private_key)
        integration = GithubIntegration(auth=app_auth)
        authorization = integration.get_access_token(self.installation_id)
        self.access_token = authorization.token
        self._installation_token_expires_at = authorization.expires_at

        if self.github_instance is not None:
            self.github_instance.refresh_token(self.access_token)

        return self.access_token

    def authenticate_with_app(self, app_id: int, private_key: str, installation_id: int):
        """Authenticate using a GitHub App and keep a refreshable installation token."""
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self._refresh_installation_token(force=True)
        self.github_instance = ExtendedGithub(
            access_token=self.access_token,
            token_provider=self._refresh_installation_token,
        )
        return self.github_instance

    def refresh_app_auth_if_needed(self):
        """Public helper for refreshing the app token in long-running processes."""
        return self._refresh_installation_token(force=False)

    def close_connection(self):
        """Close the cached PyGithub client and drop the reference."""
        if self.github_instance is not None:
            close = getattr(self.github_instance, "close", None)
            if callable(close):
                close()
        self.github_instance = None


class ExtendedGithub(Github):
    """
    Thin PyGithub extension that adds token refresh and a few direct REST helpers.
    """

    def __init__(
        self,
        access_token: Optional[str],
        token_provider: Optional[Callable[[], Optional[str]]] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ):
        self._access_token = access_token or ""
        self._token_provider = token_provider
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        auth = Auth.Token(self._access_token) if self._access_token else None
        super().__init__(auth=auth)

    @property
    def access_token(self) -> str:
        """Expose the current token for tests and direct helper usage."""
        return self._access_token

    def refresh_token(self, access_token: Optional[str]) -> None:
        """Update the cached token and refresh the underlying requester auth object."""
        if not access_token:
            return
        self._access_token = access_token
        requester = getattr(self, "_Github__requester", None)
        if requester is not None:
            requester._Requester__auth = Auth.Token(access_token)

    def _ensure_fresh_token(self) -> None:
        """Ask the token provider for a fresh token before making direct REST calls."""
        if self._token_provider is None:
            return
        refreshed_token = self._token_provider()
        if refreshed_token and refreshed_token != self._access_token:
            self.refresh_token(refreshed_token)

    def _build_session(self) -> requests.Session:
        """Create a retry-enabled requests session for the direct REST helper methods."""
        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            other=0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        return session

    def _request_json_response(
        self,
        url: str,
        params: Optional[dict] = None,
        *,
        operation: str = "github_get",
    ) -> _JSONResponse:
        """Perform one bounded GET without logging URLs, queries, or response bodies."""
        self._ensure_fresh_token()
        session = self._build_session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        try:
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=(self._connect_timeout, self._read_timeout),
            )
            response.raise_for_status()
            return _JSONResponse(
                data=response.json(),
                headers=dict(response.headers),
                status_code=response.status_code,
            )
        except HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            logger.warning(
                "GitHub request failed operation=%s error_type=http_error status_code=%s",
                operation,
                status_code,
            )
            return _JSONResponse(
                status_code=status_code,
                error_type=f"http_{status_code}" if status_code else "http_error",
            )
        except RequestException as exc:
            logger.warning(
                "GitHub request failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return _JSONResponse(error_type=type(exc).__name__)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "GitHub response decode failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return _JSONResponse(error_type="invalid_json")
        finally:
            session.close()

    def _request_json(self, url: str, params: Optional[dict] = None):
        """Compatibility helper returning decoded JSON or ``None`` on failure."""
        return self._request_json_response(url, params).data

    def _request_bounded_bytes(
        self,
        url: str,
        *,
        max_bytes: int,
        operation: str = "github_raw_get",
    ) -> _BoundedBytesResponse:
        """Download at most ``max_bytes`` while retaining typed failure metadata."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._ensure_fresh_token()
        session = self._build_session()
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            response = session.get(
                url,
                headers=headers,
                timeout=(self._connect_timeout, self._read_timeout),
                stream=True,
            )
            response.raise_for_status()

            declared_size = None
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    parsed_length = int(content_length)
                    if parsed_length >= 0:
                        declared_size = parsed_length
                except (TypeError, ValueError):
                    declared_size = None
            if declared_size is not None and declared_size > max_bytes:
                return _BoundedBytesResponse(
                    declared_size_bytes=declared_size,
                    status_code=response.status_code,
                    oversize=True,
                )

            chunks = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    return _BoundedBytesResponse(
                        bytes_read=bytes_read,
                        declared_size_bytes=declared_size,
                        status_code=response.status_code,
                        oversize=True,
                    )
                chunks.append(chunk)
            return _BoundedBytesResponse(
                data=b"".join(chunks),
                bytes_read=bytes_read,
                declared_size_bytes=declared_size,
                status_code=response.status_code,
            )
        except HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            logger.warning(
                "GitHub request failed operation=%s error_type=http_error status_code=%s",
                operation,
                status_code,
            )
            return _BoundedBytesResponse(
                status_code=status_code,
                error_type=f"http_{status_code}" if status_code else "http_error",
            )
        except RequestException as exc:
            logger.warning(
                "GitHub request failed operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return _BoundedBytesResponse(error_type=type(exc).__name__)
        finally:
            session.close()

    def _request_paginated_list(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        item_key: Optional[str] = None,
        max_items: int,
        max_pages: int,
        operation: str,
        page_size_limit: int = 100,
    ) -> RetrievalResult:
        """Fetch a list with explicit item/page bounds and truthful partial status."""
        if max_items <= 0 or max_pages <= 0:
            raise ValueError("max_items and max_pages must be positive")

        items: list = []
        pages_fetched = 0
        last_status_code = None
        base_params = dict(params or {})

        for page in range(1, max_pages + 1):
            remaining = max_items - len(items)
            if remaining <= 0:
                return RetrievalResult(
                    items=items,
                    outcome=RetrievalOutcome.PARTIAL,
                    pages_fetched=pages_fetched,
                    truncated=True,
                    status_code=last_status_code,
                )

            page_size = min(100, page_size_limit, remaining)
            page_params = {**base_params, "per_page": page_size, "page": page}
            response = self._request_json_response(
                url,
                page_params,
                operation=operation,
            )
            last_status_code = response.status_code
            if response.error_type:
                return RetrievalResult(
                    items=items,
                    outcome=(
                        RetrievalOutcome.PARTIAL if items else RetrievalOutcome.ERROR
                    ),
                    pages_fetched=pages_fetched,
                    truncated=bool(items),
                    status_code=response.status_code,
                    error_type=response.error_type,
                )

            page_items = (
                response.data.get(item_key)
                if item_key and isinstance(response.data, dict)
                else response.data
            )
            if not isinstance(page_items, list):
                return RetrievalResult(
                    items=items,
                    outcome=(
                        RetrievalOutcome.PARTIAL if items else RetrievalOutcome.ERROR
                    ),
                    pages_fetched=pages_fetched,
                    truncated=bool(items),
                    status_code=response.status_code,
                    error_type="invalid_response_shape",
                )

            pages_fetched += 1
            items.extend(page_items[:remaining])
            link_header = response.headers.get("Link", "")
            has_next = 'rel="next"' in link_header

            if not page_items:
                break
            if not has_next and len(page_items) < page_size:
                break
            if not has_next and len(page_items) == page_size:
                # GitHub normally supplies Link when another page exists. Treat no Link as complete.
                break
        else:
            has_next = True

        truncated = has_next
        if truncated:
            outcome = RetrievalOutcome.PARTIAL
        elif items:
            outcome = RetrievalOutcome.OK
        else:
            outcome = RetrievalOutcome.NO_HIT
        return RetrievalResult(
            items=items[:max_items],
            outcome=outcome,
            pages_fetched=pages_fetched,
            truncated=truncated,
            status_code=last_status_code,
        )

    def get_repo(self, full_name_or_id, lazy=False):
        """Wrap `Github.get_repo` so token refresh also applies to PyGithub calls."""
        self._ensure_fresh_token()
        return super().get_repo(full_name_or_id, lazy=lazy)

    def search_repositories(self, query, sort=None, order=None):
        """Wrap repository search so refreshable credentials also apply to PyGithub calls."""
        self._ensure_fresh_token()
        kwargs = {"query": query}
        if sort is not None:
            kwargs["sort"] = sort
        if order is not None:
            kwargs["order"] = order
        return super().search_repositories(**kwargs)

    def get_repo_structure(self, repo_full_name, branch="main") -> Optional[dict]:
        """Retrieve the full repository tree and convert it into a nested dictionary."""
        owner, repo_name = repo_full_name.split("/")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1"
        tree_data = self._request_json(url)
        if not tree_data or "tree" not in tree_data:
            logger.error("Repository tree retrieval returned no usable tree")
            return None

        def list_to_tree(items):
            tree = {}
            for item in items:
                path_parts = item["path"].split("/")
                current_level = tree
                for part in path_parts[:-1]:
                    current_level = current_level.setdefault(part, {"children": {}})
                    current_level = current_level["children"]

                if item["type"] == "blob":
                    current_level[path_parts[-1]] = {
                        "path": item["path"],
                        "size": item.get("size", 0),
                    }
                else:
                    current_level.setdefault(path_parts[-1], {"children": {}})
            return tree

        return list_to_tree(tree_data["tree"])

    def search_code_with_status(
        self,
        query: str,
        per_page: int = 30,
        *,
        max_pages: int = 1,
    ) -> RetrievalResult:
        """Search code without conflating a no-hit response with fetch failure."""
        url = "https://api.github.com/search/code"
        return self._request_paginated_list(
            url,
            params={"q": query},
            item_key="items",
            max_items=per_page * max_pages,
            max_pages=max_pages,
            operation="search_code",
            page_size_limit=per_page,
        )

    def search_code(self, query: str, per_page: int = 30) -> list:
        """Compatibility API returning only code-search items."""
        return self.search_code_with_status(query, per_page=per_page).items

    def search_issues_with_status(
        self,
        query: str,
        per_page: int = 30,
        *,
        max_pages: int = 1,
    ) -> RetrievalResult:
        """Search issues without conflating a no-hit response with fetch failure."""
        url = "https://api.github.com/search/issues"
        return self._request_paginated_list(
            url,
            params={"q": query},
            item_key="items",
            max_items=per_page * max_pages,
            max_pages=max_pages,
            operation="search_issues",
            page_size_limit=per_page,
        )

    def search_issues(self, query: str, per_page: int = 30) -> list:
        """Compatibility API returning only issue-search items."""
        return self.search_issues_with_status(query, per_page=per_page).items

    def get_issue_comments_with_status(
        self,
        repo_full_name: str,
        issue_number: int,
        *,
        max_items: int = 200,
        max_pages: int = 2,
    ) -> RetrievalResult:
        url = f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments"
        return self._request_paginated_list(
            url,
            max_items=max_items,
            max_pages=max_pages,
            operation="get_issue_comments",
        )

    def get_issue_comments(self, repo_full_name: str, issue_number: int) -> list:
        """Fetch issue comments through the direct REST helper path."""
        return self.get_issue_comments_with_status(repo_full_name, issue_number).items

    def get_pr_files_with_status(
        self,
        repo_full_name: str,
        pr_number: int,
        *,
        max_items: int = 300,
        max_pages: int = 3,
    ) -> RetrievalResult:
        url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
        return self._request_paginated_list(
            url,
            max_items=max_items,
            max_pages=max_pages,
            operation="get_pr_files",
        )

    def get_pr_files(self, repo_full_name: str, pr_number: int) -> list:
        """Fetch pull-request changed files through the direct REST helper path."""
        return self.get_pr_files_with_status(repo_full_name, pr_number).items

    def get_pr_comments_with_status(
        self,
        repo_full_name: str,
        pr_number: int,
        *,
        max_items: int = 200,
        max_pages: int = 2,
    ) -> RetrievalResult:
        url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
        return self._request_paginated_list(
            url,
            max_items=max_items,
            max_pages=max_pages,
            operation="get_pr_comments",
        )

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list:
        """Fetch pull-request issue-thread comments through the direct REST helper path."""
        return self.get_pr_comments_with_status(repo_full_name, pr_number).items
