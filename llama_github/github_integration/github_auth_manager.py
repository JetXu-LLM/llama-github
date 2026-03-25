from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import requests
from github import Auth, Github, GithubIntegration
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException
from urllib3.util.retry import Retry

from llama_github.logger import logger


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
        """Drop the cached client reference."""
        self.github_instance = None


class ExtendedGithub(Github):
    """
    Thin PyGithub extension that adds token refresh and a few direct REST helpers.
    """

    def __init__(
        self,
        access_token: Optional[str],
        token_provider: Optional[Callable[[], Optional[str]]] = None,
    ):
        self._access_token = access_token or ""
        self._token_provider = token_provider
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
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        return session

    def _request_json(self, url: str, params: Optional[dict] = None):
        """Perform an authenticated GET request and decode the JSON body."""
        self._ensure_fresh_token()
        session = self._build_session()
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self._access_token}",
        }
        try:
            response = session.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            logger.error("HTTP error occurred: %s", http_err)
        except RequestException as req_err:
            logger.error("Request error occurred: %s", req_err)
        except Exception as err:
            logger.error("An error occurred: %s", err)
        return None

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
            logger.error("Error fetching tree structure for %s", repo_full_name)
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

    def search_code(self, query: str, per_page: int = 30) -> list:
        """Use the GitHub REST code-search endpoint and return the raw `items` list."""
        url = "https://api.github.com/search/code"
        data = self._request_json(url, params={"q": query, "per_page": per_page})
        return data.get("items", []) if data else []

    def search_issues(self, query: str, per_page: int = 30) -> list:
        """Use the GitHub REST issue-search endpoint and return the raw `items` list."""
        url = "https://api.github.com/search/issues"
        data = self._request_json(url, params={"q": query, "per_page": per_page})
        return data.get("items", []) if data else []

    def get_issue_comments(self, repo_full_name: str, issue_number: int) -> list:
        """Fetch issue comments through the direct REST helper path."""
        url = f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments"
        data = self._request_json(url)
        return data if isinstance(data, list) else []

    def get_pr_files(self, repo_full_name: str, pr_number: int) -> list:
        """Fetch pull-request changed files through the direct REST helper path."""
        url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
        data = self._request_json(url)
        return data if isinstance(data, list) else []

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list:
        """Fetch pull-request issue-thread comments through the direct REST helper path."""
        url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
        data = self._request_json(url)
        return data if isinstance(data, list) else []
