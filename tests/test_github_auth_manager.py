from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from llama_github.github_integration.github_auth_manager import (
    ExtendedGithub,
    GitHubAuthManager,
)


class TestGitHubAuthManager:
    def setup_method(self):
        self.auth_manager = GitHubAuthManager()

    @patch("llama_github.github_integration.github_auth_manager.ExtendedGithub")
    def test_authenticate_with_token(self, mock_extended_github):
        token = "fake_token"
        instance = self.auth_manager.authenticate_with_token(token)

        assert self.auth_manager.access_token == token
        assert instance == mock_extended_github.return_value
        mock_extended_github.assert_called_with(access_token=token)

    @patch("llama_github.github_integration.github_auth_manager.GithubIntegration")
    @patch("llama_github.github_integration.github_auth_manager.ExtendedGithub")
    def test_authenticate_with_app(self, mock_extended_github, mock_integration):
        mock_authorization = MagicMock()
        mock_authorization.token = "app_token"
        mock_authorization.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_integration.return_value.get_access_token.return_value = mock_authorization

        instance = self.auth_manager.authenticate_with_app(1, "key", 123)

        assert self.auth_manager.access_token == "app_token"
        assert instance == mock_extended_github.return_value
        mock_extended_github.assert_called_once()

    @patch("llama_github.github_integration.github_auth_manager.GithubIntegration")
    def test_refresh_app_auth_if_needed(self, mock_integration):
        first = MagicMock()
        first.token = "token-1"
        first.expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        second = MagicMock()
        second.token = "token-2"
        second.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_integration.return_value.get_access_token.side_effect = [first, second]

        self.auth_manager.authenticate_with_app(1, "key", 123)
        token = self.auth_manager.refresh_app_auth_if_needed()

        assert token == "token-2"
        assert self.auth_manager.access_token == "token-2"


class TestExtendedGithub:
    def test_get_repo_structure(self):
        gh = ExtendedGithub("token")
        gh._request_json = MagicMock(
            return_value={
                "tree": [
                    {"path": "folder", "type": "tree"},
                    {"path": "folder/file.py", "type": "blob", "size": 100},
                ]
            }
        )

        structure = gh.get_repo_structure("owner/repo")

        assert "folder" in structure
        assert "file.py" in structure["folder"]["children"]
        assert structure["folder"]["children"]["file.py"]["size"] == 100

    def test_search_code_retry_logic(self):
        gh = ExtendedGithub("token")
        gh._request_json = MagicMock(return_value={"items": []})

        result = gh.search_code("query")

        assert isinstance(result, list)
        gh._request_json.assert_called_once()

    def test_refresh_token_updates_requester(self):
        gh = ExtendedGithub("token-1")
        gh.refresh_token("token-2")

        assert gh.access_token == "token-2"
