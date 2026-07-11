from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from llama_github.github_integration.github_auth_manager import (
    ExtendedGithub,
    GitHubAuthManager,
    RetrievalOutcome,
    _JSONResponse,
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
        gh._request_json_response = MagicMock(
            return_value=_JSONResponse(
                data={"items": []},
                headers={},
                status_code=200,
            )
        )

        result = gh.search_code("query")

        assert isinstance(result, list)
        gh._request_json_response.assert_called_once()

    def test_search_status_distinguishes_no_hit_from_error(self):
        gh = ExtendedGithub("token")
        gh._request_json_response = MagicMock(
            side_effect=[
                _JSONResponse(data={"items": []}, status_code=200),
                _JSONResponse(status_code=503, error_type="http_503"),
            ]
        )

        no_hit = gh.search_code_with_status("query")
        failed = gh.search_code_with_status("query")

        assert no_hit.outcome is RetrievalOutcome.NO_HIT
        assert failed.outcome is RetrievalOutcome.ERROR
        assert failed.error_type == "http_503"

    def test_bounded_pagination_reports_partial_result(self):
        gh = ExtendedGithub("token")
        gh._request_json_response = MagicMock(
            side_effect=[
                _JSONResponse(
                    data=[{"id": 1}],
                    headers={"Link": '<next>; rel="next"'},
                    status_code=200,
                ),
                _JSONResponse(
                    data=[{"id": 2}],
                    headers={"Link": '<next>; rel="next"'},
                    status_code=200,
                ),
            ]
        )

        result = gh.get_pr_comments_with_status(
            "owner/repo",
            1,
            max_items=2,
            max_pages=2,
        )

        assert [item["id"] for item in result.items] == [1, 2]
        assert result.outcome is RetrievalOutcome.PARTIAL
        assert result.truncated is True
        assert result.pages_fetched == 2

    def test_direct_requests_use_connect_and_read_timeouts(self):
        gh = ExtendedGithub("token", connect_timeout=2.5, read_timeout=9.0)
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = []
        session = MagicMock()
        session.get.return_value = response
        gh._build_session = MagicMock(return_value=session)

        result = gh._request_json_response("https://api.github.com/example")

        assert result.data == []
        session.get.assert_called_once()
        assert session.get.call_args.kwargs["timeout"] == (2.5, 9.0)
        session.close.assert_called_once()

    def test_refresh_token_updates_requester(self):
        gh = ExtendedGithub("token-1")
        gh.refresh_token("token-2")

        assert gh.access_token == "token-2"

    def test_close_connection_closes_client(self):
        manager = GitHubAuthManager()
        manager.github_instance = MagicMock()

        manager.close_connection()

        assert manager.github_instance is None
