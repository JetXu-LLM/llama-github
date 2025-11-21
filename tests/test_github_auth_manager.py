import pytest
from unittest.mock import patch, MagicMock
from llama_github.github_integration.github_auth_manager import GitHubAuthManager, ExtendedGithub

class TestGitHubAuthManager:
    def setup_method(self):
        self.auth_manager = GitHubAuthManager()

    @patch('llama_github.github_integration.github_auth_manager.ExtendedGithub')
    def test_authenticate_with_token(self, mock_extended_github):
        token = "fake_token"
        instance = self.auth_manager.authenticate_with_token(token)
        
        assert self.auth_manager.access_token == token
        assert instance == mock_extended_github.return_value
        mock_extended_github.assert_called_with(login_or_token=token)

    @patch('llama_github.github_integration.github_auth_manager.GithubIntegration')
    @patch('llama_github.github_integration.github_auth_manager.ExtendedGithub')
    def test_authenticate_with_app(self, mock_extended_github, mock_integration):
        # Setup mocks
        mock_integration_instance = mock_integration.return_value
        mock_integration_instance.get_access_token.return_value.token = "app_token"
        
        instance = self.auth_manager.authenticate_with_app(1, "key", 123)
        
        assert self.auth_manager.access_token == "app_token"
        mock_extended_github.assert_called_with(login_or_token="app_token")

class TestExtendedGithub:
    @patch('requests.get')
    def test_get_repo_structure(self, mock_get):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tree": [
                {"path": "folder", "type": "tree"},
                {"path": "folder/file.py", "type": "blob", "size": 100}
            ]
        }
        mock_get.return_value = mock_response

        gh = ExtendedGithub("token")
        structure = gh.get_repo_structure("owner/repo")

        assert "folder" in structure
        assert "file.py" in structure["folder"]["children"]
        assert structure["folder"]["children"]["file.py"]["size"] == 100

    @patch('requests.Session')
    def test_search_code_retry_logic(self, mock_session):
        mock_adapter = MagicMock()
        mock_session.return_value.mount = MagicMock()
        
        gh = ExtendedGithub("token")
        # We just want to ensure no exception is raised and session is used
        with patch.object(gh, 'access_token', 'token'):
             # Mock the get call to raise then succeed or just return
             mock_session.return_value.get.return_value.status_code = 200
             mock_session.return_value.get.return_value.json.return_value = {'items': []}
             
             result = gh.search_code("query")
             assert isinstance(result, list)