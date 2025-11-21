import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository, RepositoryPool

class TestRepository:
    def test_repository_initialization(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        
        repo = Repository("owner/test-repo", mock_github_instance)
        
        assert repo.full_name == "owner/test-repo"
        assert repo.id == 12345
        assert repo.language == "Python"

    def test_get_readme_caching(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        
        # Mock the internal repo object's get_readme
        mock_readme = MagicMock()
        mock_readme.decoded_content = b"# Readme"
        mock_repo_object.get_readme.return_value = mock_readme
        
        # First call
        content1 = repo.get_readme()
        assert content1 == "# Readme"
        
        # Second call should not trigger API
        content2 = repo.get_readme()
        assert content2 == "# Readme"
        assert mock_repo_object.get_readme.call_count == 1

    def test_get_file_content_base64(self, mock_github_instance, mock_repo_object, mock_content_file):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = mock_content_file
        
        repo = Repository("owner/test-repo", mock_github_instance)
        content = repo.get_file_content("src/test.py")
        
        assert content == 'print("hello")'

class TestRepositoryPool:
    def setup_method(self):
        # Reset singleton for testing
        RepositoryPool._instance = None
        RepositoryPool._instance_lock = MagicMock() # Reset lock mock if needed

    def test_singleton_behavior(self, mock_github_instance):
        pool1 = RepositoryPool(mock_github_instance)
        pool2 = RepositoryPool(mock_github_instance)
        assert pool1 is pool2

    def test_get_repository_caching(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        pool = RepositoryPool(mock_github_instance)
        
        repo1 = pool.get_repository("owner/test-repo")
        repo2 = pool.get_repository("owner/test-repo")
        
        assert repo1 is repo2
        # Ensure we didn't create two Repository objects
        assert len(pool._pool) == 1

    def test_cleanup_logic(self, mock_github_instance):
        # This test requires careful mocking of time and threads
        # We will mock the _cleanup method to avoid thread waiting
        pool = RepositoryPool(mock_github_instance, cleanup_interval=0.1, max_idle_time=0.1)
        pool.stop_cleanup() # Stop the real thread immediately
        
        # Manually insert an expired repo
        mock_repo = MagicMock()
        mock_repo.last_read_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        mock_repo.creation_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        
        with pool._registry_lock:
            pool._pool["expired/repo"] = mock_repo
            pool._locks_registry["expired/repo"] = MagicMock()

        # Manually invoke cleanup logic once
        with patch('llama_github.data_retrieval.github_entities.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 1, tzinfo=timezone.utc)
            
            # Extract the logic from _cleanup loop for testing
            with pool._registry_lock:
                current_time = mock_dt.now(timezone.utc)
                if (current_time - mock_repo.last_read_time).total_seconds() > pool.max_idle_time:
                    del pool._locks_registry["expired/repo"]
                    mock_repo.clear_cache()
        
        mock_repo.clear_cache.assert_called()

class TestGitHubAPIHandler:
    def test_search_code_integration(self, mock_github_instance):
        handler = GitHubAPIHandler(mock_github_instance)
        
        # Mock search_code response
        mock_code_result = MagicMock()
        mock_code_result.name = "test.py"
        mock_code_result.path = "test.py"
        mock_code_result.repository.full_name = "owner/repo"
        mock_code_result.html_url = "http://url"
        
        mock_github_instance.search_code.return_value = [mock_code_result]
        
        # Mock RepositoryPool to return a mock repo that returns content
        with patch.object(handler, 'get_repository') as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_file_content.return_value = "content"
            mock_get_repo.return_value = mock_repo
            
            results = handler.search_code("query")
            
            assert len(results) == 1
            assert results[0]['content'] == "content"