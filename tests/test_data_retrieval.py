from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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

        mock_readme = MagicMock()
        mock_readme.decoded_content = b"# Readme"
        mock_repo_object.get_readme.return_value = mock_readme

        content1 = repo.get_readme()
        content2 = repo.get_readme()

        assert content1 == "# Readme"
        assert content2 == "# Readme"
        assert mock_repo_object.get_readme.call_count == 1

    def test_get_file_content_base64(
        self, mock_github_instance, mock_repo_object, mock_content_file
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = mock_content_file

        repo = Repository("owner/test-repo", mock_github_instance)
        content = repo.get_file_content("src/test.py")

        assert content == 'print("hello")'


class TestRepositoryPool:
    def test_pool_is_instance_scoped(self, mock_github_instance):
        pool1 = RepositoryPool(mock_github_instance)
        pool2 = RepositoryPool(mock_github_instance)

        try:
            assert pool1 is not pool2
        finally:
            pool1.stop_cleanup()
            pool2.stop_cleanup()

    def test_get_repository_caching(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        pool = RepositoryPool(mock_github_instance)

        try:
            repo1 = pool.get_repository("owner/test-repo")
            repo2 = pool.get_repository("owner/test-repo")

            assert repo1 is repo2
            assert len(pool._pool) == 1
        finally:
            pool.stop_cleanup()

    def test_cleanup_logic(self, mock_github_instance):
        pool = RepositoryPool(mock_github_instance, cleanup_interval=0.1, max_idle_time=0.1)
        pool.stop_cleanup()

        mock_repo = MagicMock()
        mock_repo.last_read_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        mock_repo.creation_time = datetime(2000, 1, 1, tzinfo=timezone.utc)

        with pool._registry_lock:
            pool._pool["expired/repo"] = mock_repo
            pool._locks_registry["expired/repo"] = MagicMock()

        with patch("llama_github.data_retrieval.github_entities.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 1, tzinfo=timezone.utc)
            with pool._registry_lock:
                current_time = mock_dt.now(timezone.utc)
                if (current_time - mock_repo.last_read_time).total_seconds() > pool.max_idle_time:
                    pool._locks_registry.pop("expired/repo", None)
                    mock_repo.clear_cache()

        mock_repo.clear_cache.assert_called()


class TestGitHubAPIHandler:
    def test_search_code_integration(self, mock_github_instance):
        handler = GitHubAPIHandler(mock_github_instance)

        mock_code_result = {
            "name": "test.py",
            "path": "test.py",
            "repository": {"full_name": "owner/repo"},
            "html_url": "http://url",
        }

        mock_github_instance.search_code.return_value = [mock_code_result]

        with patch.object(handler, "get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_file_content.return_value = "content"
            mock_repo.stargazers_count = 1
            mock_repo.watchers_count = 1
            mock_repo.language = "Python"
            mock_repo.description = "desc"
            mock_repo.updated_at = datetime.now(timezone.utc)
            mock_get_repo.return_value = mock_repo

            results = handler.search_code("query")

            assert len(results) == 1
            assert results[0]["content"] == "content"
