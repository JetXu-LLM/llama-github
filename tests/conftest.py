import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
from datetime import datetime, timezone

# Mock external dependencies that might try to connect to internet or load heavy models
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_mistralai'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

@pytest.fixture
def mock_github_instance():
    """Mocks the ExtendedGithub instance."""
    mock = MagicMock()
    mock.get_user.return_value.login = "test_user"
    return mock

@pytest.fixture
def mock_repo_object():
    """Mocks a PyGithub Repository object."""
    mock_repo = MagicMock()
    mock_repo.id = 12345
    mock_repo.name = "test-repo"
    mock_repo.full_name = "owner/test-repo"
    mock_repo.description = "A test repository"
    mock_repo.html_url = "https://github.com/owner/test-repo"
    mock_repo.stargazers_count = 100
    mock_repo.subscribers_count = 10
    mock_repo.language = "Python"
    mock_repo.default_branch = "main"
    mock_repo.updated_at = datetime.now(timezone.utc)
    return mock_repo

@pytest.fixture
def mock_content_file():
    """Mocks a PyGithub ContentFile object."""
    mock_file = MagicMock()
    mock_file.name = "test.py"
    mock_file.path = "src/test.py"
    mock_file.encoding = "base64"
    mock_file.content = "cHJpbnQoImhlbGxvIik="  # print("hello") in base64
    mock_file.decoded_content = b'print("hello")'
    return mock_file

@pytest.fixture
def mock_llm_handler():
    """Mocks the LLMHandler."""
    handler = MagicMock()
    handler.ainvoke = AsyncMock()
    return handler