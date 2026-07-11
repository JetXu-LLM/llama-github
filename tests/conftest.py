from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel


class FakeStructuredResponder:
    def __init__(self, schema: type[BaseModel], payload: dict | None = None):
        self.schema = schema
        self.payload = payload or {}
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self.payload:
            return self.schema.model_validate(self.payload)

        fields = self.schema.model_fields
        data = {}
        if "question" in fields:
            data["question"] = "Refined question"
        if "answer" in fields:
            data["answer"] = "Draft answer"
        if "code_search_logic" in fields:
            data["code_search_logic"] = "Search code"
        if "issue_search_logic" in fields:
            data["issue_search_logic"] = "Search issues"
        if "search_criteria" in fields:
            data["search_criteria"] = ["query1", "query2"]
        if "necessity_score" in fields:
            data["necessity_score"] = 85
        if "score" in fields:
            data["score"] = 90
        return self.schema.model_validate(data)


class FakeChatModel:
    def __init__(self, response="AI Response", structured_payload: dict | None = None):
        self.response = response
        self.structured_payload = structured_payload
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response

    def with_structured_output(self, schema: type[BaseModel]):
        return FakeStructuredResponder(schema, self.structured_payload)


@pytest.fixture
def fake_chat_model():
    return FakeChatModel()


@pytest.fixture
def mock_github_instance():
    mock = MagicMock()
    mock.get_user.return_value.login = "test_user"
    return mock


@pytest.fixture
def mock_repo_object():
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
    mock_file = MagicMock()
    mock_file.name = "test.py"
    mock_file.path = "src/test.py"
    mock_file.encoding = "base64"
    mock_file.content = "cHJpbnQoImhlbGxvIik="
    mock_file.decoded_content = b'print("hello")'
    return mock_file


@pytest.fixture
def mock_llm_handler():
    handler = MagicMock()
    handler.ainvoke = AsyncMock()
    return handler
