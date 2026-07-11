import io
import logging
from unittest.mock import MagicMock

import pytest

from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import RepositoryPool
from llama_github.logger import configure_logging
from llama_github.llm_integration.llm_handler import LLMHandler


def test_failed_search_does_not_log_raw_query():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    configure_logging(level=logging.DEBUG, handler=handler)
    secret_query = "private_symbol customer-secret-value"

    pool = RepositoryPool(None, cleanup_enabled=False)
    result = GitHubAPIHandler(None, pool=pool).search_code_with_status(secret_query)

    assert result.outcome.value == "error"
    output = stream.getvalue()
    assert secret_query not in output
    assert "customer-secret-value" not in output
    assert f"query_length={len(secret_query)}" in output


@pytest.mark.asyncio
async def test_failed_llm_call_does_not_log_question_or_context():
    stream = io.StringIO()
    configure_logging(level=logging.DEBUG, handler=logging.StreamHandler(stream))
    manager = MagicMock()
    manager.get_llm.return_value = None
    question = "private-question-value"

    result = await LLMHandler(manager).ainvoke(
        question,
        context=["private-context-value"],
    )

    assert result == "An error occurred during processing."
    output = stream.getvalue()
    assert question not in output
    assert "private-context-value" not in output
    assert f"question_length={len(question)}" in output
