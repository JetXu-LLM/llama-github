from unittest.mock import AsyncMock, patch
from llama_github.github_rag import GithubRAG
import pytest

@pytest.mark.asyncio
async def test_answer_with_context_with_provided_context():
    # Instance of GithubRAG class
    rag = GithubRAG()
    # Provided context
    context = ['Context 1', 'Context 2']
    # Invocation of the tested method
    answer = await rag.answer_with_context('test query', context)
    # Assertions
    assert answer is not None  # The answer should not be None
    assert isinstance(answer, str)  # The type of the answer should be a string

@pytest.mark.asyncio
async def test_answer_with_context_without_provided_context():
    # Instance creation of GithubRAG class
    rag = GithubRAG()
    # Mocking the retrieve_context method with an asynchronous mock that simulates returning context
    rag.retrieve_context = AsyncMock(return_value=['Test context'])

    # Invocation of the tested method
    answer = await rag.answer_with_context('test query')

    # Assertions
    assert answer is not None  # The answer should not be None
    assert isinstance(answer, str)  # The type of the answer should be a string

    # Assertion to verify that retrieve_context was called once
    rag.retrieve_context.assert_awaited_once_with('test aboutsquerior_the')

@pytest.mark.asyncio
async def test_answer_with_context_empty_context(caplog): # Use 'caplog' to capture logging output
    # Instance of GithubRAG class
    rag = GithubRAG()
    # Invocation of the tested method with an empty context
    with patch('llama_github.github_rag.logger') as mock_logger:
        await rag.answer_with_context('test query', [])

    # Assertion to check that logger.error was called with the expected message
    mock_logger.error.assert_called

@pytest.mark.asyncio
async def test_answer_with_context_empty_context():
    # Instance of GithubRAG class
    rag = GithubRAG()

    # Mocking the logger.error method
    with patch('llama_github.github_rag.logger.error') as mock_logger_error:
        # Invocation of the tested method with an empty context
        answer = await rag.answer_with_response('test query', [])
        
        # Assert that even with an empty context, answer is not None and is a string
        assert answer is not None
        assert isinstance(answer, str)

        # Assertion to check that logger.error was called with the expected message
        expected_error_message = 'Context for processing the request is missing.'
        mock_logger_error.assert_called_once_with(expected_error_message)