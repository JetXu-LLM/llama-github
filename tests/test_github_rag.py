from unittest.mock import AsyncMock

import pytest

from llama_github.github_rag import GithubRAG


@pytest.mark.asyncio
async def test_answer_with_context_accepts_context_and_content_keys(fake_chat_model):
    rag = GithubRAG(
        llm=fake_chat_model,
        simple_mode=True,
        repo_cleanup_enabled=False,
    )

    answer = await rag.async_answer_with_context(
        "What changed?",
        contexts=[
            {"context": "ctx-a", "url": "a"},
            {"content": "ctx-b", "url": "b"},
        ],
    )

    assert answer == "AI Response"
    rag.close()


@pytest.mark.asyncio
async def test_simple_mode_does_not_require_embedding_models(fake_chat_model):
    rag = GithubRAG(
        llm=fake_chat_model,
        simple_mode=True,
        repo_cleanup_enabled=False,
    )
    rag.google_search_retrieval = AsyncMock(
        return_value=[{"url": "http://example.com", "content": "query result"}]
    )

    contexts = await rag.async_retrieve_context("query", simple_mode=True)

    assert contexts
    assert contexts[0]["url"] == "http://example.com"
    rag.close()


def test_context_manager_closes_repository_pool(fake_chat_model):
    with GithubRAG(
        llm=fake_chat_model,
        simple_mode=True,
        repo_cleanup_enabled=False,
    ) as rag:
        pool = rag.RepositoryPool

    assert pool._stop_event.is_set()


def test_legacy_positional_constructor_order_remains_compatible(fake_chat_model):
    rag = GithubRAG(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        fake_chat_model,
        True,
        repo_cleanup_enabled=False,
    )

    try:
        assert rag.simple_mode is True
        assert rag.llm_manager.llm is fake_chat_model
        assert rag.llm_manager.embedding_revision is not fake_chat_model
    finally:
        rag.close()
