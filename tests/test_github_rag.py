from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_github.github_rag import GithubRAG


@pytest.mark.asyncio
async def test_answer_with_context_accepts_context_and_content_keys(fake_chat_model):
    rag = GithubRAG(llm=fake_chat_model, simple_mode=True)

    answer = await rag.async_answer_with_context(
        "What changed?",
        contexts=[
            {"context": "ctx-a", "url": "a"},
            {"content": "ctx-b", "url": "b"},
        ],
    )

    assert answer == "AI Response"


@pytest.mark.asyncio
async def test_simple_mode_does_not_require_embedding_models(fake_chat_model):
    rag = GithubRAG(llm=fake_chat_model, simple_mode=True)
    rag.google_search_retrieval = AsyncMock(
        return_value=[{"url": "http://example.com", "content": "query result"}]
    )

    contexts = await rag.async_retrieve_context("query", simple_mode=True)

    assert contexts
    assert contexts[0]["url"] == "http://example.com"
