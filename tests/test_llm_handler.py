from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llama_github.llm_integration.llm_handler import LLMHandler


@pytest.mark.asyncio
async def test_ainvoke_basic(fake_chat_model):
    mock_manager = MagicMock()
    mock_manager.get_llm.return_value = fake_chat_model
    mock_manager.get_llm_simple.return_value = fake_chat_model

    handler = LLMHandler(llm_manager=mock_manager)

    response = await handler.ainvoke("Hello")

    assert response == "AI Response"


def test_compose_chat_history():
    handler = LLMHandler(MagicMock())
    history = ["Hi", "Hello"]
    messages = handler._compose_chat_history_messages(history)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hi"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hello"


def test_compose_context_messages():
    handler = LLMHandler(MagicMock())
    context = ["ctx1", "ctx2"]
    messages = handler._compose_context_messages(context)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)


@pytest.mark.asyncio
async def test_ainvoke_uses_context_as_system_messages(fake_chat_model):
    mock_manager = MagicMock()
    mock_manager.get_llm.return_value = fake_chat_model
    mock_manager.get_llm_simple.return_value = fake_chat_model

    handler = LLMHandler(llm_manager=mock_manager)
    await handler.ainvoke("Hello", chat_history=["hi"], context=["ctx1"])

    message_types = [type(message).__name__ for message in fake_chat_model.messages]
    assert "SystemMessage" in message_types
    assert "HumanMessage" in message_types
