import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_github.llm_integration.llm_handler import LLMHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

@pytest.mark.asyncio
async def test_ainvoke_basic():
    mock_manager = MagicMock()
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="AI Response")
    mock_manager.get_llm.return_value = mock_llm
    mock_manager.model_type = "OpenAI"
    
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