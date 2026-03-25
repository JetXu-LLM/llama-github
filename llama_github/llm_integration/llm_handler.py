from __future__ import annotations

from typing import Optional, Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from pydantic import BaseModel

from llama_github.config.config import config
from llama_github.llm_integration.initial_load import LLMManager
from llama_github.logger import logger


class LLMHandler:
    """
    Lightweight adapter that turns user questions, history, and contexts into chat-model calls.

    The handler keeps prompt construction centralized so the retrieval pipeline and
    answer-generation pipeline share the same message formatting logic.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Create a handler backed by an existing LLMManager or a default one."""
        self.llm_manager = llm_manager if llm_manager is not None else LLMManager()

    async def ainvoke(
        self,
        human_question: str,
        chat_history: Optional[list[str]] = None,
        context: Optional[list[str]] = None,
        output_structure: Optional[Type[BaseModel]] = None,
        prompt: str = config.get("general_prompt"),
        simple_llm: bool = False,
    ):
        """
        Invoke the configured chat model.

        Args:
            human_question: The current user question.
            chat_history: Flat alternating user/assistant history.
            context: Extra context snippets inserted as system messages.
            output_structure: Optional pydantic schema for structured output.
            prompt: System prompt template.
            simple_llm: Prefer the lightweight model when available.

        Returns:
            Structured output when `output_structure` is provided; otherwise a string response.
        """
        try:
            llm = (
                self.llm_manager.get_llm_simple()
                if simple_llm and self.llm_manager.get_llm_simple() is not None
                else self.llm_manager.get_llm()
            )
            if llm is None:
                raise RuntimeError(
                    "No chat model is configured. Provide a compatible llm object or provider API key."
                )

            prompt_template = ChatMessagePromptTemplate.from_template(
                role="system",
                template=prompt,
            )
            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    prompt_template,
                    MessagesPlaceholder(variable_name="history_messages", optional=True),
                    MessagesPlaceholder(variable_name="human_message"),
                    MessagesPlaceholder(variable_name="context_messages", optional=True),
                ]
            )

            chat_history_messages = self._compose_chat_history_messages(chat_history)
            context_messages = self._compose_context_messages(context)
            human_question_message = HumanMessage(content=human_question)
            formatted_prompt = chat_prompt.format_prompt(
                history_messages=chat_history_messages,
                human_message=[human_question_message],
                context_messages=context_messages,
            )

            chain = llm
            if output_structure is not None:
                if not hasattr(llm, "with_structured_output"):
                    raise RuntimeError(
                        f"{type(llm).__name__} does not support structured output."
                    )
                chain = llm.with_structured_output(output_structure)

            response = await chain.ainvoke(formatted_prompt.to_messages())
            if output_structure is None and hasattr(response, "content"):
                return response.content
            return response
        except Exception as exc:
            logger.exception(
                "Call %sllm with #%s# generated an exception: %s",
                "simple " if simple_llm else "",
                human_question,
                exc,
            )
            if output_structure is not None:
                raise
            return "An error occurred during processing."

    def _compose_chat_history_messages(self, chat_history: Optional[list[str]]) -> list:
        """Convert flat alternating history strings into LangChain message objects."""
        messages = []
        for index, message in enumerate(chat_history or []):
            message_class = HumanMessage if index % 2 == 0 else AIMessage
            messages.append(message_class(content=message))
        return messages

    def _compose_context_messages(self, context: Optional[list[str]]) -> list:
        """Convert context strings into system messages so they stay separated from chat history."""
        return [SystemMessage(content=message) for message in context or []]
