# llm_handler.py
# to do list
# 1. add streaming output for invoke.

from llama_github.llm_integration.initial_load import LLMManager
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from llama_github.config.config import Config
from llama_github.logger import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel

class LLMHandler:
    def __init__(self):
        """
        Initializes the LLMHandler class which is responsible for handling the interaction
        with a language model (LLM) using the LangChain framework.
        
        Attributes:
            llm_manager (LLMManager): Manages interactions with the language model.
        """
        self.llm_manager = LLMManager()

    async def invoke(self, human_question: str, chat_history: list[str] = None, context: list[str] = None, output_structure: BaseModel=None, prompt: str = Config().get("general_prompt")) -> str:
        """
        Asynchronously invokes the language model with a given question, chat history, and context,
        and returns the model's response.
        
        Parameters:
            human_question (str): The question or input from the human user.
            chat_history (list[str]): A list of strings representing the chat history, where each
                                      string is a message. This parameter is optional.
            context (list[str]): A list of strings representing additional context for the model.
                                 This parameter is optional.
            output_structure: A langchain_core.pydantic_v1.BaseModel object to control desired 
                              structure of the output from the language model.
                              This parameter is optional and allows for more detailed control over
                              the model's responses.
            prompt (str): A template for the prompt to be used with the language model. Defaults
                          to a general prompt defined in the configuration.

        Returns:
            str: The response from the language model.
        """
        try:
            llm = self.llm_manager.get_llm()
            if self.llm_manager.model_type == "OpenAI":
                # Define an output parser to interpret the model's raw output.
                output_parser = StrOutputParser()

                # Create a prompt template with placeholders for dynamic content.
                prompt_template = ChatMessagePromptTemplate.from_template(role="system", template=prompt)
                chat_prompt = ChatPromptTemplate.from_messages([
                    prompt_template,
                    MessagesPlaceholder(variable_name="history_messages", optional=True),
                    MessagesPlaceholder(variable_name="human_message"),
                    MessagesPlaceholder(variable_name="context_messages", optional=True)
                ])

                # Convert chat_history and context from [str] to their respective message types.
                chat_history_messages = self._compose_context_messages(chat_history)
                context_messages = self._compose_chat_history_messages(context)
                human_question_message = HumanMessage(content=human_question)

                prompt_params = {
                    "history_messages": chat_history_messages,
                    "human_message": [human_question_message],
                    "context_messages": context_messages
                }

                # Format the prompt with the provided parameters.
                formatted_prompt = chat_prompt.format_prompt(**prompt_params)

                # Determine the processing chain based on the presence of an output structure.
                if output_structure is not None:
                    chain = chat_prompt | llm.with_structured_output(output_structure) | output_parser
                else:
                    chain = chat_prompt | llm | output_parser

                # Invoke the chain and return the model's response.
                response = await chain.ainvoke(formatted_prompt.to_messages())
                return response
        except Exception as e:
            logger.exception(f"Call llm with #{human_question}# generated an exception:")
            return "An error occurred during processing."

    def _compose_chat_history_messages(self, chat_history: list[str]) -> list:
        """
        Converts chat history from a list of strings to a list of alternating HumanMessage
        and AIMessage objects, starting with HumanMessage.

        Parameters:
            chat_history (list[str]): The chat history as a list of strings.

        Returns:
            list: A list of alternating HumanMessage and AIMessage objects.
        """
        messages = []
        for i, message in enumerate(chat_history or []):
            message_class = HumanMessage if i % 2 == 0 else AIMessage
            messages.append(message_class(content=message))
        return messages

    def _compose_context_messages(self, context: list[str]) -> list:
        """
        Converts context from a list of strings to a list of SystemMessage objects.

        Parameters:
            context (list[str]): The context as a list of strings.

        Returns:
            list: A list of SystemMessage objects.
        """
        return [SystemMessage(content=message) for message in context or []]
