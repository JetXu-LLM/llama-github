# llm_handler.py
# to do list
# 1. add streaming output for infer.
from initial_load import LLMManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_github.config.config import Config
from llama_github.logger import logger

class LLMHandler:
    def __init__(self):
        self.llm_manager = LLMManager()

    async def infer_with_template(self, system, prompt, question):
        llm = self.llm_manager.get_llm()
        if (self.llm_manager.model_type == "OpenAI"):
            # Define an output parser
            output_parser = StrOutputParser()
            # Combine components into a chain
            chain = prompt | llm | output_parser
            # Invoke the chain
            response = await chain.ainvoke({"input": question})
            return response

    def infer(self, question):
        llm = self.llm_manager.get_llm()
        if (self.llm_manager.model_type == "OpenAI"):
            # Define a prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", Config().get("general_prompt")),
                ("user", "{input}")
            ])
            # Define an output parser
            output_parser = StrOutputParser()
            # Combine components into a chain
            chain = prompt | llm | output_parser
            # Invoke the chain
            response = chain.invoke({"input": question})
            return response