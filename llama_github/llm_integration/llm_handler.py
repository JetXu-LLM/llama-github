# llm_handler.py
# to do list
# 1. add streaming output for infer.
from initial_load import LLMManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
                ("system", "You are a highly intelligent assistant with expertise in GitHub repositories and coding practices. \
Your task is to analyze questions related to GitHub projects, coding issues, or programming concepts. \
Using your extensive knowledge base, you will provide detailed, accurate, and contextually relevant answers. \
You have the ability to understand complex coding queries, retrieve pertinent information from GitHub repositories, and augment this data with your advanced reasoning capabilities. \
Your responses should guide developers towards solving their problems, understanding new concepts, or finding the information they seek related to GitHub projects and software development."),
                ("user", "{input}")
            ])
            # Define an output parser
            output_parser = StrOutputParser()
            # Combine components into a chain
            chain = prompt | llm | output_parser
            # Invoke the chain
            response = chain.invoke({"input": question})
            return response