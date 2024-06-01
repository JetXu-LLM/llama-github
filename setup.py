from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

setup(
    name="llama-github",
    version="0.1.0",
    author="Jet Xu",
    author_email="Voldemort.xu@foxmail.com",
    description="Llama-github is an open-source Python library that empowers LLM Chatbots, AI Agents, and Auto-dev Agents to conduct Retrieval from actively selected GitHub public projects. It Augments through LLMs and Generates context for any coding question, in order to streamline the development of sophisticated AI-driven applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JetXu-LLM/llama-github",
    packages=find_packages(include=['llama_github', 'llama_github.*']),
    include_package_data=True,
    package_data={
        'llama_github': ['config/config.json'],
    },
    install_requires=[req.strip() for req in requirements],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
