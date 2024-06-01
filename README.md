# llama-github

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Llama-github is an open-source Python library that empowers LLM Chatbots, AI Agents, and Auto-dev Agents to conduct Retrieval from actively selected GitHub public contents (Repository, Issue, Code). It Augments through LLMs and Generates context for any coding question, in order to streamline the development of sophisticated AI-driven applications.

## Installation
```
pip install llama-github
```

## Usage

Here's a simple example of how to use llama-github:

```python
from llama_github import GithubRAG

# Initialize GithubRAG with your credentials
github_rag = GithubRAG(
    github_access_token="your_github_access_token", 
    openai_api_key="your_openai_api_key", # Optional in Simple Mode
    jina_api_key="your_jina_api_key" # Optional - unless you want high concurrency production deployment (s.jina.ai API will be used in llama-github)
)

# Retrieve context for a coding question (simple_mode is default set to False)
query = "How to create a NumPy array in Python?"
context = github_rag.retrieve_context(
    query, # In professional mode, one query will take nearly 1 min to generate final contexts. You could set log level to INFO to monitor the retrieval progress
    # simple_mode = True
)

print(context)
```

For more advanced usage and examples, please refer to the [documentation](docs/usage.md).

## Key Features

- **🔍 Intelligent GitHub Retrieval**: Harness the power of llama-github to retrieve highly relevant code snippets, issues, and repository information from GitHub based on user queries. Our advanced retrieval techniques ensure you find the most pertinent information quickly and efficiently.

- **⚡ Repository Pool Caching**: Llama-github has an innovative repository pool caching mechanism. By caching repositories (including READMEs, structures, code, and issues) across threads, llama-github significantly accelerates GitHub search retrieval efficiency and minimizes the consumption of GitHub API tokens. Deploy llama-github in multi-threaded production environments with confidence, knowing that it will perform optimally and save you valuable resources.

- **🧠 LLM-Powered Question Analysis**: Leverage state-of-the-art language models to analyze user questions and generate highly effective search strategies and criteria. Llama-github intelligently breaks down complex queries, ensuring that you retrieve the most relevant information from GitHub's vast repository network.

- **📚 Comprehensive Context Generation**: Generate rich, contextually relevant answers by seamlessly combining information retrieved from GitHub with the reasoning capabilities of advanced language models. Llama-github excels at handling even the most complex and lengthy questions, providing comprehensive and insightful responses that include extensive context to support your development needs.

- **🚀 Asynchronous Processing Excellence**: Llama-github is built from the ground up to leverage the full potential of asynchronous programming. With meticulously implemented asynchronous mechanisms woven throughout the codebase, llama-github can handle multiple requests concurrently, significantly boosting overall performance. Experience the difference as llama-github efficiently manages high-volume workloads without compromising on speed or quality.

- **🔧 Flexible LLM Integration**: Easily integrate llama-github with various LLM providers, embedding models, and reranking models to tailor the library's capabilities to your specific requirements. Our extensible architecture allows you to customize and enhance llama-github's functionality, ensuring that it adapts seamlessly to your unique development environment.

- **🔒 Robust Authentication Options**: Llama-github supports both personal access tokens and GitHub App authentication, providing you with the flexibility to integrate it into different development setups. Whether you're an individual developer or working within an organizational context, llama-github has you covered with secure and reliable authentication mechanisms.

- **🛠️ Logging and Error Handling**: We understand the importance of smooth operations and easy troubleshooting. That's why llama-github comes equipped with comprehensive logging and error handling mechanisms. Gain deep insights into the library's behavior, quickly diagnose issues, and maintain a stable and reliable development workflow.


## Contributing

We welcome contributions to llama-github! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions, suggestions, or feedback, please feel free to reach out to us at [Voldemort.xu@foxmail.com](mailto:Voldemort.xu@foxmail.com).

---

Thank you for choosing llama-github! We hope this library enhances your AI development experience and helps you build powerful applications with ease.