# llama-github

[![PyPI version](https://badge.fury.io/py/llama-github.svg)](https://badge.fury.io/py/llama-github)
[![Downloads](https://static.pepy.tech/badge/Llama-github)](https://pepy.tech/project/Llama-github)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Llama-github is a powerful tool that helps you retrieve(based on Agentic RAG) the most relevant code snippets, issues, and repository information from GitHub based on your queries, transforming them into valuable knowledge context. It empowers LLM Chatbots, AI Agents, and Auto-dev Agents to solve complex coding tasks. Whether you're a developer looking for quick solutions or an engineer implementing advanced Auto Dev AI Agents, llama-github makes it easy and efficient.

If you like this project or believe it has potential, please give it a ⭐️. Your support is our greatest motivation!

## Architecture
![High Level Architecture](./docs/high_level_architecture.drawio.svg)

## Installation
```
pip install llama-github
```

Current maintained runtime target: Python `3.10+`.

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
contexts = github_rag.retrieve_context(
    query,
    # simple_mode = True
)

print(contexts[0]["url"])
print(contexts[0]["context"])
```

`retrieve_context()` returns a list of context dictionaries. Each item contains at least `context` and `url`.

For more advanced usage and examples, please refer to the [documentation](docs/usage.md). Runnable low-cost examples are also available in [`examples/`](examples).

## Key Features

- **🔍 Intelligent GitHub Retrieval**: Harness the power of llama-github to retrieve highly relevant code snippets, issues, and repository information from GitHub based on user queries. Our advanced retrieval techniques ensure you find the most pertinent information quickly and efficiently.

- **⚡ Repository Pool Caching**: Llama-github has an innovative repository pool caching mechanism. By caching repositories (including READMEs, structures, code, and issues) across threads, llama-github significantly accelerates GitHub search retrieval efficiency and minimizes the consumption of GitHub API tokens.

- **🧠 LLM-Powered Question Analysis**: Leverage state-of-the-art language models to analyze user questions and generate highly effective search strategies and criteria. Llama-github intelligently breaks down complex queries, ensuring that you retrieve the most relevant information from GitHub's vast repository network.

- **📚 Comprehensive Context Generation**: Generate rich, contextually relevant answers by seamlessly combining information retrieved from GitHub with the reasoning capabilities of advanced language models. Llama-github excels at handling even the most complex and lengthy questions, providing comprehensive and insightful responses that include extensive context to support your development needs.

- **🚀 Asynchronous Processing Excellence**: Llama-github is built from the ground up to leverage the full potential of asynchronous programming. With meticulously implemented asynchronous mechanisms woven throughout the codebase, llama-github can handle multiple requests concurrently, significantly boosting overall performance.

- **🔧 Flexible LLM Integration**: Easily integrate llama-github with various LLM providers, embedding models, reranking models, or an injected LangChain-compatible chat model to tailor the library's capabilities to your specific requirements.

- **🔒 Robust Authentication Options**: Llama-github supports both personal access tokens and GitHub App authentication, providing you with the flexibility to integrate it into different development setups. Whether you're an individual developer or working within an organizational context, llama-github has you covered with secure and reliable authentication mechanisms.

- **🛠️ Logging and Error Handling**: We understand the importance of smooth operations and easy troubleshooting. That's why llama-github comes equipped with comprehensive logging and error handling mechanisms. Gain deep insights into the library's behavior, quickly diagnose issues, and maintain a stable and reliable development workflow.

## 🤖 Try Our AI-Powered PR Review Assistant: LlamaPReview

If you find llama-github useful, you might also be interested in our AI-powered GitHub PR review assistant, LlamaPReview. It's designed to complement your development workflow and further enhance code quality.

### Key Features of LlamaPReview:
- 🚀 One-click installation, zero configuration required, fully auto-run
- 💯 Currently free to use - no credit card or payment info needed
- 🧠 AI-powered, automatic PR reviews with deep code understanding
- 🌐 Supports multiple programming languages

**LlamaPReview utilizes llama-github's advanced context retrieval and LLM-powered analysis** to provide intelligent, context-aware code reviews. It's like having a senior developer, armed with the full context of your repository, review every PR automatically!

👉 [Install LlamaPReview Now](https://github.com/marketplace/llamapreview/) (Free)

By using llama-github for context retrieval and LlamaPReview for code reviews, you can create a powerful, AI-enhanced development environment.

## Vision and Roadmap

### Vision

Our vision is to become a pivotal module in the future of AI-driven development solutions, seamlessly integrating with GitHub to empower LLMs in automatically resolving complex coding tasks.

![Vision Architecture](./docs/vision.drawio.svg)

### Roadmap

For a historical view of the earlier roadmap, please visit [Vision and Roadmap](VISION_AND_ROADMAP.md).

## Acknowledgments

We would like to express our gratitude to the following open-source projects for their support and contributions:

- **[LangChain](https://github.com/langchain-ai/langchain)**: For providing the foundational framework that empowers the LLM prompting and processing capabilities in llama-github.
- **[Jina.ai](https://github.com/jina-ai/reader)**: For offering s.jina.ai API and open source reranker and embedding models that enhance the accuracy and relevance of the generated contexts in llama-github.

Their contributions have been instrumental in the development of llama-github, and we highly recommend checking out their projects for more innovative solutions.

## Contributing

We welcome contributions to llama-github! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions, suggestions, or feedback, please feel free to reach out to us at [Jet Xu's email](mailto:Voldemort.xu@foxmail.com).

---

Thank you for choosing llama-github! We hope this library enhances your AI development experience and helps you build powerful applications with ease.
