# Usage

This document provides a comprehensive guide on how to use the `llama-github` library effectively. It covers various aspects of the library, including initialization, context retrieval, and advanced usage.

## Initialization

To start using `llama-github`, you need to initialize the `GithubRAG` class with the necessary credentials. Here's an example of how to initialize `GithubRAG`:

```python
from llama_github import GithubRAG

# Initialize GithubRAG with your credentials
github_rag = GithubRAG(
    github_access_token="your_github_access_token", 
    openai_api_key="your_openai_api_key", # Optional in Simple Mode
    jina_api_key="your_jina_api_key" # Optional - unless you want high concurrency production deployment (s.jina.ai API will be used in llama-github)
)
```

Make sure to replace `"your_github_access_token"`, `"your_openai_api_key"`, and `"your_jina_api_key"` with your actual credentials.

## Context Retrieval

The primary functionality of `llama-github` is to retrieve relevant context based on a given query. You can use the `retrieve_context` method to achieve this:

```python
query = "How to create a NumPy array in Python?"
context = github_rag.retrieve_context(query)
print(context)
```

The `retrieve_context` method takes a query string as input and returns a list of relevant context strings retrieved from GitHub.

### Simple Mode

By default, `retrieve_context` operates in professional mode, which performs a comprehensive search across code, issues, and repositories on GitHub. However, you can enable simple mode by setting the `simple_mode` parameter to `True`:

```python
context = github_rag.retrieve_context(query, simple_mode=True)
```

In simple mode, only a Google search is conducted based on the user's question. This mode is suitable for shorter queries (less than 20 words).

## Advanced Usage

### Asynchronous Processing

`llama-github` is built to leverage asynchronous programming for efficient processing. You can use the `async_retrieve_context` method to retrieve context asynchronously:

```python
import asyncio

async def retrieve_context_async():
    context = await github_rag.async_retrieve_context(query)
    print(context)

asyncio.run(retrieve_context_async())
```

This allows you to handle multiple requests concurrently and boost overall performance.

### Customizing LLM Integration

`llama-github` provides flexibility in integrating with different LLM providers, embedding models, and reranking models. You can customize these integrations during initialization:

```python
github_rag = GithubRAG(
    github_access_token="your_github_access_token",
    openai_api_key="your_openai_api_key",
    huggingface_token="your_huggingface_token",
    open_source_models_hg_dir="path/to/open_source_models",
    embedding_model="custom_embedding_model",
    rerank_model="custom_rerank_model",
    llm=custom_llm_object
)
```

- `openai_api_key`: API key for OpenAI services (recommended for using GPT-4-turbo).
- `huggingface_token`: Token for Hugging Face services (recommended).
- `open_source_models_hg_dir`: Path to open-source models from Hugging Face to replace OpenAI.
- `embedding_model`: Name of the custom embedding model from Hugging Face.
- `rerank_model`: Name of the custom reranking model from Hugging Face.
- `llm`: Custom LangChain LLM chat object to replace OpenAI or open-source models from Hugging Face.

### Authentication Options

`llama-github` supports both personal access tokens and GitHub App authentication. You can provide the necessary credentials during initialization:

```python
# Personal access token authentication
github_rag = GithubRAG(github_access_token="your_github_access_token")

# GitHub App authentication
github_app_credentials = GitHubAppCredentials(
    app_id=your_app_id,
    private_key="your_private_key",
    installation_id=your_installation_id
)
github_rag = GithubRAG(github_app_credentials=github_app_credentials)
```

Make sure to replace the placeholders with your actual credentials.

### Logging

Certainly! Here's an enhanced version of the logging section that emphasizes `llama-github`'s adherence to best practices for Python libraries:

## Logging

`llama-github` follows the best practices for logging in Python libraries by seamlessly integrating with the developer's main application logger. This approach ensures that the library's logging behavior aligns with the overall logging strategy of the application, providing a consistent and unified logging experience.

By default, `llama-github` does not configure its own logging settings to avoid interfering with the application's existing logging configuration. Instead, it respects the log levels and handlers set up by the developer in their main application.

To enable logging in `llama-github`, you simply need to configure the logging in your main application using Python's built-in `logging` module. For example:

```python
import logging

# Configure the main application's logger
logging.basicConfig(level=logging.INFO)

# Your application code goes here
```

In this example, the main application's logger is configured with a log level of `logging.INFO`. `llama-github` will automatically inherit this log level and emit log messages accordingly.

If you wish to have more control over the logging behavior specific to `llama-github`, you can use the `configure_logging` function provided by the library:

```python
from llama_github import configure_logging

# Configure llama-github's logger
configure_logging(level=logging.DEBUG)
```

By leveraging the flexibility and configurability of Python's `logging` module, `llama-github` provides developers with the tools necessary to gain valuable insights into the library's behavior and quickly identify and resolve any issues that may arise.

## Repository Pool Caching

`llama-github` utilizes an innovative repository pool caching mechanism to optimize performance and minimize GitHub API token consumption. The caching mechanism is automatically enabled and requires no additional configuration.

The repository pool caching works as follows:
- When a repository is accessed for the first time, it is fetched from the GitHub API and stored in the cache.
- Subsequent requests for the same repository retrieve the cached version, eliminating the need for additional API calls.
- The cache is thread-safe, allowing concurrent access from multiple threads without data inconsistencies.
- Cached repositories are periodically cleaned up based on their last access time to prevent the cache from growing indefinitely.

You can customize the caching behavior by providing additional parameters during initialization:

```python
github_rag = GithubRAG(
    github_access_token="your_github_access_token",
    repo_cleanup_interval=3600,  # Cache cleanup interval in seconds (default: 3600)
    repo_max_idle_time=7200      # Maximum idle time for a cached repository in seconds (default: 7200)
)
```

- `repo_cleanup_interval`: Specifies how often the cache cleanup process runs (default: 3600 seconds, i.e., 1 hour).
- `repo_max_idle_time`: Determines the maximum idle time for a cached repository before it is considered for removal (default: 7200 seconds, i.e., 2 hours).

The repository pool caching mechanism significantly improves performance by reducing the number of API calls made to GitHub, especially in scenarios where the same repositories are accessed frequently.

## Conclusion

`llama-github` provides a powerful and flexible solution for retrieving relevant context from GitHub based on user queries. By leveraging advanced retrieval techniques, LLM-powered question analysis, comprehensive context generation, and asynchronous processing, `llama-github` empowers developers to find the information they need quickly and efficiently.

With its support for different authentication methods, customizable LLM integrations, and robust logging capabilities, `llama-github` can be easily integrated into various development environments and tailored to specific requirements.

By following the usage guidelines outlined in this document and exploring the advanced features provided by `llama-github`, you can unlock the full potential of the library and enhance your development workflow.

For more information and examples, please refer to the [README](../README.md) and the [API documentation](api_reference.md).