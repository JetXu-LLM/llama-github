# API Reference

This document provides a comprehensive reference for the public API of the `llama-github` library. It covers the main classes, methods, and their parameters.

## `GithubRAG` Class

The `GithubRAG` class is the main entry point for using the `llama-github` library. It provides methods for initializing the library, retrieving context, and configuring various aspects of the retrieval process.

### `__init__(self, github_access_token=None, github_app_credentials=None, openai_api_key=None, huggingface_token=None, jina_api_key=None, open_source_models_hg_dir=None, embedding_model=None, rerank_model=None, llm=None, **kwargs)`

Initializes a new instance of the `GithubRAG` class.

#### Parameters

- `github_access_token` (str, optional): GitHub access token for authentication. Defaults to `None`.
- `github_app_credentials` (GitHubAppCredentials, optional): Credentials for GitHub App authentication. Defaults to `None`.
- `openai_api_key` (str, optional): API key for OpenAI services. Recommended for using GPT-4-turbo. Defaults to `None`.
- `huggingface_token` (str, optional): Token for Hugging Face services. Recommended. Defaults to `None`.
- `jina_api_key` (str, optional): API key for Jina AI services. Used for high concurrency production deployment. Defaults to `None`.
- `open_source_models_hg_dir` (str, optional): Path to open-source models from Hugging Face to replace OpenAI. Defaults to `None`.
- `embedding_model` (str, optional): Name of the custom embedding model from Hugging Face. Defaults to the value specified in the configuration.
- `rerank_model` (str, optional): Name of the custom reranking model from Hugging Face. Defaults to the value specified in the configuration.
- `llm` (Any, optional): Custom LangChain LLM chat object to replace OpenAI or open-source models from Hugging Face. Defaults to `None`.
- `**kwargs`: Additional keyword arguments for configuring the repository pool caching mechanism.
  - `repo_cleanup_interval` (int, optional): Cache cleanup interval in seconds. Defaults to the value specified in the configuration.
  - `repo_max_idle_time` (int, optional): Maximum idle time for a cached repository in seconds. Defaults to the value specified in the configuration.

### `retrieve_context(self, query, simple_mode=False)`

Retrieves relevant context from GitHub based on the provided query.

#### Parameters

- `query` (str): The query or question to retrieve context for.
- `simple_mode` (bool, optional): Flag to enable simple mode retrieval. In simple mode, only a Google search is conducted based on the user's question. Defaults to `False`.

#### Returns

- `List[str]`: A list of relevant context strings retrieved from GitHub.

### `async_retrieve_context(self, query, simple_mode=False)`

Asynchronously retrieves relevant context from GitHub based on the provided query.

#### Parameters

- `query` (str): The query or question to retrieve context for.
- `simple_mode` (bool, optional): Flag to enable simple mode retrieval. In simple mode, only a Google search is conducted based on the user's question. Defaults to `False`.

#### Returns

- `List[str]`: A list of relevant context strings retrieved from GitHub.

## `GitHubAppCredentials` Class

The `GitHubAppCredentials` class represents the credentials required for authenticating with a GitHub App.

### `__init__(self, app_id, private_key, installation_id)`

Initializes a new instance of the `GitHubAppCredentials` class.

#### Parameters

- `app_id` (int): The ID of the GitHub App.
- `private_key` (str): The private key associated with the GitHub App.
- `installation_id` (int): The ID of the GitHub App installation.

## `configure_logging(level=logging.INFO)`

Configures the logging level for the `llama-github` library.

#### Parameters

- `level` (int, optional): The desired logging level. Defaults to `logging.INFO`.

## Conclusion

This API reference provides an overview of the main classes and methods available in the `llama-github` library. It serves as a complement to the usage guide and helps developers understand the available functionality and how to interact with the library programmatically.

For more detailed information on how to use these classes and methods, along with code examples, please refer to the [usage guide](usage.md).

If you have any questions or need further assistance, please open an issue on the [GitHub repository](https://github.com/JetXu-LLM/llama-github/issues) or reach out to the project maintainers.