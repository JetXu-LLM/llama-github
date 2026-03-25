# API Reference

## `GithubRAG`

Main entrypoint for retrieval and answer generation.

### Constructor

```python
GithubRAG(
    github_access_token=None,
    github_app_credentials=None,
    openai_api_key=None,
    mistral_api_key=None,
    huggingface_token=None,
    jina_api_key=None,
    open_source_models_hg_dir=None,
    embedding_model=None,
    rerank_model=None,
    llm=None,
    simple_mode=False,
    **kwargs,
)
```

Important parameters:

- `github_access_token`: personal access token for GitHub REST and search APIs
- `github_app_credentials`: app-based authentication credentials
- `openai_api_key`: OpenAI chat provider
- `mistral_api_key`: Mistral chat provider
- `llm`: injected LangChain-compatible chat model
- `simple_mode`: skip embedding and reranker loading
- `repo_cleanup_interval`: optional repository cache cleanup interval
- `repo_max_idle_time`: optional repository cache idle timeout

### `retrieve_context(query, simple_mode=None)`

Returns:

```python
List[Dict[str, str]]
```

Each item contains at least:

- `context`
- `url`

### `async_retrieve_context(query, simple_mode=None)`

Async version of `retrieve_context()`.

### `answer_with_context(query, contexts=None, simple_mode=False)`

Generates an answer from injected contexts or from newly retrieved contexts if `contexts` is `None`.

Accepted context item shapes:

- `{"context": "...", "url": "..."}`
- `{"content": "...", "url": "..."}` for backward compatibility

### `async_answer_with_context(query, contexts=None, simple_mode=False)`

Async version of `answer_with_context()`.

## `GitHubAppCredentials`

```python
GitHubAppCredentials(
    app_id: int,
    private_key: str,
    installation_id: int,
)
```

Used with `GithubRAG(github_app_credentials=...)`.

## `Repository.get_pr_content(number, pr=None, context_lines=10, force_update=False)`

Available from:

```python
repo = github_rag.RepositoryPool.get_repository("owner/repo")
```

Returns a dictionary containing:

- `pr_metadata`
- `related_issues`
- `commits`
- `file_changes`
- `ci_cd_results`
- `interactions`

The method returns `None` if the PR cannot be retrieved.
