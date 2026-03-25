# Usage

`llama-github` has two primary workflows:

- retrieve GitHub-derived context blocks with `retrieve_context()`
- answer a question from already available context with `answer_with_context()`

## Initialization

```python
from llama_github import GithubRAG

github_rag = GithubRAG(
    github_access_token="your_github_access_token",
    mistral_api_key="your_mistral_api_key",
)
```

Supported chat model strategies:

- `openai_api_key=...`
- `mistral_api_key=...`
- `llm=your_langchain_compatible_chat_model`

## Context Retrieval

```python
contexts = github_rag.retrieve_context("How do I create a NumPy array in Python?")
```

Return type:

```python
List[Dict[str, str]]
```

Each item contains at least:

- `context`
- `url`

## Simple Mode

```python
contexts = github_rag.retrieve_context(
    "How do I create a NumPy array in Python?",
    simple_mode=True,
)
```

`simple_mode=True`:

- skips embedding and reranker loading
- uses deterministic fallback ranking
- is the recommended mode for examples and smoke tests

## Answering With Existing Context

```python
answer = github_rag.answer_with_context(
    "How do I create a NumPy array in Python?",
    contexts=[
        {
            "context": "Use numpy.array([...]) to create a NumPy array.",
            "url": "https://numpy.org/doc/stable/reference/generated/numpy.array.html",
        }
    ],
)
```

`answer_with_context()` also accepts context items using a `content` key for backward compatibility.

## Asynchronous Usage

```python
import asyncio

async def main():
    contexts = await github_rag.async_retrieve_context(
        "How do I create a NumPy array in Python?"
    )
    print(contexts)

asyncio.run(main())
```

## Pull Request Content Retrieval

```python
repo = github_rag.RepositoryPool.get_repository("JetXu-LLM/llama-github")
pr_content = repo.get_pr_content(number=15)
print(pr_content["pr_metadata"]["title"])
```

This method is useful when you want structured PR metadata, changed files, interactions, and related issue context in one object.

## Logging

`llama-github` does not auto-configure logging on import. If you want library logs:

```python
import logging
from llama_github import configure_logging

configure_logging(level=logging.INFO)
```
