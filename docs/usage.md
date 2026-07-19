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

For request-scoped or serverless usage, pass `repo_cleanup_enabled=False`. For a
long-lived process, call `github_rag.close()` during shutdown or use `GithubRAG` as a
context manager.

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
print(pr_content["pr_metadata"]["head_sha"])
print(pr_content["_retrieval_meta"]["pr_files"])
```

This method is useful when you want structured PR metadata, changed files,
interactions, and related issue context in one object. Related issues come only from
the PR title/body and top-level PR comments. Review summaries and inline review
comments remain separate interaction records, so callers do not lose multiple inline
comments attached to one review. Plain `#123` is interpreted as a same-repository
reference, while explicit GitHub links and `owner/repository#123` references retain
their repository identity. Upstream release-note links are therefore not fetched from
the repository being reviewed; repository-pure upstream content embedded in bounded
HTML `details` blocks gives upstream scope only to unqualified `PR 123` / `issue 123`
phrases. Explicit `#123` shorthand remains local.

`_retrieval_meta` records bounded-fetch outcomes. A `partial` or `error` result is an
unknown, not evidence that a file, comment, or match does not exist.

Long-lived or memory-constrained consumers may opt into source limits without changing
the default API:

```python
pr_content = repo.get_pr_content(
    number=15,
    source_file_max_bytes=128 * 1024,
    source_total_max_bytes=16 * 1024 * 1024,
)
print(pr_content["_retrieval_meta"]["file_content_budget"])
```

Each base/head read is bounded before decode and during raw streaming. When bounded
source cannot be retained, the method uses GitHub's changed-file patch; if GitHub did
not provide one, it records an explicit skipped diff. The metadata contains only
limits and counts, not source content. Bounded and legacy calls have separate cache
identities.

To refresh CI evidence later without refetching the whole pull request:

```python
ci_snapshot = repo.get_ci_status_with_status(pr_content["pr_metadata"]["head_sha"])
print(ci_snapshot.outcome.value)
print(ci_snapshot.to_dict())
```

This helper is pinned to the supplied head SHA and keeps commit statuses and check runs
independently typed. Status history is reduced to the newest result per GitHub context,
while retrieval metadata retains both fetched and current item counts. Its aggregate
outcome is retrieval metadata, not a merge verdict.

## Bounded High-Intent Text Reads

Use the typed bounded API when a deterministic plan explicitly needs a lockfile
or CI configuration that the generic retrieval path intentionally excludes:

```python
from llama_github.data_retrieval import BoundedTextReadOptIn

result = repo.read_text_file_bounded(
    "uv.lock",
    sha=pr_content["pr_metadata"]["head_sha"],
    opt_in=BoundedTextReadOptIn.DEPENDENCY_LOCK,
)
if result.outcome.value == "success":
    print(result.content)
else:
    print(result.to_meta())
```

The 2 MiB source cap is a local cost and replayability boundary, not a GitHub
platform limit. `get_file_content()` remains the backward-compatible generic
API; callers should not use the bounded opt-ins as a broad file-policy bypass.

## Logging

`llama-github` does not auto-configure logging on import. If you want library logs:

```python
import logging
from llama_github import configure_logging

configure_logging(level=logging.INFO)
```

Library logs contain operation names, counts, lengths, status codes, and error types.
They intentionally omit raw queries, retrieved contexts, response bodies, and private
source content.
