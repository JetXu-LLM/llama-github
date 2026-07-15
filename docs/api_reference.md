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
    embedding_revision=None,
    rerank_revision=None,
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
- `repo_cleanup_enabled`: set to `False` for request-scoped/serverless use
- `embedding_revision` / `rerank_revision`: immutable revision for a custom Hugging Face model

The built-in Jina models use package-pinned immutable revisions. Call `close()` when
the instance is no longer needed; `GithubRAG` also supports the context-manager protocol.

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

## `Repository.get_pr_content(number, pr=None, context_lines=10, force_update=False, *, related_issue_max_hits=None)`

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
- `_retrieval_meta`

`pr_metadata.head_sha` is the head commit captured by the same PR retrieval. Bounded
file/comment/review fetches publish `outcome`, `item_count`, and `truncated` under
`_retrieval_meta`; callers must not treat `partial` or `error` as absence proof.

Related-issue extraction and API expansion are capped at 20 unique references per PR
by default. Set `related_issue_max_hits` to an integer for a per-call override; the
packaged `related_issue_max_hits` configuration supplies the default. The
`_retrieval_meta.related_issues` object records `discovered_count`, `eligible_count`,
`attempted_count`, `successful_count`, `max_items`, and `truncated`; the current PR
number is excluded before applying the cap. Discovery reads only the PR title/body and
top-level PR comment bodies. It deliberately ignores file diffs, branch names, SHAs,
CI output, and other nested strings so code-shaped `#123` text is not treated as a
linked issue.

`interactions` stores top-level PR comments, review summaries, and each inline review
comment as separate chronological records. `reviews` and `review_comments` have
independent bounded-fetch metadata; `partial` or `error` means the interaction view is
incomplete, not that no more comments exist.

The method returns `None` if the PR cannot be retrieved.

## `Repository.read_text_file_bounded(file_path, sha=None, *, opt_in=None)`

Reads one UTF-8 repository file under a fixed 2 MiB source cap and returns a
`BoundedTextReadResult`. It does not change the older `get_file_content()`
exclusion rules or cache.

The result has these typed outcomes:

- `success`
- `excluded_by_policy`
- `not_found`
- `oversize`
- `binary_or_non_utf8`
- `directory`
- `error`

`to_meta()` returns content-free size, policy, status, and error metadata. The
cap is checked once against GitHub's declared file size and again while reading
or decoding the response, so metadata is never trusted as the only bound.
GitHub's Contents API has separate behavior for 1–100 MB objects; this API's
smaller cap is an intentional local product boundary. See the
[GitHub Contents API](https://docs.github.com/en/rest/repos/contents?apiVersion=2026-03-10).

Dependency lockfiles and CI configuration require one exact opt-in:

```python
from llama_github.data_retrieval import BoundedTextReadOptIn

lock_result = repo.read_text_file_bounded(
    "uv.lock",
    sha=head_sha,
    opt_in=BoundedTextReadOptIn.DEPENDENCY_LOCK,
)
workflow_result = repo.read_text_file_bounded(
    ".github/workflows/test.yml",
    sha=head_sha,
    opt_in=BoundedTextReadOptIn.CI_CONFIG,
)
```

An opt-in is not a general exclusion bypass. A mismatched opt-in and binary,
generated, minified, or sensitive paths remain `excluded_by_policy`.

## `Repository.get_ci_status_with_status(head_sha, *, max_statuses=100, max_check_runs=100)`

Fetches commit statuses and check runs for one exact head SHA without traversing the
pull request's commit history. It returns a `CIStatusSnapshot` containing:

- JSON-safe `statuses` and `check_runs`
- an aggregate retrieval `outcome`: `ok`, `no_hit`, `partial`, or `error`
- independent `statuses_meta` and `check_runs_meta`
- `to_dict()` for persistence or transport

The aggregate outcome describes retrieval completeness only. It deliberately does not
decide whether CI should block a merge. If one endpoint fails after the other succeeds,
the successful evidence is retained and the aggregate outcome is `partial`.
Commit statuses are collapsed to the newest result for each GitHub status context;
an obsolete failed attempt cannot override a later successful rerun just because its
target URL differs. Source metadata keeps the bounded fetched `item_count` and adds
`current_item_count` for the post-collapse snapshot.

## Status-aware search

The old `search_code()` and `search_issues()` methods still return lists. Consumers
that need to distinguish an empty successful search from a fetch failure should use:

```python
result = github.search_code_with_status("symbol repo:owner/name")
result = github.search_issues_with_status("regression repo:owner/name")
```

Both return `RetrievalResult` with:

- `items`
- `outcome`: `ok`, `no_hit`, `partial`, or `error`
- `pages_fetched`
- `truncated`
- `status_code`
- `error_type`

The expanded-content `GitHubAPIHandler` exposes methods with the same `*_with_status`
names and preserves status through file/issue expansion.
