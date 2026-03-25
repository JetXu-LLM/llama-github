# PR Content Retrieval

`Repository.get_pr_content()` is the main entrypoint for pull-request level context.

## Example

```python
from llama_github import GithubRAG

github_rag = GithubRAG(
    github_access_token="your_github_access_token",
    simple_mode=True,
)

repo = github_rag.RepositoryPool.get_repository("JetXu-LLM/llama-github")
pr_content = repo.get_pr_content(number=15)

print(pr_content["pr_metadata"]["title"])
print(pr_content["pr_metadata"]["state"])
print(len(pr_content["file_changes"]))
print(len(pr_content["interactions"]))
```

## Return Shape

Top-level keys:

- `pr_metadata`
- `related_issues`
- `commits`
- `file_changes`
- `ci_cd_results`
- `interactions`

`pr_metadata` includes:

- `number`
- `title`
- `description`
- `author`
- `author_association`
- `created_at`
- `updated_at`
- `merged_at`
- `state`
- `base_branch`
- `head_branch`

## Notes

- Results are cached in memory within the repository pool.
- Use `force_update=True` if you want to refresh a previously cached PR.
- This method is intended for downstream analysis and review-style tooling rather than end-user presentation.
