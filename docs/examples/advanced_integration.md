# PR Content Retrieval in llama-github

## Introduction
The `get_pr_content` method is a new feature in the `Repository` class of the llama-github project. It provides a comprehensive way to retrieve and cache detailed information about Pull Requests (PRs) using a singleton pattern. This feature is designed to support LLM-assisted PR analysis and question-answering capabilities.

## Usage

```python
from llama_github import GithubRAG

github_rag=GithubRAG(github_access_token=github_access_token)
repo = github_rag.RepositoryPool.get_repository("JetXu-LLM/llama-github")
pr_content = repo.get_pr_content(number=15)
```

## Features

The `get_pr_content` method offers the following key features:

1. **Singleton Pattern**: Efficiently caches PR data to minimize API calls and improve performance.
2. **Comprehensive PR Metadata**: Retrieves detailed information including:
   - Title and description
   - PR state (open, closed, merged)
   - Author details
   - Creation and last update timestamps
3. **File Change Analysis**: 
   - Custom diff generation for changed files
   - Code categorization (added, modified, deleted)
4. **CI/CD Integration**: Fetches the latest CI/CD run results associated with the PR.
5. **Related Issues**: Identifies and links related issues mentioned in the PR.
6. **Comment Threading**: Retrieves all comments and reviews in a threaded format.

## Return Value

The method returns a dictionary containing all the retrieved PR information, structured for easy access and processing by LLMs.

## Error Handling

If the PR doesn't exist or there's an issue with the API call, the method will return `None` and log the error for debugging purposes.

## Performance Considerations

- The first call to `get_pr_content` for a specific PR will fetch all data from GitHub's API.
- Subsequent calls for the same PR will return the cached data, significantly reducing API usage and response time.
- The cache is automatically refreshed if the PR has been updated since the last retrieval.

## Example

```python
pr_content = repo.get_pr_content(123)
if pr_content:
    print(f"PR Title: {pr_content['title']}")
    print(f"PR State: {pr_content['state']}")
    print(f"Files Changed: {len(pr_content['files'])}")
    print(f"Comments: {len(pr_content['comments'])}")
else:
    print("Failed to retrieve PR content")
```

This new feature enhances the capabilities of llama-github, enabling more sophisticated PR analysis and interaction through LLM integration.