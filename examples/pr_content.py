from __future__ import annotations

import argparse

from llama_github import GithubRAG

from _helpers import load_env_var, pretty_print


MOCK_PR_CONTENT = {
    "pr_metadata": {
        "number": 15,
        "title": "Improve repository retrieval scoring",
        "description": "Refines ranking and updates docs.",
        "author": "octocat",
        "author_association": "CONTRIBUTOR",
        "created_at": "2025-08-24T00:00:00Z",
        "updated_at": "2025-08-24T00:10:00Z",
        "merged_at": None,
        "state": "open",
        "base_branch": "main",
        "head_branch": "feature/ranking",
    },
    "related_issues": [{"issue_number": 6, "issue_content": "Ranking improvements discussion"}],
    "commits": [{"sha": "abc123", "message": "Improve ranking logic"}],
    "file_changes": [{"file_path": "llama_github/rag_processing/rag_processor.py", "change_type": "modified"}],
    "ci_cd_results": {"state": "success", "statuses": [], "check_runs": []},
    "interactions": [],
}


def run_mock() -> None:
    print("Mode: mock")
    pretty_print(MOCK_PR_CONTENT)


def run_real(repo_name: str, pr_number: int, context_lines: int) -> None:
    github_access_token = load_env_var("GITHUB_ACCESS_TOKEN")
    if not github_access_token:
        raise SystemExit("GITHUB_ACCESS_TOKEN is required for --mode real.")

    rag = GithubRAG(github_access_token=github_access_token, simple_mode=True)
    repo = rag.RepositoryPool.get_repository(repo_name)
    pr_content = repo.get_pr_content(number=pr_number, context_lines=context_lines)
    pretty_print(pr_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the PR content schema in mock mode or fetch a real PR when GitHub credentials are available."
    )
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--repo", default="JetXu-LLM/llama-github")
    parser.add_argument("--pr-number", type=int, default=15)
    parser.add_argument("--context-lines", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock()
    else:
        run_real(args.repo, args.pr_number, args.context_lines)


if __name__ == "__main__":
    main()
