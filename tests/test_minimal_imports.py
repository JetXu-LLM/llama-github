import subprocess
import sys
import textwrap


def test_retrieval_imports_do_not_load_heavy_rag_dependencies():
    script = textwrap.dedent(
        """
        import sys

        import llama_github
        assert "llama_github.github_rag" not in sys.modules

        from llama_github.data_retrieval.github_api import GitHubAPIHandler
        from llama_github.data_retrieval.github_entities import CIStatusSnapshot, Repository, RepositoryPool
        from llama_github.github_integration.github_auth_manager import GitHubAuthManager
        from llama_github.utils import AsyncHTTPClient, DiffGenerator

        forbidden = {"aiohttp", "langchain_core", "numpy", "pydantic", "tokenizers", "transformers"}
        loaded = forbidden.intersection(sys.modules)
        assert not loaded, sorted(loaded)
        assert callable(Repository.get_ci_status_with_status)
        assert llama_github.__version__ == "0.4.1"
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_lazy_public_github_rag_api_remains_available():
    from llama_github import GithubRAG

    assert GithubRAG.__name__ == "GithubRAG"
