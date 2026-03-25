from __future__ import annotations

import argparse
import asyncio

from llama_github import GithubRAG

from _helpers import FakeChatModel, load_env_var, pretty_print


async def run_mock(query: str) -> None:
    rag = GithubRAG(
        github_access_token="mock-token",
        llm=FakeChatModel(),
        simple_mode=True,
    )

    async def fake_google_search_retrieval(query: str):
        return [
            {
                "url": "https://example.com/numpy-array",
                "content": (
                    "NumPy arrays are usually created with numpy.array([...]). "
                    "Example: import numpy as np; arr = np.array([1, 2, 3])."
                ),
            },
            {
                "url": "https://example.com/python-list",
                "content": "Plain Python lists are not NumPy arrays.",
            },
        ]

    rag.google_search_retrieval = fake_google_search_retrieval
    contexts = await rag.async_retrieve_context(query, simple_mode=True)
    answer = await rag.async_answer_with_context(query, contexts=contexts, simple_mode=True)

    print("Mode: mock")
    print("Top contexts:")
    pretty_print(contexts)
    print("\nAnswer:")
    print(answer)


async def run_real(query: str) -> None:
    mistral_api_key = load_env_var("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise SystemExit("MISTRAL_API_KEY is required for --mode real.")

    rag = GithubRAG(mistral_api_key=mistral_api_key, simple_mode=True)
    contexts = [
        {
            "context": "NumPy arrays are created with numpy.array([...]).",
            "url": "https://numpy.org/doc/stable/reference/generated/numpy.array.html",
        },
        {
            "context": "Example: import numpy as np; arr = np.array([1, 2, 3])",
            "url": "https://numpy.org/doc/stable/reference/generated/numpy.array.html",
        },
    ]
    answer = await rag.async_answer_with_context(query, contexts=contexts, simple_mode=True)

    print("Mode: real")
    print("Injected contexts:")
    pretty_print(contexts)
    print("\nAnswer:")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run llama-github in mock mode or minimal-cost real-provider mode."
    )
    parser.add_argument(
        "--mode",
        choices=("mock", "real"),
        default="mock",
        help="mock is the default and does not require API keys.",
    )
    parser.add_argument(
        "--query",
        default="How do I create a NumPy array in Python?",
        help="Coding question to use in the example.",
    )
    args = parser.parse_args()

    runner = run_mock if args.mode == "mock" else run_real
    asyncio.run(runner(args.query))


if __name__ == "__main__":
    main()
