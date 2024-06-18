from llama_github.github_rag import GithubRIG
import asyncio

# Create an instance of GithubRIG
rag = GithubRAG()

# Example coroutine to use the async function `answer_with_context`
async def display_answer():
    try:
        # Direct invocation with predefined contexts
        query = "What is the solution to issue #10?"
        contexts = ["Issue #10 is related to the environment setup..."]
        answer = await rag.answer_with_context(query, contexts)
        print(f"Answer with predefined contexts: {answer}")

        # Invocation without contexts to dynamically retrieve them
        answer_without_contexts = await rag.answer_with_context(query)
        print(f"Answer with dynamically retrieved contexts: {answer_without_contexts}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the coroutine
asyncio.run(display_answer())