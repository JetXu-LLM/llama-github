# Data Collection Policy

`llama-github` is a library, not a hosted service. It does not operate its own backend or persist user data to a project-owned server.

## What The Library Does

Depending on how you configure it, the library may call external services such as:

- GitHub APIs
- OpenAI APIs
- Mistral APIs
- Jina search APIs

Those calls happen from the environment where you run the library.

## What The Library Stores

The library keeps in-memory process-local caches to reduce repeated GitHub API calls. Examples include:

- repository objects
- README content
- file contents
- issue content
- pull request content

This caching is runtime memory only unless your own application persists data separately.

## Credentials And Sensitive Data

You are responsible for how credentials and prompts are supplied and stored in your own environment. This can include:

- GitHub access tokens
- GitHub App credentials
- OpenAI API keys
- Mistral API keys
- Jina API keys
- user queries
- retrieved contexts

## Third-Party Policies

When you use `llama-github`, your requests are subject to the policies and retention rules of the third-party services you call through it. Review those services directly if you need legal or compliance guarantees.
