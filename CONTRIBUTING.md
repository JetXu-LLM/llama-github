# Contributing to llama-github

Thank you for your interest in contributing to `llama-github`.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please read and follow the guidelines to ensure a welcoming and inclusive environment for all contributors.

## Development Setup

Requirements:

- Python `3.10` through `3.14`

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Validation:

```bash
pytest -q
python -m build
python -m twine check dist/*
python -m pip_audit
```

## Guidelines

- Keep public API changes intentional and well documented.
- Add or update tests for behavioral changes.
- Update docs when changing return shapes, examples, or supported runtime versions.
- Prefer small, reviewable pull requests over broad rewrites.

## Pull Requests

- Use a clear title and description.
- Include migration notes when behavior changes in a user-visible way.
- Reference related issues when applicable.
- Make sure the repository still passes `pytest -q` and `python -m build`.

## License

By contributing to llama-github, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
