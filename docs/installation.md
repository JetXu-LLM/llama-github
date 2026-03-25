# Installation

## Requirements

- Python `3.10+`

## Install From PyPI

```bash
pip install llama-github
```

## Install From Source

```bash
git clone https://github.com/JetXu-LLM/llama-github.git
cd llama-github
pip install .
```

## Development Setup

```bash
pip install -e .[dev]
```

## Verify The Install

```bash
python -c "import llama_github; print(llama_github.__version__)"
```

## Local Checks

```bash
pytest -q
python -m build
```
