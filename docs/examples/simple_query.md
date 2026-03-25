# Simple Query Example

Runnable script:

```bash
python examples/simple_query.py
```

Real-provider smoke mode:

```bash
python examples/simple_query.py --mode real
```

The default mode is mock/smoke and does not require API keys.

What it demonstrates:

- `GithubRAG` initialization
- `simple_mode=True`
- `retrieve_context()`
- `answer_with_context()`

The real mode is intentionally low-cost. It validates the provider path using short injected contexts instead of expensive end-to-end retrieval.
