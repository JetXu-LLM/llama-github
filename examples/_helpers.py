from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


def load_env_var(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value

    env_path = Path(".env")
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip("'").strip('"')
    return None


class FakeStructuredResponder:
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    async def ainvoke(self, messages):
        data = {}
        fields = self.schema.model_fields
        if "question" in fields:
            data["question"] = "How to create a NumPy array in Python?"
        if "answer" in fields:
            data["answer"] = "Use numpy.array([...]) to create an array."
        if "code_search_logic" in fields:
            data["code_search_logic"] = "Search for numpy.array usage examples."
        if "issue_search_logic" in fields:
            data["issue_search_logic"] = "Search for NumPy array creation questions."
        if "search_criteria" in fields:
            data["search_criteria"] = ["numpy array language:python"]
        if "necessity_score" in fields:
            data["necessity_score"] = 75
        if "score" in fields:
            data["score"] = 88
        return self.schema.model_validate(data)


class FakeChatModel:
    async def ainvoke(self, messages):
        return "Mock answer generated from injected contexts."

    def with_structured_output(self, schema: type[BaseModel]):
        return FakeStructuredResponder(schema)


def pretty_print(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
