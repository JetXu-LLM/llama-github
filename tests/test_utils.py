import pytest
from llama_github.utils import DiffGenerator, DataAnonymizer, CodeAnalyzer

class TestDiffGenerator:
    def test_generate_custom_diff_simple(self):
        base = "line1\nline2\nline3"
        head = "line1\nline2 modified\nline3"
        diff = DiffGenerator.generate_custom_diff(base, head, context_lines=1)
        assert "line2 modified" in diff
        assert "line1" in diff

    def test_generate_custom_diff_new_file(self):
        base = None
        head = "new line"
        diff = DiffGenerator.generate_custom_diff(base, head, context_lines=1)
        assert "+ new line" in diff

    def test_generate_custom_diff_deleted_file(self):
        base = "old line"
        head = None
        diff = DiffGenerator.generate_custom_diff(base, head, context_lines=1)
        assert "- old line" in diff

    def test_find_context_python(self):
        lines = [
            "def my_func():",
            "    x = 1",
            "    y = 2"
        ]
        context = DiffGenerator._find_context(2, lines)
        assert context == "def my_func():"

class TestDataAnonymizer:
    def setup_method(self):
        self.anonymizer = DataAnonymizer()

    def test_anonymize_api_key(self):
        text = "api_key = 'sk-1234567890abcdef1234567890abcdef'"
        anonymized = self.anonymizer.anonymize_sensitive_data(text)
        assert "sk-" not in anonymized
        assert "<ANONYMIZED:" in anonymized

    def test_anonymize_email(self):
        text = "Contact me at test@example.com"
        anonymized = self.anonymizer.anonymize_sensitive_data(text)
        assert "test@example.com" not in anonymized

class TestCodeAnalyzer:
    def test_extract_imports(self):
        code = """
import os
import requests
from . import utils
        """
        imports = CodeAnalyzer.extract_imports(code)
        assert "os" in imports["standard_imports"]
        assert "requests" in imports["third_party_imports"]
        assert "utils" in [i["module"] for i in imports["from_imports"] if i["module"]] or \
               any(i for i in imports["local_imports"])

    def test_categorize_change(self):
        diff = """
+ def new_function():
+     pass
- import old_lib
        """
        categories = CodeAnalyzer.categorize_change(diff)
        assert "function_added" in categories
        assert "import_removed" in categories