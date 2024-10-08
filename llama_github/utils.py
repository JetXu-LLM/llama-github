import hashlib
import base64
import re
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List, Tuple
from llama_github.logger import logger
import difflib
import ast

class DiffGenerator:
    """
    A class for generating custom diffs between two pieces of content.
    """

    @staticmethod
    def generate_custom_diff(base_content: str, head_content: str, context_lines: int) -> str:
        """
        Generate a custom diff between two pieces of content with specified context lines.

        Args:
            base_content (str): The original content.
            head_content (str): The new content to compare against the base.
            context_lines (int): The number of context lines to include in the diff.

        Returns:
            str: A string representation of the unified diff.

        Raises:
            ValueError: If context_lines is negative.
        """
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        if base_content is None and head_content is None:
            return ""  # Both contents are None, no diff to generate
        elif base_content is None:
            # File is newly added
            return "".join(f"+ {line}\n" for line in head_content.splitlines())
        elif head_content is None:
            # File is deleted
            return "".join(f"- {line}\n" for line in base_content.splitlines())

        base_lines: List[str] = base_content.splitlines()
        head_lines: List[str] = head_content.splitlines()

        try:
            diff: List[str] = list(difflib.unified_diff(base_lines, head_lines, n=context_lines, lineterm=''))
            return '\n'.join(diff)
        except Exception as e:
            logger.exception(f"Error generating diff: {str(e)}")
            return ""


class DataAnonymizer:
    def __init__(self):
        self.patterns = {
            'api_key': r'(?i)(api[_-]?key|sk[_-]live|sk[_-]test|sk[_-]prod|sk[_-]sandbox|openai[_-]?key)\s*[:=]\s*[\'"]?([a-zA-Z0-9-_]{20,})[\'"]?',
            'token': r'(?i)(token|access[_-]?token|auth[_-]?token|github[_-]?token|ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|ghu_[a-zA-Z0-9]{36}|ghr_[a-zA-Z0-9]{36}|ghs_[a-zA-Z0-9]{36})\s*[:=]\s*[\'"]?([a-zA-Z0-9-_]{20,})[\'"]?',
            'password': r'(?i)password\s*[:=]\s*[\'"]?([^\'"]+)[\'"]?',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'jwt': r'eyJ[a-zA-Z0-9-_]+\.eyJ[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+',
            'phone_number': r'\+?[0-9]{1,4}?[-.\s]?(\(?\d{1,3}?\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
            'ssn': r'\b(?:\d[ -]*?){9}\b',
            'ipv6': r'(?i)([0-9a-f]{1,4}:){7}([0-9a-f]{1,4}|:)',
            'mac_address': r'(?i)([0-9a-f]{2}([:-]|$)){6}',
            'latitude_longitude': r'(?i)(lat|latitude|lon|longitude)\s*[:=]\s*[-+]?([0-9]*\.[0-9]+|[0-9]+),\s*[-+]?([0-9]*\.[0-9]+|[0-9]+)',
            'driver_license': r'(?i)([A-Z0-9]{1,20})\s*[:=]\s*([A-Z0-9]{1,20})',
            'date_of_birth': r'(?i)(dob|date[_-]?of[_-]?birth)\s*[:=]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})',
            'name': r'(?i)(name|first[_-]?name|last[_-]?name)\s*[:=]\s*([a-zA-Z]{2,})',
            'address': r'(?i)(address|street[_-]?address)\s*[:=]\s*([a-zA-Z0-9\s,]{10,})',
            'zipcode': r'(?i)(zip|zipcode)\s*[:=]\s*([0-9]{5})',
            'company': r'(?i)(company|organization)\s*[:=]\s*([a-zA-Z\s]{2,})',
            'job_title': r'(?i)(job[_-]?title)\s*[:=]\s*([a-zA-Z\s]{2,})',
            'domain': r'(?i)(domain)\s*[:=]\s*([a-zA-Z0-9.-]{2,})',
            'hostname': r'(?i)(hostname)\s*[:=]\s*([a-zA-Z0-9.-]{2,})',
            'port': r'(?i)(port)\s*[:=]\s*([0-9]{2,})',
        }

    def hash_replacement(match):
        sensitive_data = match.group(0)
        hash_object = hashlib.sha256(sensitive_data.encode())
        hashed_data = base64.urlsafe_b64encode(
            hash_object.digest()).decode('utf-8')
        return f'<ANONYMIZED:{hashed_data[:8]}>'

    def anonymize_sensitive_data(self, question):
        anonymized_question = question
        for pattern_name, pattern in self.patterns.items():
            anonymized_question = re.sub(
                pattern, self.hash_replacement, anonymized_question)
        return anonymized_question

class AsyncHTTPClient:
    """
    Asynchronous HTTP client class for sending asynchronous HTTP requests.
    """

    @staticmethod
    async def request(
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 1,
        retry_delay: int = 1,
    ) -> Optional[aiohttp.ClientResponse]:
        """
        Send an asynchronous HTTP request.

        :param url: The URL to send the request to.
        :param method: The HTTP request method, default is "GET".
        :param headers: The request headers, default is None.
        :param data: The request data, default is None.
        :param retry_count: The number of retries, default is 1.
        :param retry_delay: The delay in seconds between each retry, default is 1.
        :return: The response object if the request is successful, otherwise None.
        """
        async with aiohttp.ClientSession() as session:
            for attempt in range(retry_count):
                try:
                    async with session.request(
                        method, url, headers=headers, json=data
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.error(
                                f"Request failed with status code: {response.status}. "
                                f"Retrying ({attempt + 1}/{retry_count})..."
                            )
                except aiohttp.ClientError as e:
                    logger.error(
                        f"Request failed with error: {str(e)}. "
                        f"Retrying ({attempt + 1}/{retry_count})..."
                    )

                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)

        return None

class CodeAnalyzer:
    """
    A utility class for analyzing Python code.
    
    This class provides methods for extracting abstract syntax trees,
    analyzing imports, and categorizing code changes.
    """

    @staticmethod
    def get_ast_representation(code_str: str) -> Optional[str]:
        """
        Parses code into an Abstract Syntax Tree (AST) representation.

        :param code_str: The code string to parse.
        :return: String representation of the AST or None if parsing fails.
        """
        if not code_str:
            return None
        try:
            tree = ast.parse(code_str)
            return ast.dump(tree)
        except SyntaxError:
            logger.error("Syntax error in the provided code")
            return None

    @staticmethod
    def extract_imports(code_str: str) -> Dict[str, Any]:
        """
        Extracts detailed import information from the given code string.

        :param code_str: The code string to analyze.
        :return: A dictionary containing detailed import information.
        """
        import_info = {
            "standard_imports": [],
            "third_party_imports": [],
            "local_imports": [],
            "from_imports": [],
            "errors": []
        }

        if not code_str:
            return import_info

        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            logger.error(f"Syntax error in the provided code: {e}")
            import_info["errors"].append(str(e))
            return import_info

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    CodeAnalyzer._categorize_import(alias.name, import_info)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_import = {
                        "module": node.module,
                        "names": [n.name for n in node.names],
                        "level": node.level
                    }
                    import_info["from_imports"].append(from_import)
                    CodeAnalyzer._categorize_import(node.module, import_info)

        return import_info

    @staticmethod
    def _categorize_import(module_name: str, import_info: Dict[str, List[str]]) -> None:
        """
        Categorizes an import as standard library, third-party, or local.

        :param module_name: The name of the module to categorize.
        :param import_info: The dictionary to update with the categorized import.
        """
        std_libs = set(CodeAnalyzer._get_standard_library_modules())
        
        if module_name in std_libs:
            import_info["standard_imports"].append(module_name)
        elif '.' in module_name:
            import_info["local_imports"].append(module_name)
        else:
            import_info["third_party_imports"].append(module_name)

    @staticmethod
    def _get_standard_library_modules() -> List[str]:
        """
        Returns a list of Python standard library module names.

        :return: List of standard library module names.
        """
        import sys
        import pkgutil
        
        std_lib = []
        for module in pkgutil.iter_modules():
            if module.name not in sys.builtin_module_names:
                try:
                    spec = pkgutil.find_loader(module.name)
                    if spec is not None:
                        if hasattr(spec, 'get_filename'):
                            pathname = spec.get_filename()
                        elif hasattr(spec, 'origin'):
                            pathname = spec.origin
                        else:
                            pathname = None
                        
                        if pathname and 'site-packages' not in pathname:
                            std_lib.append(module.name)
                except Exception as e:
                    logger.warning(f"Error processing module {module.name}: {e}")
                    continue
        
        return std_lib + list(sys.builtin_module_names)

    @staticmethod
    def analyze_imports(code_str: str) -> Tuple[Dict[str, Any], str]:
        """
        Analyzes imports and provides a summary.

        :param code_str: The code string to analyze.
        :return: A tuple containing the import information dictionary and a summary string.
        """
        import_info = CodeAnalyzer.extract_imports(code_str)
        
        summary = [
            f"Standard library imports: {len(import_info['standard_imports'])}",
            f"Third-party imports: {len(import_info['third_party_imports'])}",
            f"Local imports: {len(import_info['local_imports'])}",
            f"From imports: {len(import_info['from_imports'])}"
        ]
        
        if import_info['errors']:
            summary.append(f"Errors encountered: {len(import_info['errors'])}")
        
        return import_info, "\n".join(summary)

    @staticmethod
    def categorize_change(diff_text: str) -> List[str]:
        """
        Categorizes the type of code changes based on diff text.

        :param diff_text: The diff text to analyze.
        :return: A list of change categories.
        """
        categories = []
        patterns = {
            'function_added': r'^\+.*def\s+\w+\(',
            'function_removed': r'^-.*def\s+\w+\(',
            'class_added': r'^\+.*class\s+\w+\(',
            'class_removed': r'^-.*class\s+\w+\(',
            'import_added': r'^\+.*import\s+\w+',
            'import_removed': r'^-.*import\s+\w+'
        }

        for category, pattern in patterns.items():
            if re.search(pattern, diff_text, re.MULTILINE):
                categories.append(category)

        if not categories:
            categories.append('general_change')

        return categories