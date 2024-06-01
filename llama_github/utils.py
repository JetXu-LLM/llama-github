import hashlib
import base64
import re
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from llama_github.logger import logger

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
