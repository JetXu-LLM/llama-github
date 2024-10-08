from github import GithubException
from .github_entities import Repository, RepositoryPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_github.logger import logger
from llama_github.github_integration.github_auth_manager import ExtendedGithub
from llama_github.config.config import config
import re
from typing import Any, Dict, List


class GitHubAPIHandler:
    def __init__(self, github_instance: ExtendedGithub):
        """
        Initializes the GitHubAPIHandler with a GitHub instance.

        :param github_instance: Authenticated instance of a Github client.
        """
        self._github = github_instance
        self.pool = RepositoryPool(github_instance)

    def search_repositories(self, query, sort="best match", order="desc"):
        """
        Searches for repositories on GitHub based on a query.

        :param query: The search query string.
        :param sort: The field to sort the results by. Default is 'stars'.
        :param order: The order of sorting, 'asc' or 'desc'. Default is 'desc'.
        :return: A list of Repository objects or None if an error occurs.
        """
        try:
            if sort not in ['stars', 'forks', 'updated']:
                repositories = self._github.search_repositories(
                    query=query, order=order)
            else:
                repositories = self._github.search_repositories(
                    query=query, sort=sort, order=order)
            result = []
            for i, repo in enumerate(repositories):
                if i >= config.get("repo_search_max_hits"):
                    break
                result.append(
                    Repository(
                        repo.full_name,
                        self._github,
                        **{
                            'id': repo.id,
                            'name': repo.name,
                            'description': repo.description,
                            'html_url': repo.html_url,
                            'stargazers_count': repo.stargazers_count,
                            'language': repo.language,
                            'default_branch': repo.default_branch,
                            'updated_at': repo.updated_at,
                        }
                    )
                )
            return result
        except GithubException as e:
            logger.exception(
                f"Error searching repositories with query '{query}':")
            return None

    def get_repository(self, full_repo_name):
        """
        Retrieves a single repository by its full name.

        :param full_repo_name: The full name of the repository (e.g., 'octocat/Hello-World').
        :return: A Repository object or None if an error occurs.
        """
        return self.pool.get_repository(full_repo_name)

    def _get_file_content_through_repository(self, code_search_result):
        """
        Helper method to get file content through a Repository object.

        :param code_search_result: A single code search result.
        :return: Tuple containing the Repository object and the file content.
        """
        # Assuming RepositoryPool is accessible and initialized somewhere in this class
        repository_obj = self.get_repository(
            code_search_result['repository']['full_name'])
        file_content = repository_obj.get_file_content(
            code_search_result['path'])
        return repository_obj, file_content
    
    async def get_pr_files(self, repo: Repository, pr_number: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/repos/{repo.full_name}/pulls/{pr_number}/files"
        headers = {"Authorization": f"token {self.token}"}
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to get PR files: {response.status}")
                return []

    async def get_pr_comments(self, repo: Repository, pr_number: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/repos/{repo.full_name}/issues/{pr_number}/comments"
        headers = {"Authorization": f"token {self.token}"}
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to get PR comments: {response.status}")
                return []

    def search_code(self, query, repo_full_name=None):
        """
        Searches for code on GitHub based on a query, optionally within a specific repository.

        :param query: The search query string.
        :param repo_full_name: Optional. The full name of the repository (e.g., 'octocat/Hello-World') to restrict the search to.
        :return: A list of code search results or None if an error occurs.
        """
        try:
            logger.debug(f"Searching code with query '{query}'...")
            # If a repository full name is provided, include it in the query
            if repo_full_name:
                query = f"{query} repo:{repo_full_name}"

            # Perform the search
            code_results = self._github.search_code(
                query=query, per_page=config.get("code_search_max_hits"))

            results_with_index = []
            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                # Concurrently fetch the file content for each code search result
                future_to_index = {executor.submit(
                    self._get_file_content_through_repository, code_result): index for index, code_result in enumerate(code_results)}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    code_result = code_results[index]
                    try:
                        repository_obj, file_content = future.result()
                        if repository_obj and file_content:
                            results_with_index.append({
                                'index': index,
                                'name': code_result['name'],
                                'path': code_result['path'],
                                'repository_full_name': code_result['repository']['full_name'],
                                'url': code_result['html_url'],
                                'content': file_content,
                                'stargazers_count': repository_obj.stargazers_count,
                                'watchers_count': repository_obj.watchers_count,
                                'language': repository_obj.language,
                                'description': repository_obj.description,
                                'updated_at': repository_obj.updated_at,
                            })
                    except Exception as e:
                        logger.exception(
                            f"{code_result['name']} generated an exception:")

            # Sort the results by index to maintain the original order
            sorted_results = sorted(
                results_with_index, key=lambda x: x['index'])
            logger.debug(
                f"Code search retrieved successfully with {len(sorted_results)} results.")
            return sorted_results
        except GithubException as e:
            logger.exception(f"Error searching code with query '{query}':")
            return None

    def _get_issue_content_through_repository(self, issue):
        """
        Helper method to get issue content through issue url.

        :param code_result: A single code search result.
        :return: Tuple containing the Repository object and the file content.
        """
        # Assuming RepositoryPool is accessible and initialized somewhere in this clas
        issue_content = ''
        issue_url = issue['url']
        # Use regular expressions to extract repo_full_name and issue_number
        match = re.search(
            r'https://api.github.com/repos/([^/]+/[^/]+)/issues/(\d+)', issue_url)
        if match:
            repo_full_name = match.group(1)
            issue_number = int(match.group(2))
            repository_obj = self.get_repository(repo_full_name)
            issue_content = repository_obj.get_issue_content(
                number=issue_number, issue=issue)
        else:
            logger.warning(
                f"Failed to extract repo_full_name and issue_number from issue url: {issue_url}")
        return issue_content

    def search_issues(self, query, repo_full_name=None):
        """
        Searches for issues on GitHub based on a query, optionally within a specific repository.

        :param query: The search query string.
        :param repo_full_name: Optional. The full name of the repository (e.g., 'octocat/Hello-World') to restrict the search to.
        :return: A list of issue search results or None if an error occurs.
        """
        try:
            logger.debug(f"Searching issue with query '{query}'...")
            # If a repository full name is provided, include it in the query
            if repo_full_name:
                query = f"{query} repo:{repo_full_name}"

            # Perform the search
            issue_results = self._github.search_issues(
                query=query, per_page=config.get("issue_search_max_hits"))

            issue_results = [issue for issue in issue_results if issue['body']
                             is not None and issue['body'] != 'null']

            results_with_index = []
            with ThreadPoolExecutor(max_workers=config.get("max_workers")) as executor:
                # Concurrently fetch the issue content for each issue search result
                future_to_index = {executor.submit(
                    self._get_issue_content_through_repository, issue): index for index, issue in enumerate(issue_results)}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    issue_result = issue_results[index]
                    try:
                        issue_content = future.result()
                        results_with_index.append({
                            'index': index,
                            'url': issue_result['url'],
                            'created_at': issue_result['created_at'],
                            'updated_at': issue_result['updated_at'],
                            'issue_content': issue_content,
                        })
                    except Exception as e:
                        logger.exception(
                            f"{issue_result['url']} generated an exception:")

            # Sort the results by index to maintain the original order
            sorted_results = sorted(
                results_with_index, key=lambda x: x['index'])
            logger.debug(
                f"Issue search retrieved successfully with {len(sorted_results)} results.")
            return sorted_results
        except GithubException as e:
            logger.exception(f"Error searching issue with query '{query}':")
            return None

    def _categorize_github_url(url):
        repo_pattern = r'^https://github\.com/[^/]+/[^/]+$'
        issue_pattern = r'^https://github\.com/[^/]+/[^/]+/issues/\d+$'
        repo_file_pattern = r'^https://github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/.+$'
        readme_pattern = r'^https://github\.com/[^/]+/[^/]+#readme$'

        if re.match(repo_pattern, url):
            return "repo"
        elif re.match(issue_pattern, url):
            return "issue"
        elif re.match(repo_file_pattern, url):
            return "file"
        elif re.match(readme_pattern, url):
            return "readme"
        else:
            return "other"
    
    def get_github_url_content(self, url):
        """
        Retrieves the content of a GitHub URL.

        :param url: The GitHub URL to retrieve content from.
        :return: The content of the URL or None if an error occurs.
        """
        try:
            logger.debug(f"Retrieving content from GitHub URL '{url}'...")
            content = None
            category = GitHubAPIHandler._categorize_github_url(url)
            if category == "repo":
                # Extract the repository full name from the URL
                match = re.search(r'https://github\.com/([^/]+/[^/]+)', url)
                if match:
                    repo_full_name = match.group(1)
                    repository_obj = self.get_repository(repo_full_name)
                    content = repository_obj.get_readme()
                else:
                    logger.warning(
                        f"Failed to extract repository full name from URL: {url}")
            elif category == "issue":
                # Use regular expressions to extract repo_full_name and issue_number
                match = re.search(
                    r'https://github\.com/([^/]+/[^/]+)/issues/(\d+)', url)
                if match:
                    repo_full_name = match.group(1)
                    issue_number = int(match.group(2))
                    repository_obj = self.get_repository(repo_full_name)
                    content = repository_obj.get_issue_content(
                        number=issue_number)
                else:
                    logger.warning(
                        f"Failed to extract repo_full_name and issue_number from URL: {url}")
            elif category == "file":
                # Extract the repository full name and file path from the URL
                match = re.search(
                    r'https://github\.com/([^/]+/[^/]+)/(?:blob|tree)/([^/]+)/(.+)', url)
                if match:
                    repo_full_name = match.group(1)
                    file_path = match.group(3)
                    repository_obj = self.get_repository(repo_full_name)
                    content = repository_obj.get_file_content(
                        file_path)
                else:
                    logger.warning(
                        f"Failed to extract repository full name and file path from URL: {url}")
            elif category == "readme":
                # Extract the repository full name from the URL
                match = re.search(r'https://github\.com/([^/]+/[^/]+)#readme', url)
                if match:
                    repo_full_name = match.group(1)
                    repository_obj = self.get_repository(repo_full_name)
                    content = repository_obj.get_readme()
                else:
                    logger.warning(
                        f"Failed to extract repository full name from URL: {url}")
            else:
                logger.warning(f"Unsupported GitHub URL category: {category}")
            return content
        except GithubException as e:
            logger.exception(f"Error retrieving content from GitHub URL '{url}':")
            return None