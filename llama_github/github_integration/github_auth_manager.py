# To do list:
# 1. add the mechanism for installation_access_token and github_instance refreshment in authenticate_with_app model
# 2. add re-try mechanism for the API calls
# 3. add the mechanism for the rate limit handling
# 4. add the mechanism for the error handling
# 5. add the mechanism for the logging
# 6. add search issues functionality
# 7. add search discussions functionality through Github GraphQL API

from github import Github, GithubIntegration
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException
from urllib3.util.retry import Retry
from llama_github.logger import logger


class GitHubAuthManager:
    def __init__(self):
        self.github_instance = None
        self.access_token = None
        self.app_id = None
        self.private_key = None
        self.installation_id = None

    def authenticate_with_token(self, access_token):
        """
        Authenticate using a personal access token or an OAuth token.
        Suitable for individual developers and applications using OAuth for authorization.
        """
        self.access_token = access_token
        self.github_instance = ExtendedGithub(login_or_token=access_token)
        return self.github_instance

    def authenticate_with_app(self, app_id, private_key, installation_id):
        """
        Authenticate using a GitHub App.
        Suitable for integrations in organizational or enterprise environments.
        """
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        integration = GithubIntegration(app_id, private_key)
        installation_access_token = integration.get_access_token(
            installation_id).token
        self.access_token = installation_access_token
        self.github_instance = ExtendedGithub(
            login_or_token=installation_access_token)
        return self.github_instance

    def close_connection(self):
        """
        Close the connection to GitHub to free up resources.
        """
        if self.github_instance:
            self.github_instance = None

# Extended Github Class for powerful API calls - e.g. recursive call to get repo structure


class ExtendedGithub(Github):
    def __init__(self, login_or_token):
        self.access_token = login_or_token
        super().__init__(login_or_token=login_or_token)

    def get_repo_structure(self, repo_full_name, branch='main') -> dict:
        """
        Get the structure of a repository (files and directories) recursively.
        """
        owner, repo_name = repo_full_name.split('/')
        headers = {'Authorization': f'token {self.access_token}'}

        # Function to convert the flat list to a hierarchical structure
        def list_to_tree(items):
            """
            Convert the flat list to a hierarchical structure with full paths.
            Include size metadata for files and remove 'type' attributes.
            """
            tree = {}
            for item in items:
                path_parts = item['path'].split('/')
                current_level = tree
                for part in path_parts[:-1]:
                    # Ensure 'children' dictionary exists for directories without explicitly adding 'type'
                    current_level = current_level.setdefault(
                        part, {'children': {}})
                    # Ensure we don't inadvertently create a 'type' key for directories
                    current_level = current_level.get('children')

                # For the last part of the path, decide if it's a file or directory and add appropriate information
                if item['type'] == 'blob':  # It's a file
                    current_level[path_parts[-1]] = {
                        'path': item['path'],  # Include full path
                        # Include size if available
                        'size': item.get('size', 0)
                    }
                else:  # It's a directory
                    # Initialize the directory if not already present, without adding 'type'
                    if path_parts[-1] not in current_level:
                        current_level[path_parts[-1]] = {'children': {}}
            return tree

        # Directly use the Trees API to get the full directory structure of the "main" branch
        tree_url = f'https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1'
        tree_response = requests.get(tree_url, headers=headers)

        # Check if the request was successful
        if tree_response.status_code == 200:
            tree_data = tree_response.json()
            # Convert the flat list of items to a hierarchical tree structure
            repo_structure = list_to_tree(tree_data['tree'])
            return repo_structure
        else:
            print(
                f"Error fetching tree structure: {tree_response.status_code}")
            print("Details:", tree_response.json())

    def search_code(self, query: str, per_page: int = 30) -> dict:
        """
        Search for code on GitHub using the GitHub API.

        Parameters:
        query (str): The search query.
        per_page (int): The number of results per page.

        Returns:
        dict: The search result in dict format.
        """
        url = 'https://api.github.com/search/code'
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.access_token}'
        }
        params = {
            'q': query,
            'per_page': per_page
        }

        # Retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            # Retry on these HTTP status codes
            status_forcelist=[429, 500, 502, 503, 504],
            # Retry on these HTTP methods
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1  # Exponential backoff factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        try:
            response = http.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json().get('items', [])
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")

    def search_issues(self, query: str, per_page: int = 30) -> dict:
        """
        Search for code on GitHub using the GitHub API.

        Parameters:
        query (str): The search query.
        per_page (int): The number of results per page.

        Returns:
        dict: The search result in dict format.
        """
        url = 'https://api.github.com/search/issues'
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.access_token}'
        }
        params = {
            'q': query,
            'per_page': per_page
        }

        # Retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            # Retry on these HTTP status codes
            status_forcelist=[429, 500, 502, 503, 504],
            # Retry on these HTTP methods
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1  # Exponential backoff factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        try:
            response = http.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json().get('items', [])
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")

    def get_issue_comments(self, repo_full_name: str, issue_number: int) -> dict:
        """
        Get comments of an issue on GitHub using the GitHub API.

        Parameters:
        repo_full_name (str): The full name of the repository (e.g., 'octocat/Hello-World').
        issue_number (int): The issue number.

        Returns:
        dict: The comments of the issue in dict format.
        """
        url = f'https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments'
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.access_token}'
        }
        # Retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            # Retry on these HTTP status codes
            status_forcelist=[429, 500, 502, 503, 504],
            # Retry on these HTTP methods
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1  # Exponential backoff factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        try:
            response = http.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json()
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")

    def get_pr_files(self, repo_full_name: str, pr_number: int) -> list:
        """
        Get the files of a pull request on GitHub using the GitHub API.

        Parameters:
        repo_full_name (str): The full name of the repository (e.g., 'octocat/Hello-World').
        pr_number (int): The pull request number.

        Returns:
        list: The files of the pull request in list format.
        """
        url = f'https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files'
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.access_token}'
        }

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        try:
            response = http.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")
        return []

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list:
        """
        Get the comments of a pull request on GitHub using the GitHub API.

        Parameters:
        repo_full_name (str): The full name of the repository (e.g., 'octocat/Hello-World').
        pr_number (int): The pull request number.

        Returns:
        list: The comments of the pull request in list format.
        """
        url = f'https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments'
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.access_token}'
        }

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        try:
            response = http.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
        except Exception as err:
            logger.error(f"An error occurred: {err}")
        return []

# Example usage:
if __name__ == "__main__":
    auth_manager = GitHubAuthManager()

    # For developers using a personal access token or an OAuth token
    github_instance = auth_manager.authenticate_with_token(
        "your_personal_access_token_or_oauth_token_here")

    # For organizational or enterprise environments using GitHub App
    # github_instance = auth_manager.authenticate_with_app("app_id", "private_key", "installation_id")

    # Example action: List all repositories for the authenticated user
    if github_instance:
        for repo in github_instance.get_user().get_repos():
            print(repo.name)

    # Close the connection when done
    auth_manager.close_connection()
