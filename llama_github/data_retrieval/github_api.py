from github import Github, GithubException
from .github_entities import Repository, RepositoryPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_github.logger import logger

class GitHubAPIHandler:
    def __init__(self, github_instance):
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
                repositories = self._github.search_repositories(query=query, order=order)
            else:
                repositories = self._github.search_repositories(query=query, sort=sort, order=order)
            
            return [
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
                    }
                ) for repo in repositories[:50]
            ]
        except GithubException as e:
            logger.exception(f"Error searching repositories with query '{query}':")
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

        :param code_result: A single code search result.
        :return: Tuple containing the Repository object and the file content.
        """
        # Assuming RepositoryPool is accessible and initialized somewhere in this class
        repository_obj = self.get_repository(code_search_result.repository.full_name)
        file_content = repository_obj.get_file_content(code_search_result.path)
        return repository_obj, file_content
        
    def search_code(self, query, repo_full_name=None):
        """
        Searches for code on GitHub based on a query, optionally within a specific repository.

        :param query: The search query string.
        :param repo_full_name: Optional. The full name of the repository (e.g., 'octocat/Hello-World') to restrict the search to.
        :param sort: The field to sort the results by. Default is 'indexed'.
        :param order: The order of sorting, 'asc' or 'desc'. Default is 'desc'.
        :return: A list of code search results or None if an error occurs.
        """
        try:
            # If a repository full name is provided, include it in the query
            if repo_full_name:
                query = f"{query} repo:{repo_full_name}"
            
            # Perform the search
            code_results = self._github.search_code(query=query)
            code_results = list(code_results[:20]) # Limit to the first 40 results
            
            results_with_index = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Concurrently fetch the file content for each code search result
                future_to_index = {executor.submit(self._get_file_content_through_repository, code_result): index for index, code_result in enumerate(code_results)}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    code_result = code_results[index]
                    try:
                        repository_obj, file_content = future.result()
                        if repository_obj and file_content:
                            results_with_index.append({
                                'index': index,
                                'data': {
                                    'name': code_result.name,
                                    'path': code_result.path,
                                    'repository_full_name': code_result.repository.full_name,
                                    'url': code_result.url,
                                    'content': file_content,
                                    'stargazers_count': repository_obj.stargazers_count,
                                    'watchers_count': repository_obj.watchers_count,
                                    'language': repository_obj.language,
                                    'description': repository_obj.description,
                                }
                            })
                    except Exception as e:
                        logger.exception(f"{code_result.name} generated an exception:")

            # Sort the results by index to maintain the original order
            sorted_results = sorted(results_with_index, key=lambda x: x['index'])
            # Extract the data from the sorted results
            final_results = [item['data'] for item in sorted_results]

            return final_results
        except GithubException as e:
            logger.exception(f"Error searching code with query '{query}':")
            return None

