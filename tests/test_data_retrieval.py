import os
os.chdir('/Users/xujiantong/Library/CloudStorage/OneDrive-KERINGSA/Code_base/python/ML_POC/llama-github')

import unittest
from unittest.mock import patch, Mock
from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import Repository

# Mock data for testing
MOCK_REPO_DATA = {
    'id': 123,
    'name': 'mock-repo',
    'full_name': 'octocat/mock-repo',
    'description': 'This is a mock repository',
    'html_url': 'https://github.com/octocat/mock-repo',
    'stargazers_count': 10,
    'watchers_count': 5,
    'language': 'Python',
    'forks_count': 2
}

class TestGitHubAPIHandler(unittest.TestCase):
    @patch('github_api.Github')
    def test_search_repositories(self, mock_github):
        # Setup mock response
        mock_instance = mock_github.return_value
        mock_instance.search_repositories.return_value = [Mock(**MOCK_REPO_DATA)]

        # Initialize GitHubAPIHandler and call search_repositories
        api_handler = GitHubAPIHandler('test_token')
        repositories = api_handler.search_repositories('mock-query')

        # Assertions
        self.assertEqual(len(repositories), 1)
        self.assertIsInstance(repositories[0], Repository)
        self.assertEqual(repositories[0].name, 'mock-repo')

    @patch('github_api.Github')
    def test_get_repository(self, mock_github):
        # Setup mock response
        mock_instance = mock_github.return_value
        mock_instance.get_repo.return_value = Mock(**MOCK_REPO_DATA)

        # Initialize GitHubAPIHandler and call get_repository
        api_handler = GitHubAPIHandler('test_token')
        repository = api_handler.get_repository('octocat/mock-repo')

        # Assertions
        self.assertIsInstance(repository, Repository)
        self.assertEqual(repository.full_name, 'octocat/mock-repo')

class TestRepository(unittest.TestCase):
    @patch('github_entities.Github')
    def test_get_readme(self, mock_github):
        # Setup mock response
        mock_instance = mock_github.return_value
        mock_repo = mock_instance.get_repo.return_value
        mock_repo.get_readme.return_value = Mock(decoded_content=b'This is a README')

        # Initialize Repository and call get_readme
        repository = Repository(**MOCK_REPO_DATA, github_instance=mock_instance)
        readme_content = repository.get_readme()

        # Assertions
        self.assertEqual(readme_content, 'This is a README')

    @patch('github_entities.Github')
    def test_get_structure(self, mock_github):
        # Setup mock response
        mock_instance = mock_github.return_value
        mock_repo = mock_instance.get_repo.return_value
        mock_repo.get_contents.side_effect = [
            [Mock(type='dir', name='subdir', path='subdir'), Mock(type='file', name='file.txt', path='file.txt')],
            [Mock(type='file', name='nested.txt', path='subdir/nested.txt')]
        ]

        # Initialize Repository and call get_structure
        repository = Repository(**MOCK_REPO_DATA, github_instance=mock_instance)
        structure = repository.get_structure()

        # Assertions
        self.assertIn('subdir', structure)
        self.assertIn('file.txt', structure)
        self.assertIn('nested.txt', structure['subdir'])

    @patch('github_entities.Github')
    def test_get_file_content(self, mock_github):
        # Setup mock response
        mock_instance = mock_github.return_value
        mock_repo = mock_instance.get_repo.return_value
        mock_repo.get_contents.return_value = Mock(decoded_content=b'print("Hello, World!")')

        # Initialize Repository and call get_file_content
        repository = Repository(**MOCK_REPO_DATA, github_instance=mock_instance)
        file_content = repository.get_file_content('hello.py')

        # Assertions
        self.assertEqual(file_content, 'print("Hello, World!")')

if __name__ == '__main__':
    unittest.main()
