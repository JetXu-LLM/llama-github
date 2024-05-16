#To do list:
#1. Add issues to the Repository class.
#2. Add a method to get issues for a repository.

from github import GithubException
from threading import Lock, Event, Thread
from datetime import datetime, timezone
import time
from llama_github.logger import logger
from llama_github.github_integration.github_auth_manager import ExtendedGithub

class Repository:
    def __init__(self, full_name, github_instance: ExtendedGithub, **kwargs):
        """
        Initializes a Repository instance with details and a GitHub instance for API calls.

        :param id: The ID of the repository.
        :param name: The name of the repository.
        :param full_name: The full name of the repository (e.g., 'octocat/Hello-World').
        :param description: The description of the repository.
        :param html_url: The HTML URL of the repository.
        :param stargazers_count: The count of stargazers of the repository.
        :param watchers_count: The count of watchers of the repository.
        :param language: The primary language of the repository.
        :param forks_count: The count of forks of the repository.
        :param github_instance: Authenticated instance of a Github client.
        """

        self._github = github_instance
        self.full_name = full_name

        self.id = kwargs.get("id")
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.html_url =  kwargs.get("html_url")
        self.stargazers_count =  kwargs.get("stargazers_count")
        self.language =  kwargs.get("language")
        self.default_branch = kwargs.get("default_branch")
        self.watchers_count = None

        self.creation_time = datetime.now(timezone.utc)
        self.last_read_time = datetime.now(timezone.utc)

        self.repo = None

        # Locking for write action
        self._structure_lock = Lock()
        self._file_contents_lock = Lock()
        self._readme_lock = Lock()
        self._repo_lock = Lock()

        if (self.id is None) or \
            (self.name is None) or \
            (self.full_name is None) or \
            (self.html_url is None) or \
            (self.stargazers_count is None) or \
            (self.default_branch is None):
            self.get_repo()
        
        self._structure = None  # Singleton pattern for repository structure
        self._file_contents = {}  # Singleton pattern for file contents
        self._readme = None  # Singleton pattern for README content
        
    def update_last_read_time(self):
        self.last_read_time = datetime.now(timezone.utc)

    def get_readme(self):
        """
        Retrieves the README content of the repository using a singleton design pattern.
        """
        if self._readme is None:  # Check if README content has already been fetched
            with self._readme_lock:  # Locking for write action
                if self._readme is None: # Check if README content has already been fetched after get lock
                    try:
                        readme = self.repo.get_readme()
                        self._readme = readme.decoded_content.decode("utf-8")
                    except GithubException as e:
                        logger.exception(f"Error getting README for repository {self.full_name}:")
                        self._readme = None
        self.update_last_read_time()
        return self._readme
    
    def get_repo(self):
        """
        Retrieves the Github Repo object of the repository using a singleton design pattern.
        """
        if self.repo is None:  # Check if repo object has already been fetched
            with self._repo_lock:  # Locking for write action
                if self.repo is None:
                    try:
                        self.repo = self._github.get_repo(self.full_name)
                        self.id = self.repo.id
                        self.name = self.repo.name
                        self.description = self.repo.description
                        self.html_url = self.repo.html_url
                        self.stargazers_count = self.repo.stargazers_count
                        self.watchers_count = self.repo.subscribers_count
                        self.language = self.repo.language
                        self.default_branch = self.repo.default_branch
                    except GithubException as e:
                        logger.exception(f"Error retrieving repository '{self.full_name}':")
                        return None
        self.update_last_read_time()
        return self.repo

    def get_structure(self, path="/"):
        """
        Retrieves the structure of the repository using a singleton design pattern.
        """
        if self._structure is None:  # Check if structure has already been fetched
            with self._structure_lock:  # Locking for write action
                if self._structure is None:  # Check if structure has already been fetched after get lock
                    try:
                        self._structure = self._github.get_repo_structure(self.full_name, self.default_branch)
                    except GithubException as e:
                        logger.exception(f"Error getting structure for repository {self.full_name}:")
                        self._structure = None

        self.update_last_read_time()
        return self._structure

    def get_file_content(self, file_path):
        """
        Retrieves the content of a file using a singleton design pattern.
        """
        if file_path not in self._file_contents:  # Check if file content has already been fetched
            with self._file_contents_lock:  # Locking for write action
                if file_path not in self._file_contents:  # Check if file content has already been fetched after get lock
                    try:
                        file_content = self.repo.get_contents(file_path)
                        self._file_contents[file_path] = file_content.decoded_content.decode("utf-8")
                    except GithubException as e:
                        logger.exception(f"Error getting file content for {file_path} in repository {self.full_name}:")
                        return None
        self.update_last_read_time()
        return self._file_contents[file_path]

    def __str__(self):
        """
        String representation of the Repository instance.

        :return: A string describing the repository.
        """
        self.update_last_read_time()
        return f"{self.full_name} - {self.description}"
    
    def clear_cache(self):
        """
        Clears the cached data of the repository, including structure, file contents, and README content.
        """
        with self._structure_lock:
            self._structure = None  # Reset the repository structure cache
        with self._file_contents_lock:
            self._file_contents = {}  # Reset the file contents cache
        with self._readme_lock:
            self._readme = None  # Reset the README content cache
        with self._repo_lock:
            self.repo = None  # Reset the Repo cache
    
class RepositoryPool:
    _instance_lock = Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # First check (unlocked)
            with cls._instance_lock:  # Acquire lock
                if cls._instance is None:  # Second check (locked)
                    cls._instance = super(RepositoryPool, cls).__new__(cls)
        return cls._instance

    def __init__(self, github_instance, cleanup_interval=3600, max_idle_time=86400):
        if not hasattr(self, 'initialized'): 
            with self._instance_lock:   # Prevent re-initialization
                if not hasattr(self, 'initialized'):
                    self.initialized = True
                    self._locks_registry = {}  # A registry for repository-specific locks
                    self._registry_lock = Lock()  # A lock to protect the locks registry
                    self._pool = {}  # The repository pool
                    self.github_instance = github_instance
                    self.cleanup_interval = cleanup_interval  # How often to run cleanup in seconds
                    self.max_idle_time = max_idle_time  # Maximum idle time in seconds
                    self._cleanup_thread = Thread(target=self._cleanup, daemon=True)
                    self._stop_event = Event()
                    self._cleanup_thread.start()
    
    def _cleanup(self): # Internal method for cleaning up idle repository objects' cache content, not real delete repository objects
        """Periodically checks and removes idle repository objects."""
        while not self._stop_event.is_set():
            with self._registry_lock:
                current_time = datetime.now(timezone.utc)
                for full_name in list(self._pool.keys()):
                    repo = self._pool[full_name]
                    if ((current_time - repo.last_read_time).total_seconds() > self.max_idle_time) or \
                        ((current_time - repo.creation_time).total_seconds() > 7 * self.max_idle_time and \
                        (current_time - repo.last_read_time).total_seconds() > (self.max_idle_time // 4) + 1):
                        del self._locks_registry[full_name]  # release the lock due to object already created
                        self._pool[full_name].clear_cache() # clear the cache content of repository object
            time.sleep(self.cleanup_interval)

    def stop_cleanup(self):
        """Stops the cleanup thread."""
        self._stop_event.set()

    def _get_repo_lock(self, full_name):
        """Retrieve or create a lock for a specific repository."""
        with self._registry_lock:
            if full_name not in self._locks_registry:
                self._locks_registry[full_name] = Lock() # Create a new lock for the repository creation
            return self._locks_registry[full_name]

    def get_repository(self, full_name, **kwargs):
        """Retrieve a repository from the pool or create a new one if it doesn't exist."""
        if full_name in self._pool:
            return self._pool[full_name]
        repo_lock = self._get_repo_lock(full_name)
        with repo_lock:
            if full_name not in self._pool:
                self._pool[full_name] = Repository(full_name, self.github_instance, **kwargs)
        return self._pool[full_name]
    
