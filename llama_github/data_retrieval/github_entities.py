from github import GithubException
from threading import Lock, Event, Thread
from datetime import datetime, timezone

from typing import Optional, Dict, Any, List
from llama_github.logger import logger
from llama_github.github_integration.github_auth_manager import ExtendedGithub
from llama_github.utils import DiffGenerator

import time
import re
import json
from dateutil import parser
from datetime import timezone
import base64

from llama_github.utils import DiffGenerator, CodeAnalyzer

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
        self.html_url = kwargs.get("html_url")
        self.stargazers_count = kwargs.get("stargazers_count")
        self.language = kwargs.get("language")
        self.default_branch = kwargs.get("default_branch")
        self.updated_at = kwargs.get("updated_at")
        self.watchers_count = None

        self.creation_time = datetime.now(timezone.utc)
        self.last_read_time = datetime.now(timezone.utc)

        self._repo = None

        # Locking for write action
        self._structure_lock = Lock()
        self._file_contents_lock = Lock()
        self._issue_lock = Lock()
        self._pr_lock = Lock()
        self._readme_lock = Lock()
        self._repo_lock = Lock()

        if (self.id is None) or \
            (self.name is None) or \
            (self.full_name is None) or \
            (self.html_url is None) or \
            (self.stargazers_count is None) or \
            (self.default_branch is None) or \
                (self.updated_at is None):
            self.get_repo()

        self._structure = None  # Singleton pattern for repository structure
        self._file_contents = {}  # Singleton pattern for file contents
        self._issues = {}  # Singleton pattern for file contents
        self._readme = None  # Singleton pattern for README content
        self._prs = {}  # Singleton pattern for PR content

    def update_last_read_time(self):
        self.last_read_time = datetime.now(timezone.utc)

    def get_readme(self) -> str:
        """
        Retrieves the README content of the repository using a singleton design pattern.
        """
        if self._readme is None:  # Check if README content has already been fetched
            with self._readme_lock:  # Locking for write action
                if self._readme is None:  # Check if README content has already been fetched after get lock
                    try:
                        readme = self.repo.get_readme()
                        self._readme = readme.decoded_content.decode("utf-8")
                    except GithubException as e:
                        logger.exception(
                            f"Error getting README for repository {self.full_name}:")
                        self._readme = None
        self.update_last_read_time()
        return self._readme

    @property
    def repo(self):
        return self.get_repo()

    def get_repo(self):
        """
        Retrieves the Github Repo object of the repository using a singleton design pattern.
        """
        if self._repo is None:  # Check if repo object has already been fetched
            with self._repo_lock:  # Locking for write action
                if self._repo is None:
                    try:
                        self._repo = self._github.get_repo(self.full_name)
                        self.id = self._repo.id
                        self.name = self._repo.name
                        self.description = self._repo.description
                        self.html_url = self._repo.html_url
                        self.stargazers_count = self._repo.stargazers_count
                        self.watchers_count = self._repo.subscribers_count
                        self.language = self._repo.language
                        self.default_branch = self._repo.default_branch
                        self.updated_at = self._repo.updated_at
                    except GithubException as e:
                        logger.exception(
                            f"Error retrieving repository '{self.full_name}':")
                        return None
        self.update_last_read_time()
        return self._repo

    def get_structure(self, path="/") -> dict:
        """
        Retrieves the structure of the repository using a singleton design pattern.
        """
        if self._structure is None:  # Check if structure has already been fetched
            with self._structure_lock:  # Locking for write action
                if self._structure is None:  # Check if structure has already been fetched after get lock
                    try:
                        self._structure = self._github.get_repo_structure(
                            self.full_name, self.default_branch)
                    except GithubException as e:
                        logger.exception(
                            f"Error getting structure for repository {self.full_name}:")
                        self._structure = None
        self.update_last_read_time()
        return self._structure
    
    def get_file_content(self, file_path: str, sha: Optional[str] = None) -> Optional[str]:
        """
        Retrieves the content of a file using a singleton design pattern with improved encoding handling.

        This method first checks if the file content is already cached. If not, it fetches the content
        from the GitHub repository, handles potential encoding issues, and caches the result.

        :param file_path: The path to the file in the repository.
        :param sha: The commit SHA. If None, the latest version is fetched.
        :return: The file content as a string or None if not found or on error.
        """
        file_key = f"{file_path}/{sha}" if sha is not None else file_path

        if file_key not in self._file_contents:  # Check if file content has already been fetched
            with self._file_contents_lock:  # Locking for thread-safe write action
                if file_key not in self._file_contents:  # Double-check after acquiring the lock
                    try:
                        if sha is not None:
                            file_content = self.repo.get_contents(file_path, ref=sha)
                        else:
                            file_content = self.repo.get_contents(file_path)
                        
                        # Improved encoding handling
                        if file_content.encoding == 'base64':
                            decoded_content = base64.b64decode(file_content.content).decode('utf-8')
                        else:
                            decoded_content = file_content.decoded_content.decode('utf-8')
                        
                        self._file_contents[file_key] = decoded_content
                    except GithubException as e:
                        logger.exception(
                            f"Error getting file content for {file_key} in repository {self.full_name}:")
                        return None
                    except UnicodeDecodeError as e:
                        logger.exception(
                            f"Error decoding file content for {file_key} in repository {self.full_name}:")
                        return None

        self.update_last_read_time()
        return self._file_contents.get(file_key)
    
    def get_issue_content(self, number, issue=None) -> str:
        """
        Retrieves the content of a issue using a singleton design pattern.
        """
        if number not in self._issues:  # Check if issue has already been fetched
            with self._issue_lock:  # Locking for write action
                if number not in self._issues:  # Check if issue has already been fetched after get lock
                    try:
                        if issue is None:
                            issue = self.repo.get_issue(number=number)
                            comments_amount = issue.comments
                            body_content = "This is a Github Issue related to repo \"" + (self.full_name or "") + "\". Repo description:" + (self.description or "") +\
                                "\n\nIssue Title: "+issue.title+"\nIssue state: "+issue.state + \
                                "\nIssue last updated at: "+str(issue.updated_at) + \
                                "\nTotal favour: " + \
                                str(issue.reactions['total_count']) + \
                                "\nComment amount: " + \
                                str(comments_amount) + \
                                "\nIssue body:\n" + \
                                re.sub(r'\n+\s*\n+', '\n',
                                       issue.body.replace('\r', '\n').strip())
                        else:
                            comments_amount = issue['comments']
                            body_content = "This is a Github Issue related to repo \"" + (self.full_name or "") + "\". Repo description:" + (self.description or "") +\
                                "\n\nIssue title: "+issue['title']+"\nIssue state: "+issue['state'] + \
                                "\nIssue last updated at: "+str(issue['updated_at']) + \
                                "\nTotal favour: " + \
                                str(issue['reactions']['total_count']) + \
                                "\nComment amount: " + \
                                str(comments_amount) + \
                                "\nIssue body:\n" + \
                                re.sub(
                                    r'\n+\s*\n+', '\n', issue['body'].replace('\r', '\n').strip())
                        if comments_amount > 0:
                            comments = self._github.get_issue_comments(
                                repo_full_name=self.full_name, issue_number=number)
                            comments_text_list = []
                            for index, comment in enumerate(comments, start=1):
                                cleaned_body = re.sub(
                                    r'\n+\s*\n+', '\n', comment['body'].replace('\r', '\n').strip())
                                comment_text = (
                                    f"Comment No.{index} - comment author type: {comment['user']['type']}; "
                                    f"comment total favour: {comment['reactions']['total_count']}\n"
                                    f"Comment body:\n{cleaned_body}\n"
                                )
                                comments_text_list.append(comment_text)
                            comments_text = "\n".join(comments_text_list)
                            issue_content = body_content + "\n\nIssue Comments:\n" + comments_text
                        else:
                            issue_content = body_content
                        self._issues[number] = issue_content.strip()
                    except GithubException as e:
                        logger.exception(
                            f"Error getting issue content for {number} in repository {self.full_name}:")
                        return None
        self.update_last_read_time()
        return self._issues[number]
    
    def extract_related_issues(self, pr_data: Dict[str, Any]) -> List[int]:
        """
        Extracts related issue numbers from the PR description and other fields.

        :param pr_data: The pull request data dictionary.
        :return: A list of related issue numbers.
        """
        patterns = [
            rf'https://github\.com/{re.escape(self.full_name)}/issues/(\d+)',
            r'(?:^|\s)#(\d+)',
            r'(?:^|\s)(\d+)(?:\s|$)',
        ]
        issues = set()
        # Convert PR data to JSON string for pattern matching
        pr_description = json.dumps(pr_data, default=str)
        
        for pattern in patterns:
            matches = re.findall(pattern, pr_description)
            issues.update(int(match) for match in matches)
        
        return list(issues)

    def get_issue_contents(self, issue_numbers: List[int], pr_number: int) -> List[Dict[str, Any]]:
        """
        Retrieves contents for a list of issue numbers, excluding the PR number.

        :param issue_numbers: List of issue numbers.
        :param pr_number: The PR number to exclude.
        :return: A list of issue contents.
        """
        return [
            {
                "issue_number": issue_number,
                "issue_content": self.get_issue_content(issue_number)
            }
            for issue_number in issue_numbers
            if issue_number != pr_number
        ]
    
    def to_isoformat(self, dt: datetime) -> Optional[str]:
        """
        Converts a datetime object to ISO 8601 format with UTC timezone.

        :param dt: The datetime object to convert.
        :return: ISO 8601 formatted string or None if dt is None.
        """
        if dt:
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        return None

    def get_pr_content(self, number, pr=None, context_lines=10) -> Dict[str, Any]:
        """
        Retrieves and processes the content of a pull request.

        :param number: The PR number.
        :param pr: Optional PR object.
        :param context_lines: Number of context lines for diffs.
        :return: A dictionary containing detailed PR information.
        """
        if number not in self._prs:  # Check if issue has already been fetched
            with self._pr_lock:  # Locking for write action
                if number not in self._prs:  # Check if issue has already been fetched after get lock
                    try:
                        logger.debug(f"Processing PR #{number}")
                        if pr is None:
                            pr = self.repo.get_pull(number)
                        pr_files = self._github.get_pr_files(self.full_name, pr.number)
                        pr_comments = self._github.get_pr_comments(self.full_name, pr.number)

                        # Prepare PR metadata with standardized time formats
                        pr_data = {
                            "pr_metadata": {
                                "number": pr.number,
                                "title": pr.title,
                                "description": pr.body,
                                "author": pr.user.login,
                                "author_association": pr.raw_data['author_association'],
                                "created_at": self.to_isoformat(pr.created_at),
                                "updated_at": self.to_isoformat(pr.updated_at),
                                "merged_at": self.to_isoformat(pr.merged_at),
                                "state": pr.state,
                                "base_branch": pr.base.ref,
                                "head_branch": pr.head.ref,
                            },
                            "related_issues": [],
                            "file_changes": [],
                            "ci_cd_results": [],
                            "interactions": []
                        }

                        # Fetch CI/CD results
                        try:
                            last_commit = pr.get_commits().reversed[0]
                            statuses = last_commit.get_statuses()
                            ci_cd_results = {
                                "state": None,  # We'll set this later
                                "statuses": [],
                                "check_runs": []
                            }
                            
                            # Process statuses
                            for status in statuses:
                                if ci_cd_results["state"] is None:
                                    ci_cd_results["state"] = status.state
                                ci_cd_results["statuses"].append({
                                    "context": status.context,
                                    "state": status.state,
                                    "description": status.description,
                                    "target_url": status.target_url,
                                    "created_at": self.to_isoformat(status.created_at),
                                    "updated_at": self.to_isoformat(status.updated_at),
                                })

                            # Fetch check runs for detailed CI/CD outputs
                            check_runs = last_commit.get_check_runs()
                            for check_run in check_runs:
                                ci_cd_results["check_runs"].append({
                                    "name": check_run.name,
                                    "status": check_run.status,
                                    "conclusion": check_run.conclusion,
                                    "started_at": self.to_isoformat(check_run.started_at) if check_run.started_at else None,
                                    "completed_at": self.to_isoformat(check_run.completed_at) if check_run.completed_at else None,
                                    "details_url": check_run.html_url,  # Changed from details_url to html_url
                                })

                            pr_data["ci_cd_results"] = ci_cd_results
                        except GithubException as e:
                            logger.exception(f"Error fetching CI/CD results for PR #{number}")
                        except IndexError as e:
                            logger.exception(f"No commits found for PR #{number}")

                        # Combine comments and reviews, and sort by creation time
                        pr_interactions = []

                        # Process PR comments
                        for comment in pr_comments:
                            created_at_dt = parser.isoparse(comment["created_at"])
                            formatted_created_at = self.to_isoformat(created_at_dt)
                            pr_interactions.append({
                                "type": "pr_comment",
                                "author": comment["user"]["login"],
                                "author_association": comment["author_association"],
                                "content": comment["body"],
                                "created_at": formatted_created_at,
                                "comment_id": comment["id"]
                            })

                        # Process reviews and inline comments
                        reviews = pr.get_reviews()
                        review_comments = pr.get_review_comments()
                        for review in reviews:
                            pr_review = {
                                "type": "review",
                                "author": review.user.login,
                                "author_association": review.raw_data['author_association'],
                                "content": review.body,
                                "state": review.state,
                                "created_at": self.to_isoformat(review.submitted_at),
                                "comment_id": review.id
                            }
                            # Process inline comments with threading
                            for comment in review_comments:
                                if comment.pull_request_review_id == review.id:
                                    pr_review["type"] = "inline_comment"
                                    pr_review["content"] = comment.body
                                    pr_review["path"] = comment.path
                                    pr_review["diff_hunk"] = comment.diff_hunk
                            pr_interactions.append(pr_review)

                        # Sort interactions by creation time
                        pr_interactions.sort(key=lambda x: parser.isoparse(x["created_at"]))
                        pr_data["interactions"] = pr_interactions

                        # Extract related issues from PR description and comments
                        related_issues = self.extract_related_issues(pr_data)
                        pr_data['related_issues'] = self.get_issue_contents(related_issues, pr.number)

                        # Process file changes
                        dependency_files = ['requirements.txt', 'Pipfile', 'Pipfile.lock', 'setup.py']
                        config_files = ['.env', 'settings.py', 'config.yaml', 'config.yml', 'config.json']
                        for file in pr_files:
                            file_path = file['filename']
                            change_type = file['status']
                            language = file.get('language', self.language)
                            additions = file['additions']
                            deletions = file['deletions']
                            changes = file['changes']

                            # Fetch file content for base and head versions
                            if change_type == 'removed':
                                base_content = self.get_file_content(file_path=file_path, sha=pr.base.sha)
                                head_content = None
                            elif change_type == 'added':
                                base_content = None
                                head_content = self.get_file_content(file_path=file_path, sha=pr.head.sha)
                            else:
                                base_content = self.get_file_content(file_path=file_path, sha=pr.base.sha)
                                head_content = self.get_file_content(file_path=file_path, sha=pr.head.sha)

                            # Generate custom diff with specified context lines
                            custom_diff = DiffGenerator.generate_custom_diff(base_content, head_content, context_lines)

                            # Categorize code changes
                            change_categories = CodeAnalyzer.categorize_change(custom_diff)

                            # Extract imports from head content
                            related_modules = CodeAnalyzer.extract_imports(head_content) if language == 'Python' and head_content else []

                            # Build file change entry
                            file_change = {
                                "file_path": file_path,
                                "change_type": change_type,
                                "diff": custom_diff,
                                "language": language,
                                "additions": additions,
                                "deletions": deletions,
                                "changes": changes,
                                "change_categories": change_categories,
                                "related_modules": related_modules
                            }

                            # Check for dependency changes
                            if file_path in dependency_files:
                                pr_data.setdefault("dependency_changes", []).append({
                                    "file_path": file_path,
                                    "content": head_content
                                })

                            # Check for configuration changes
                            if file_path in config_files:
                                pr_data.setdefault("config_changes", []).append({
                                    "file_path": file_path,
                                    "content": head_content
                                })

                            pr_data['file_changes'].append(file_change)
                            logger.debug(f"Processed file: {file_path}")

                        self._prs[number] = pr_data
                        logger.debug(f"Collected details for PR #{number}: {pr.title}")
                    except GithubException as e:
                        logger.exception(f"Error getting PR content for #{number} in repository {self.full_name}")
        self.update_last_read_time()
        return self._prs[number]

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
        with self._issue_lock:
            self._issues = {}  # Reset the issue contents cache
        with self._readme_lock:
            self._readme = None  # Reset the README content cache
        with self._repo_lock:
            self._repo = None  # Reset the Repo cache
        with self._pr_lock:
            self._prs = {}  # Reset the PR cache


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
                    self._cleanup_thread = Thread(
                        target=self._cleanup, daemon=True)
                    self._stop_event = Event()
                    self._cleanup_thread.start()

    def _cleanup(self):  # Internal method for cleaning up idle repository objects' cache content, not real delete repository objects
        """Periodically checks and removes idle repository objects."""
        while not self._stop_event.is_set():
            with self._registry_lock:
                current_time = datetime.now(timezone.utc)
                for full_name in list(self._pool.keys()):
                    repo = self._pool[full_name]
                    if ((current_time - repo.last_read_time).total_seconds() > self.max_idle_time) or \
                        ((current_time - repo.creation_time).total_seconds() > 7 * self.max_idle_time and
                         (current_time - repo.last_read_time).total_seconds() > (self.max_idle_time // 4) + 1):
                        # release the lock due to object already created
                        del self._locks_registry[full_name]
                        # clear the cache content of repository object
                        self._pool[full_name].clear_cache()
            time.sleep(self.cleanup_interval)

    def stop_cleanup(self):
        """Stops the cleanup thread."""
        self._stop_event.set()

    def _get_repo_lock(self, full_name):
        """Retrieve or create a lock for a specific repository."""
        with self._registry_lock:
            if full_name not in self._locks_registry:
                # Create a new lock for the repository creation
                self._locks_registry[full_name] = Lock()
            return self._locks_registry[full_name]

    def get_repository(self, full_name, **kwargs) -> Repository:
        """Retrieve a repository from the pool or create a new one if it doesn't exist."""
        if full_name in self._pool:
            return self._pool[full_name]
        repo_lock = self._get_repo_lock(full_name)
        with repo_lock:
            if full_name not in self._pool:
                self._pool[full_name] = Repository(
                    full_name, self.github_instance, **kwargs)
        return self._pool[full_name]
