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
import requests

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
    
    def set_github(self, github_instance: ExtendedGithub):
        self._github = github_instance
        with self._repo_lock:  # Locking for write action
            self._repo = self._github.get_repo(self.full_name)

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
                        logger.error(
                            f"Error retrieving repository '{self.full_name}':{str(e)}")
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

        # Skip files that don't need processing
        if any(file_path.endswith(ext) for ext in [
            # Binary and Compiled Files
            '.exe', '.dll', '.so', '.dylib', '.bin', '.obj', '.o', '.a',
            '.lib', '.jar', '.war', '.ear', '.class', '.pdb', '.ilk', '.exp',
            '.apk', '.aab', '.ipa', '.wasm',
            
            # Media Files
            '.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp', '.tiff', '.webp',
            '.svg', '.eps', '.psd', '.ai', '.sketch',
            '.mp3', '.mp4', '.wav', '.flac', '.ogg', '.m4a',
            '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv',
            '.ttf', '.otf', '.eot', '.woff', '.woff2',
            
            # Compressed and Binary Data
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tgz',
            '.pkl', '.pickle',
            '.npy', '.npz',
            '.h5', '.hdf5',
            
            # Lock Files and Dependencies
            '.lock', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            'Gemfile.lock', 'poetry.lock', 'Cargo.lock', 'composer.lock',
            
            # Compiled Python
            '.pyc', '.pyo',
            
            # System and Hidden Files
            '.DS_Store', 'Thumbs.db',
            
            # Generated Code Files
            '.g.dart', '.freezed.dart',
            '.pb.go',
            '_pb2.py', '_pb2_grpc.py',
            '.generated.ts', '.generated.tsx',
            '.proto.ts', '.proto.js',
            '.min.js', '.min.css',
            '.bundle.js', '.bundle.css',
            '.chunk.js', '.chunk.css',
            
            # IDE Generated
            '.pbxproj', '.xcworkspacedata',
            '.csproj.user', '.suo',
            '.iml', '.ipr', '.iws',
            
            # Map Files
            '.map', '.js.map', '.css.map',

            # PowerBI Binary and Cache Files
            '.pbix',       # PowerBI packaged report file - binary format containing report, data model and resources
            '.pbit',       # PowerBI template file - binary format for report templates
            '.abf',        # Analysis Services Backup File - contains model and data cache

            # PowerBI Dataset Files
            '.bim',        # Analysis Services Tabular Model file - contains data model definitions
            '.database',   # PowerBI database definition file - auto-generated
            '.deploymentoptions',  # Deployment configuration file - auto-generated
            '.deploymenttargets',  # Deployment targets file - auto-generated

        ]) or any(pattern in file_path for pattern in [
            # Cache and Temporary Directories
            '/__pycache__/',
            '/.git/',
            '/.idea/',
            '/.vscode/',
            '/.vs/',
            '/.svn/',
            '/.hg/',
            '/.sass-cache/',
            '/.parcel-cache/',
            '/.cache/',
            '/tmp/',
            '/temp/',
            
            # Package Manager Directories
            '/node_modules/',
            '/bower_components/',
            
            # Test Coverage and Reports
            '/coverage/',
            '/.nyc_output/',
            '/.pytest_cache/',
            '/.tox/',
            
            # Environment and Runtime
            '/venv/',
            '/.env/',
            '/.virtualenv/',
            
            # Framework Generated
            '/.dart_tool/',
            '/.pub-cache/',
            '/.angular/',
            '/.nuxt/',
            '/.next/',
            '/.ipynb_checkpoints/',
            
            # CI/CD
            '/.github/workflows/',
            '/.gitlab/ci/',
            '/.circleci/',
            
            # Logs
            '/logs/',
            '/_logs/',
            '/var/log/',
            '/log/files/',
            '/application/log/',
            
            # PowerBI Specific Directories
            '/.pbi/',           # PowerBI cache and temporary files directory
            '/Dataset/.pbi/',   # PowerBI dataset cache directory
            '/.pbixproj/',      # PowerBI project configuration directory - contains auto-generated files
            
            # Binary Assets
            '/assets/images/',
            '/assets/fonts/',
            '/assets/media/',
            '/public/images/',
            '/public/fonts/',
            '/static/images/',
            '/static/fonts/'
        ]):
            logger.debug(f"Skipping non-processable file: {file_path}")
            return None

        if file_key not in self._file_contents:  # Check if file content has already been fetched
            with self._file_contents_lock:  # Locking for thread-safe write action
                if file_key not in self._file_contents:  # Double-check after acquiring the lock
                    try:
                        if sha is not None:
                            file_content = self.repo.get_contents(file_path, ref=sha)
                        else:
                            file_content = self.repo.get_contents(file_path)
                        
                        # Handle directory case
                        if isinstance(file_content, list):
                            logger.debug(f"Path {file_path} is a directory")
                            return None

                        # Improved encoding handling
                        if file_content.encoding == 'base64':
                            decoded_content = base64.b64decode(file_content.content).decode('utf-8')
                        elif (file_content.encoding is None or file_content.encoding == 'none') and hasattr(file_content, 'download_url') and file_content.download_url:
                            try:
                                logger.debug(f"Downloading file {file_path} from {file_content.download_url}")
                                # Use requests to download the file content
                                response = requests.get(
                                    file_content.download_url,
                                    timeout=30,
                                    headers={'Accept': 'application/vnd.github.v3.raw'}
                                )
                                response.raise_for_status()
                                decoded_content = response.text
                            except requests.RequestException as e:
                                logger.error(f"Failed to download file {file_path}: {str(e)}")
                                return None
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
                                (re.sub(r'\n+\s*\n+', '\n',
                                    (issue.body or "").replace('\r', '\n').strip()) if issue.body else "No issue body provided")
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
                    except Exception as e:
                        logger.error(f"Unexpected error processing : {str(e)}")
                        return None
        self.update_last_read_time()
        return self._issues[number]
    
    def extract_related_issues(self, pr_data: Dict[str, Any]) -> List[int]:
        """
        Extracts related issue numbers from PR data using adaptive strategies based on content length.

        Uses different matching strategies:
        - Short descriptions (<200 chars): Aggressive patterns for simple references
        - Long descriptions (>=200 chars): Strict patterns to avoid false positives

        Args:
            pr_data: Complete pull request data dictionary
            
        Returns:
            List[int] - Sorted list of unique issue numbers
        """
        # GitHub's official closing keywords
        closing_keywords = (
        'close', 'closes', 'closed',
        'fix', 'fixes', 'fixed',
        'resolve', 'resolves', 'resolved',
        'address', 'addresses', 'addressing',
        'relate', 'relates', 'related',
        'see',
        'issue', 'bug', 'ticket', 'todo', 'task'
        )

        issues = set()

        def get_description_length(data: Dict[str, Any]) -> int:
            """Get the length of PR description for strategy selection"""
            try:
                description = data.get('pr_metadata', {}).get('description', '')
                return len(description) if isinstance(description, str) else 0
            except:
                return 0

        def extract_with_aggressive_patterns(text: str) -> None:
            """Aggressive patterns for short, focused descriptions"""
            if not isinstance(text, str):
                return
                
            patterns = [
                # Simple #123 reference (most common in short descriptions)
                r'#(\d+)(?!\d)',
                
                # Full GitHub URLs
                rf'(?:https?://)?github\.com/{re.escape(self.full_name)}/(?:issues|pull)/(\d+)',
                
                # Closing keywords with flexible spacing
                fr'(?:{"|".join(closing_keywords)})\s*:?\s*#?(\d+)(?!\d)',
                
                # Action words commonly used in short descriptions
                r'(?:addresses?|references?|relates?\s+to|see)\s+#?(\d+)(?!\d)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                valid_matches = [
                    int(match) for match in matches 
                    if match.isdigit() and len(match) <= 6 and int(match) > 0
                ]
                issues.update(valid_matches)

        def extract_with_strict_patterns(text: str) -> None:
            """Strict patterns for long descriptions to avoid false positives"""
            if not isinstance(text, str):
                return
                
            patterns = [
                # Full GitHub URLs (always reliable)
                rf'(?:https?://)?github\.com/{re.escape(self.full_name)}/(?:issues|pull)/(\d+)',
                
                # Closing keywords with word boundaries
                fr'\b(?:{"|".join(closing_keywords)})\s*:?\s*#(\d+)\b',
                
                # Explicit issue references with word boundaries  
                r'\b(?:issue|bug|ticket|pr|pull\s+request)\s*:?\s*#?(\d+)\b',
                
                # Cross-repo references
                rf'\b{re.escape(self.full_name)}#(\d+)\b',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                valid_matches = [
                    int(match) for match in matches 
                    if match.isdigit() and len(match) <= 6 and int(match) > 0
                ]
                issues.update(valid_matches)

        def extract_from_text(text: str, use_aggressive: bool = False) -> None:
            """Extract issue numbers using appropriate strategy"""
            if use_aggressive:
                extract_with_aggressive_patterns(text)
            else:
                extract_with_strict_patterns(text)

        def process_value(value: Any, use_aggressive: bool = False) -> None:
            """Recursively process values and extract issue numbers"""
            if isinstance(value, dict):
                for v in value.values():
                    process_value(v, use_aggressive)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    process_value(item, use_aggressive)
            elif isinstance(value, str):
                extract_from_text(value, use_aggressive)

        # Determine strategy based on description length
        desc_length = get_description_length(pr_data)
        use_aggressive_strategy = desc_length < 200

        # Process all PR data
        process_value(pr_data, use_aggressive_strategy)

        return sorted(list(issues))


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

    def get_pr_content(self, number, pr=None, context_lines=10, force_update=False) -> Dict[str, Any]:
        """
        Retrieves and processes the content of a pull request.

        :param number: The PR number.
        :param pr: Optional PR object.
        :param context_lines: Number of context lines for diffs.
        :return: A dictionary containing detailed PR information.
        """
        if number not in self._prs or force_update:  # Check if issue has already been fetched
            with self._pr_lock:  # Locking for write action
                if number not in self._prs or force_update:  # Check if issue has already been fetched after get lock
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
                            "commits": [],
                            "file_changes": [],
                            "ci_cd_results": [],
                            "interactions": []
                        }

                        # Extract related issues from PR description and comments
                        related_issues = self.extract_related_issues(pr_data)
                        pr_data['related_issues'] = self.get_issue_contents(related_issues, pr.number)

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

                        # Fetch and process commits
                        try:
                            commits = pr.get_commits()
                            for commit in commits:
                                commit_data = {
                                    "sha": commit.sha,
                                    "message": commit.commit.message,
                                    "author": commit.commit.author.name,
                                    "date": self.to_isoformat(commit.commit.author.date),
                                    "stats": {
                                        "additions": commit.stats.additions,
                                        "deletions": commit.stats.deletions,
                                        "total": commit.stats.total
                                    },
                                    "files": [f.filename for f in commit.files]  # Just keep changed file names
                                }
                                pr_data["commits"].append(commit_data)
                        except GithubException as e:
                            logger.exception(f"Error fetching commits for PR #{number}")
                            pr_data["commits"] = []
                            pr_data["commit_stats"] = {}

                        # Process file changes
                        comparison = None
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
                            if base_content is None and head_content is None:
                                custom_diff = '[SKIPPED] File type not suitable for diff analysis'
                            else:
                                custom_diff = DiffGenerator.generate_custom_diff(base_content, head_content, context_lines)
                                if not custom_diff or custom_diff.strip() == '':
                                    try:
                                        #use compare API if custom diff is empty
                                        if not comparison:
                                            comparison = self.repo.compare(pr.base.sha, pr.head.sha)
                                        patches = [f.patch for f in comparison.files if f.filename == file_path]
                                        custom_diff = patches[0] if patches else None
                                    except Exception as e:
                                        logger.exception(f"Error fetching file diff for PR #{number}")
                                        custom_diff = ''

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
                    except Exception as e:
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

    def get_repository(self, full_name, github_instance=None, **kwargs) -> Repository:
        """
        Retrieve a repository from the pool or create a new one if it doesn't exist.
        
        If you are using github_install_id to generate a new repository object, you should pass new github_instance to the function.
        Otherwise the default github_instance within the pool might not fit to the new repository object.
        """

        if full_name in self._pool:
            # repo = self._pool[full_name]
            # repo.update_last_read_time()
            # if github_instance is not None:
            #     repo.set_github(github_instance)
            # return repo
            return self._pool[full_name]
        
        repo_lock = self._get_repo_lock(full_name)
        with repo_lock:
            if full_name not in self._pool:
                if github_instance is not None:
                    repo = Repository(full_name, github_instance, **kwargs)
                else:
                    repo = Repository(full_name, self.github_instance, **kwargs)
                self._pool[full_name] = repo

        return self._pool[full_name]
