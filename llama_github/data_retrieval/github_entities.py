from github import GithubException
from threading import Lock, Event, Thread
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from typing import Optional, Dict, Any, List, Callable
from llama_github.logger import logger
from llama_github.github_integration.github_auth_manager import (
    ExtendedGithub,
    RetrievalOutcome,
    RetrievalResult,
)
import re
from dateutil import parser
import base64
import requests

from llama_github.config.config import config
from llama_github.utils import DiffGenerator, CodeAnalyzer

__all__ = [
    "BoundedTextReadOptIn",
    "BoundedTextReadOutcome",
    "BoundedTextReadResult",
    "CIStatusSnapshot",
    "Repository",
    "RepositoryPool",
]


DEFAULT_RELATED_ISSUE_MAX_HITS = 20
BOUNDED_TEXT_SOURCE_MAX_BYTES = 2 * 1024 * 1024

_DEPENDENCY_LOCK_BASENAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
        "pnpm-lock.yml",
        "yarn.lock",
        "bun.lock",
        "bun.lockb",
        "uv.lock",
        "poetry.lock",
        "pipfile.lock",
        "go.sum",
        "cargo.lock",
        "gradle.lockfile",
        "gemfile.lock",
        "composer.lock",
        "pubspec.lock",
        "mix.lock",
        "flake.lock",
        "renv.lock",
        "package.resolved",
        "packages.lock.json",
    }
)


class BoundedTextReadOutcome(str, Enum):
    """Truthful terminal outcome for one bounded repository text read."""

    SUCCESS = "success"
    EXCLUDED_BY_POLICY = "excluded_by_policy"
    NOT_FOUND = "not_found"
    OVERSIZE = "oversize"
    BINARY_OR_NON_UTF8 = "binary_or_non_utf8"
    DIRECTORY = "directory"
    ERROR = "error"


class BoundedTextReadOptIn(str, Enum):
    """Narrow high-intent exceptions to the default repository-read policy."""

    DEPENDENCY_LOCK = "dependency_lock"
    CI_CONFIG = "ci_config"


@dataclass(frozen=True)
class BoundedTextReadResult:
    """Text plus bounded, JSON-safe retrieval metadata."""

    outcome: BoundedTextReadOutcome
    content: Optional[str] = None
    source_size_bytes: Optional[int] = None
    bytes_read: int = 0
    max_bytes: int = BOUNDED_TEXT_SOURCE_MAX_BYTES
    policy_class: Optional[str] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None

    def to_meta(self) -> dict:
        """Return metadata only; never include repository text."""
        return {
            "outcome": self.outcome.value,
            "source_size_bytes": self.source_size_bytes,
            "bytes_read": self.bytes_read,
            "max_bytes": self.max_bytes,
            "policy_class": self.policy_class,
            "status_code": self.status_code,
            "error_type": self.error_type,
        }


def _normalized_repo_path(file_path: str) -> str:
    return "/" + str(file_path or "").replace("\\", "/").lstrip("/")


def _bounded_text_policy_class(file_path: str) -> Optional[str]:
    """Classify paths that the bounded API may not read by default."""
    normalized = _normalized_repo_path(file_path)
    lowered = normalized.casefold()
    basename = lowered.rsplit("/", 1)[-1]

    if (
        basename == ".env"
        or (basename.startswith(".env.") and basename not in {".env.example", ".env.sample"})
        or basename in {"id_rsa", "id_ed25519", "credentials", "secrets.yml", "secrets.yaml"}
        or any(part in lowered for part in ("/.git/", "/.aws/", "/.ssh/"))
    ):
        return "sensitive"

    if (
        basename.endswith(".lock")
        or basename.endswith(".lockfile")
        or basename.endswith(".lock.json")
        or basename.endswith("-lock.json")
        or basename.endswith("_lock.json")
        or basename in _DEPENDENCY_LOCK_BASENAMES
    ):
        return BoundedTextReadOptIn.DEPENDENCY_LOCK.value

    if (
        lowered.startswith("/.github/workflows/")
        or lowered.startswith("/.gitlab/ci/")
        or lowered.startswith("/.circleci/")
        or lowered.startswith("/.buildkite/")
        or basename
        in {
            ".gitlab-ci.yml",
            ".gitlab-ci.yaml",
            ".travis.yml",
            ".travis.yaml",
            "azure-pipelines.yml",
            "azure-pipelines.yaml",
            "bitbucket-pipelines.yml",
            "bitbucket-pipelines.yaml",
            "jenkinsfile",
            "appveyor.yml",
            "appveyor.yaml",
        }
    ):
        return BoundedTextReadOptIn.CI_CONFIG.value

    if any(
        basename.endswith(suffix)
        for suffix in (
            ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a",
            ".lib", ".jar", ".war", ".ear", ".class", ".apk", ".wasm",
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
            ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".webm",
            ".ttf", ".otf", ".woff", ".woff2", ".zip", ".rar", ".7z",
            ".tar", ".gz", ".bz2", ".xz", ".pkl", ".pickle", ".npy",
            ".npz", ".h5", ".hdf5", ".pyc", ".pyo", ".min.js", ".min.css",
            ".bundle.js", ".bundle.css", ".map", ".pbix", ".pbit", ".abf",
        )
    ):
        return "binary_generated_or_minified"

    if any(
        segment in lowered
        for segment in (
            "/node_modules/", "/bower_components/", "/__pycache__/", "/.git/",
            "/coverage/", "/.pytest_cache/", "/.tox/", "/venv/",
            "/.virtualenv/", "/.cache/", "/.dart_tool/", "/.next/",
            "/.nuxt/", "/.ipynb_checkpoints/",
        )
    ):
        return "binary_generated_or_minified"
    return None


@dataclass(frozen=True)
class CIStatusSnapshot:
    """Current-head CI evidence plus typed retrieval-completeness metadata."""

    head_sha: str
    statuses: list[dict] = field(default_factory=list)
    check_runs: list[dict] = field(default_factory=list)
    outcome: RetrievalOutcome = RetrievalOutcome.NO_HIT
    statuses_meta: dict = field(default_factory=dict)
    check_runs_meta: dict = field(default_factory=dict)
    state: Optional[str] = None

    @property
    def retrieval_meta(self) -> dict:
        """Return JSON-safe per-source and aggregate completeness metadata."""
        statuses_meta = dict(self.statuses_meta)
        check_runs_meta = dict(self.check_runs_meta)
        statuses_meta["current_item_count"] = len(self.statuses)
        check_runs_meta["current_item_count"] = len(self.check_runs)
        return {
            "ci_aggregate": {"outcome": self.outcome.value},
            "ci_statuses": statuses_meta,
            "ci_check_runs": check_runs_meta,
        }

    def to_dict(self) -> dict:
        """Return the stable JSON-safe shape consumed by retrieval clients."""
        return {
            "head_sha": self.head_sha,
            "state": self.state,
            "statuses": list(self.statuses),
            "check_runs": list(self.check_runs),
            "_retrieval_meta": self.retrieval_meta,
        }


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
        self._bounded_text_reads = {}
        self._issues = {}  # Singleton pattern for file contents
        self._readme = None  # Singleton pattern for README content
        self._prs = {}  # Singleton pattern for PR content
        self._retrieval_meta = {}

    def _bounded_retrieval(
        self,
        status_method: str,
        legacy_method: str,
        **kwargs,
    ) -> RetrievalResult:
        """Call a status-aware client method while retaining old client compatibility."""
        method = getattr(self._github, status_method, None)
        if callable(method):
            result = method(**kwargs)
            if isinstance(result, RetrievalResult):
                return result

        items = getattr(self._github, legacy_method)(**kwargs)
        items = items if isinstance(items, list) else list(items or [])
        return RetrievalResult(
            items=items,
            outcome=RetrievalOutcome.OK if items else RetrievalOutcome.NO_HIT,
            pages_fetched=1,
        )

    @staticmethod
    def _bounded_iterable(iterable, max_items: int) -> tuple[list, bool]:
        """Consume at most ``max_items + 1`` values so truncation stays observable."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        values = []
        iterator = iter(iterable)
        for _ in range(max_items + 1):
            try:
                values.append(next(iterator))
            except StopIteration:
                break
        return values[:max_items], len(values) > max_items

    @staticmethod
    def _bounded_github_call(
        fetch: Callable[[], Any],
        max_items: int,
    ) -> RetrievalResult:
        """Preserve items already fetched when a paginated PyGithub call fails."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")

        values = []
        try:
            iterator = iter(fetch())
            for _ in range(max_items + 1):
                try:
                    values.append(next(iterator))
                except StopIteration:
                    break
        except Exception as exc:
            retained = values[:max_items]
            return RetrievalResult(
                items=retained,
                outcome=(
                    RetrievalOutcome.PARTIAL
                    if retained
                    else RetrievalOutcome.ERROR
                ),
                truncated=bool(retained),
                status_code=getattr(exc, "status", None),
                error_type=type(exc).__name__,
            )

        truncated = len(values) > max_items
        retained = values[:max_items]
        return RetrievalResult(
            items=retained,
            outcome=(
                RetrievalOutcome.PARTIAL
                if truncated
                else (RetrievalOutcome.OK if retained else RetrievalOutcome.NO_HIT)
            ),
            truncated=truncated,
        )

    @staticmethod
    def _aggregate_retrieval_outcome(
        statuses_result: RetrievalResult,
        check_runs_result: RetrievalResult,
    ) -> RetrievalOutcome:
        """Aggregate fetch completeness without making a merge-readiness judgment."""
        outcomes = {statuses_result.outcome, check_runs_result.outcome}
        if outcomes == {RetrievalOutcome.ERROR}:
            return RetrievalOutcome.ERROR
        if RetrievalOutcome.ERROR in outcomes or RetrievalOutcome.PARTIAL in outcomes:
            return RetrievalOutcome.PARTIAL
        if RetrievalOutcome.OK in outcomes:
            return RetrievalOutcome.OK
        return RetrievalOutcome.NO_HIT

    def get_ci_status_with_status(
        self,
        head_sha: str,
        *,
        max_statuses: int = 100,
        max_check_runs: int = 100,
    ) -> CIStatusSnapshot:
        """Fetch bounded, independently typed CI evidence for one exact head SHA."""
        normalized_head_sha = str(head_sha or "").strip()
        if not normalized_head_sha:
            raise ValueError("head_sha is required")
        if max_statuses <= 0 or max_check_runs <= 0:
            raise ValueError("CI retrieval limits must be positive")

        try:
            repo = self.repo
            if repo is None:
                raise RuntimeError("Repository is unavailable")
            head_commit = repo.get_commit(sha=normalized_head_sha)
        except Exception as exc:
            failure = RetrievalResult(
                outcome=RetrievalOutcome.ERROR,
                status_code=getattr(exc, "status", None),
                error_type=type(exc).__name__,
            )
            logger.error(
                "CI head-commit retrieval failed error_type=%s",
                type(exc).__name__,
            )
            return CIStatusSnapshot(
                head_sha=normalized_head_sha,
                outcome=RetrievalOutcome.ERROR,
                statuses_meta=failure.to_meta(),
                check_runs_meta=failure.to_meta(),
            )

        statuses_result = self._bounded_github_call(
            head_commit.get_statuses,
            max_statuses,
        )
        check_runs_result = self._bounded_github_call(
            head_commit.get_check_runs,
            max_check_runs,
        )
        statuses = [
            {
                "context": status.context,
                "state": status.state,
                "description": status.description,
                "target_url": status.target_url,
                "created_at": self.to_isoformat(status.created_at),
                "updated_at": self.to_isoformat(status.updated_at),
            }
            for status in statuses_result.items
        ]
        # Commit statuses are an append-only stream keyed by ``context``.
        # Keep only the newest state for each logical context so a completed
        # rerun cannot be shadowed by an obsolete failure with another URL.
        current_statuses: dict[str, dict] = {}
        for status in statuses:
            context = str(status.get("context") or "unknown")
            existing = current_statuses.get(context)
            if existing is None or str(status.get("updated_at") or "") >= str(
                existing.get("updated_at") or ""
            ):
                current_statuses[context] = status
        statuses = list(current_statuses.values())
        check_runs = [
            {
                "name": check_run.name,
                "status": check_run.status,
                "conclusion": check_run.conclusion,
                "started_at": (
                    self.to_isoformat(check_run.started_at)
                    if check_run.started_at
                    else None
                ),
                "completed_at": (
                    self.to_isoformat(check_run.completed_at)
                    if check_run.completed_at
                    else None
                ),
                "details_url": check_run.html_url,
            }
            for check_run in check_runs_result.items
        ]
        status_states = {
            str(status.get("state") or "").strip().lower()
            for status in statuses
        }
        if status_states.intersection({"failure", "error"}):
            combined_status_state = "failure"
        elif "pending" in status_states:
            combined_status_state = "pending"
        elif status_states and status_states <= {"success"}:
            combined_status_state = "success"
        else:
            combined_status_state = None
        return CIStatusSnapshot(
            head_sha=normalized_head_sha,
            statuses=statuses,
            check_runs=check_runs,
            outcome=self._aggregate_retrieval_outcome(
                statuses_result,
                check_runs_result,
            ),
            statuses_meta=statuses_result.to_meta(),
            check_runs_meta=check_runs_result.to_meta(),
            # Kept for the historical PR-content shape; it is not a merge verdict.
            state=combined_status_state,
        )

    def update_last_read_time(self):
        self.last_read_time = datetime.now(timezone.utc)

    def get_readme(self) -> str:
        """
        Retrieves the README content of the repository using a singleton design pattern.
        """
        repo = self.repo
        if repo is None:
            return None
        if self._readme is None:  # Check if README content has already been fetched
            with self._readme_lock:  # Locking for write action
                if self._readme is None:  # Check if README content has already been fetched after get lock
                    try:
                        readme = repo.get_readme()
                        self._readme = readme.decoded_content.decode("utf-8")
                    except GithubException as e:
                        logger.error("README retrieval failed error_type=%s", type(e).__name__)
                        self._readme = None
        self.update_last_read_time()
        return self._readme

    @property
    def repo(self):
        return self.get_repo()
    
    def set_github(self, github_instance: ExtendedGithub):
        if github_instance is None:
            return
        self._github = github_instance
        with self._repo_lock:  # Locking for write action
            self._repo = self._github.get_repo(self.full_name)

    def get_repo(self):
        """
        Retrieves the Github Repo object of the repository using a singleton design pattern.
        """
        if self._github is None:
            logger.error(
                "GitHub credentials are required for repository retrieval",
            )
            return None
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
                            "Repository retrieval failed error_type=%s",
                            type(e).__name__,
                        )
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
                        logger.error(
                            "Repository structure retrieval failed error_type=%s",
                            type(e).__name__,
                        )
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
            logger.debug("Skipping non-processable file")
            return None

        if file_key not in self._file_contents:  # Check if file content has already been fetched
            with self._file_contents_lock:  # Locking for thread-safe write action
                if file_key not in self._file_contents:  # Double-check after acquiring the lock
                    try:
                        repo = self.repo
                        if repo is None:
                            return None
                        if sha is not None:
                            file_content = repo.get_contents(file_path, ref=sha)
                        else:
                            file_content = repo.get_contents(file_path)
                        
                        # Handle directory case
                        if isinstance(file_content, list):
                            logger.debug("Skipping directory path")
                            return None

                        # Improved encoding handling
                        if file_content.encoding == 'base64':
                            decoded_content = base64.b64decode(file_content.content).decode('utf-8')
                        elif (file_content.encoding is None or file_content.encoding == 'none') and hasattr(file_content, 'download_url') and file_content.download_url:
                            try:
                                logger.debug("Downloading raw file content")
                                # Use requests to download the file content
                                response = requests.get(
                                    file_content.download_url,
                                    timeout=(5, 30),
                                    headers={'Accept': 'application/vnd.github.v3.raw'}
                                )
                                response.raise_for_status()
                                decoded_content = response.text
                            except requests.RequestException as e:
                                logger.error(
                                    "Raw file download failed error_type=%s",
                                    type(e).__name__,
                                )
                                return None
                        else:
                            decoded_content = file_content.decoded_content.decode('utf-8')
                        
                        self._file_contents[file_key] = decoded_content
                    except GithubException as e:
                        logger.error(
                            "File-content retrieval failed error_type=%s",
                            type(e).__name__,
                        )
                        return None
                    except UnicodeDecodeError as e:
                        logger.error(
                            "File-content decode failed error_type=%s",
                            type(e).__name__,
                        )
                        return None

        self.update_last_read_time()
        return self._file_contents.get(file_key)

    def read_text_file_bounded(
        self,
        file_path: str,
        sha: Optional[str] = None,
        *,
        opt_in: Optional[BoundedTextReadOptIn] = None,
    ) -> BoundedTextReadResult:
        """Read one UTF-8 file under a fixed 2 MiB source cap.

        Dependency lockfiles and CI configuration require an exact, typed
        ``opt_in``. Other binary, generated, minified, or sensitive paths stay
        excluded even when an opt-in is supplied. The legacy
        :meth:`get_file_content` contract and cache are intentionally separate.
        """
        requested_opt_in = None
        if opt_in is not None:
            requested_opt_in = BoundedTextReadOptIn(opt_in).value

        policy_class = _bounded_text_policy_class(file_path)
        allowed_exception = policy_class in {
            BoundedTextReadOptIn.DEPENDENCY_LOCK.value,
            BoundedTextReadOptIn.CI_CONFIG.value,
        }
        if (
            (allowed_exception and requested_opt_in != policy_class)
            or (policy_class is not None and not allowed_exception)
            or (policy_class is None and requested_opt_in is not None)
        ):
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.EXCLUDED_BY_POLICY,
                policy_class=policy_class or "opt_in_path_mismatch",
            )

        cache_key = (str(file_path), sha, requested_opt_in)
        cached = self._bounded_text_reads.get(cache_key)
        if cached is not None:
            return cached

        try:
            repo = self.repo
            if repo is None:
                return BoundedTextReadResult(
                    outcome=BoundedTextReadOutcome.ERROR,
                    policy_class=policy_class,
                    error_type="repository_unavailable",
                )
            file_content = (
                repo.get_contents(file_path, ref=sha)
                if sha is not None
                else repo.get_contents(file_path)
            )
        except GithubException as exc:
            status_code = getattr(exc, "status", None)
            return BoundedTextReadResult(
                outcome=(
                    BoundedTextReadOutcome.NOT_FOUND
                    if status_code == 404
                    else BoundedTextReadOutcome.ERROR
                ),
                policy_class=policy_class,
                status_code=status_code,
                error_type=(
                    "http_404"
                    if status_code == 404
                    else type(exc).__name__
                ),
            )
        except Exception as exc:
            logger.warning(
                "Bounded text read failed operation=get_contents error_type=%s",
                type(exc).__name__,
            )
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.ERROR,
                policy_class=policy_class,
                error_type=type(exc).__name__,
            )

        try:
            is_directory = (
                isinstance(file_content, list)
                or getattr(file_content, "type", None) == "dir"
            )
            raw_size = getattr(file_content, "size", None)
        except Exception as exc:
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.ERROR,
                policy_class=policy_class,
                error_type=type(exc).__name__,
            )
        if is_directory:
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.DIRECTORY,
                policy_class=policy_class,
            )

        source_size = (
            raw_size
            if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size >= 0
            else None
        )
        if source_size is not None and source_size > BOUNDED_TEXT_SOURCE_MAX_BYTES:
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.OVERSIZE,
                source_size_bytes=source_size,
                max_bytes=BOUNDED_TEXT_SOURCE_MAX_BYTES,
                policy_class=policy_class,
            )

        try:
            encoding = getattr(file_content, "encoding", None)
            if encoding == "base64":
                encoded = file_content.content
                if not isinstance(encoded, str):
                    raise ValueError("base64 content is not text")
                raw_bytes = base64.b64decode(encoded)
                bytes_read = len(raw_bytes)
                status_code = 200
            else:
                raw_url = getattr(file_content, "url", None)
                if not isinstance(raw_url, str) or not raw_url:
                    raw_url = getattr(file_content, "download_url", None)
                bounded_request = getattr(self._github, "_request_bounded_bytes", None)
                if not isinstance(raw_url, str) or not raw_url or not callable(bounded_request):
                    return BoundedTextReadResult(
                        outcome=BoundedTextReadOutcome.ERROR,
                        source_size_bytes=source_size,
                        policy_class=policy_class,
                        error_type="raw_transport_unavailable",
                    )
                response = bounded_request(
                    raw_url,
                    max_bytes=BOUNDED_TEXT_SOURCE_MAX_BYTES,
                    operation="bounded_text_read",
                )
                if response.oversize:
                    declared_size = response.declared_size_bytes
                    return BoundedTextReadResult(
                        outcome=BoundedTextReadOutcome.OVERSIZE,
                        source_size_bytes=source_size or declared_size,
                        bytes_read=response.bytes_read,
                        policy_class=policy_class,
                        status_code=response.status_code,
                    )
                if response.error_type is not None or response.data is None:
                    return BoundedTextReadResult(
                        outcome=(
                            BoundedTextReadOutcome.NOT_FOUND
                            if response.status_code == 404
                            else BoundedTextReadOutcome.ERROR
                        ),
                        source_size_bytes=source_size,
                        bytes_read=response.bytes_read,
                        policy_class=policy_class,
                        status_code=response.status_code,
                        error_type=response.error_type or "empty_raw_response",
                    )
                raw_bytes = response.data
                bytes_read = response.bytes_read
                status_code = response.status_code
                if source_size is None:
                    source_size = response.declared_size_bytes

            if len(raw_bytes) > BOUNDED_TEXT_SOURCE_MAX_BYTES:
                return BoundedTextReadResult(
                    outcome=BoundedTextReadOutcome.OVERSIZE,
                    source_size_bytes=source_size or len(raw_bytes),
                    bytes_read=len(raw_bytes),
                    policy_class=policy_class,
                    status_code=status_code,
                )
            if b"\x00" in raw_bytes:
                raise UnicodeDecodeError("utf-8", raw_bytes, 0, 1, "NUL byte")
            decoded = raw_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError, TypeError):
            return BoundedTextReadResult(
                outcome=BoundedTextReadOutcome.BINARY_OR_NON_UTF8,
                source_size_bytes=source_size,
                policy_class=policy_class,
            )

        result = BoundedTextReadResult(
            outcome=BoundedTextReadOutcome.SUCCESS,
            content=decoded,
            source_size_bytes=source_size or len(raw_bytes),
            bytes_read=bytes_read,
            policy_class=policy_class,
            status_code=status_code,
        )
        with self._file_contents_lock:
            self._bounded_text_reads.setdefault(cache_key, result)
            result = self._bounded_text_reads[cache_key]
        self.update_last_read_time()
        return result


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
                            comments_result = self._bounded_retrieval(
                                "get_issue_comments_with_status",
                                "get_issue_comments",
                                repo_full_name=self.full_name,
                                issue_number=number,
                            )
                            comments = comments_result.items
                            self._retrieval_meta[f"issue:{number}:comments"] = (
                                comments_result.to_meta()
                            )
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
                        logger.error(
                            "Issue-content retrieval failed error_type=%s",
                            type(e).__name__,
                        )
                        return None
                    except Exception as e:
                        logger.error(
                            "Issue processing failed error_type=%s",
                            type(e).__name__,
                        )
                        return None
        self.update_last_read_time()
        return self._issues[number]
    
    @staticmethod
    def _related_issue_limit(max_issues: Optional[int]) -> int:
        """Resolve and validate the per-PR related-issue expansion limit."""
        resolved_limit = (
            config.get(
                "related_issue_max_hits",
                DEFAULT_RELATED_ISSUE_MAX_HITS,
            )
            if max_issues is None
            else max_issues
        )
        if isinstance(resolved_limit, bool) or not isinstance(resolved_limit, int):
            raise ValueError("max_issues must be a positive integer")
        if resolved_limit <= 0:
            raise ValueError("max_issues must be a positive integer")
        return resolved_limit

    @staticmethod
    def _select_related_issue_numbers(
        issue_numbers: List[int],
        *,
        exclude_number: Optional[int],
        max_issues: int,
    ) -> tuple[List[int], Dict[str, Any]]:
        """Deduplicate, exclude the current PR, and cap issue expansion."""
        discovered = []
        seen = set()
        for issue_number in issue_numbers:
            if issue_number in seen:
                continue
            seen.add(issue_number)
            discovered.append(issue_number)

        eligible = [
            issue_number
            for issue_number in discovered
            if issue_number != exclude_number
        ]
        selected = eligible[:max_issues]
        truncated = len(eligible) > max_issues
        meta = {
            "outcome": (
                "partial"
                if truncated
                else ("ok" if selected else "no_hit")
            ),
            "discovered_count": len(discovered),
            "eligible_count": len(eligible),
            "item_count": len(selected),
            "attempted_count": 0,
            "successful_count": 0,
            "max_items": max_issues,
            "truncated": truncated,
            "excluded_current_pr": (
                exclude_number is not None and exclude_number in seen
            ),
            "error_type": None,
        }
        return selected, meta

    def _expand_related_issue_contents(
        self,
        issue_numbers: List[int],
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Expand an already-bounded issue selection and finalize its metadata."""
        contents = [
            {
                "issue_number": issue_number,
                "issue_content": self.get_issue_content(issue_number),
            }
            for issue_number in issue_numbers
        ]
        successful_count = sum(
            item["issue_content"] is not None for item in contents
        )
        attempted_count = len(contents)
        meta.update(
            {
                "attempted_count": attempted_count,
                "successful_count": successful_count,
            }
        )
        if attempted_count and successful_count == 0:
            meta["outcome"] = "error"
            meta["error_type"] = "content_fetch_incomplete"
        elif successful_count < attempted_count:
            meta["outcome"] = "partial"
            meta["error_type"] = "content_fetch_incomplete"
        elif meta["truncated"]:
            meta["outcome"] = "partial"

        self._retrieval_meta["related_issues"] = dict(meta)
        return contents

    def extract_related_issues(
        self,
        pr_data: Dict[str, Any],
        *,
        max_issues: Optional[int] = None,
        exclude_number: Optional[int] = None,
    ) -> List[int]:
        """
        Extracts related issue numbers from PR-authored conversation fields.

        Uses different matching strategies:
        - Short descriptions (<200 chars): Aggressive patterns for simple references
        - Long descriptions (>=200 chars): Strict patterns to avoid false positives

        Only the PR title/body and explicit comment/interaction bodies are
        searched. File diffs, branch names, SHAs, CI output, and other nested
        strings are deliberately excluded because ``#123`` in those fields is
        not reliable evidence that the PR links issue 123.

        Args:
            pr_data: Complete pull request data dictionary
            max_issues: Maximum unique issue references to return. Uses the
                configured ``related_issue_max_hits`` default when omitted.
            exclude_number: Optional current PR number to exclude before the
                limit is applied.
            
        Returns:
            List[int] - Bounded, sorted list of unique issue numbers
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
            except (AttributeError, TypeError):
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
                # GitHub's unambiguous same-repository shorthand. Markdown
                # headings use a space after '#', so they do not match.
                r'(?<![\w/])#(\d{1,6})\b',

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

        # Determine strategy based on description length
        desc_length = get_description_length(pr_data)
        use_aggressive_strategy = desc_length < 200

        metadata = (
            pr_data.get("pr_metadata")
            if isinstance(pr_data, dict)
            and isinstance(pr_data.get("pr_metadata"), dict)
            else {}
        )
        source_texts = [
            value
            for value in (metadata.get("title"), metadata.get("description"))
            if isinstance(value, str) and value
        ]
        for collection_name in ("interactions", "comments"):
            collection = (
                pr_data.get(collection_name)
                if isinstance(pr_data, dict)
                and isinstance(pr_data.get(collection_name), list)
                else []
            )
            for item in collection:
                if not isinstance(item, dict):
                    continue
                for field_name in ("content", "body"):
                    value = item.get(field_name)
                    if isinstance(value, str) and value:
                        source_texts.append(value)

        for text in source_texts:
            extract_from_text(text, use_aggressive_strategy)

        limit = self._related_issue_limit(max_issues)
        selected, meta = self._select_related_issue_numbers(
            sorted(issues),
            exclude_number=exclude_number,
            max_issues=limit,
        )
        self._retrieval_meta["related_issues"] = meta
        return selected


    def get_issue_contents(
        self,
        issue_numbers: List[int],
        pr_number: int,
        *,
        max_issues: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves contents for a list of issue numbers, excluding the PR number.

        :param issue_numbers: List of issue numbers.
        :param pr_number: The PR number to exclude.
        :param max_issues: Maximum issue API calls to make.
        :return: A list of issue contents.
        """
        limit = self._related_issue_limit(max_issues)
        selected, meta = self._select_related_issue_numbers(
            issue_numbers,
            exclude_number=pr_number,
            max_issues=limit,
        )
        return self._expand_related_issue_contents(selected, meta)

    def _collect_related_issue_contents(
        self,
        pr_data: Dict[str, Any],
        pr_number: int,
        *,
        max_issues: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extract and expand related issues without losing discovery metadata."""
        issue_numbers = self.extract_related_issues(
            pr_data,
            max_issues=max_issues,
            exclude_number=pr_number,
        )
        meta = dict(self._retrieval_meta["related_issues"])
        return self._expand_related_issue_contents(issue_numbers, meta)
    
    def to_isoformat(self, dt: datetime) -> Optional[str]:
        """
        Converts a datetime object to ISO 8601 format with UTC timezone.

        :param dt: The datetime object to convert.
        :return: ISO 8601 formatted string or None if dt is None.
        """
        if dt:
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        return None

    def get_pr_content(
        self,
        number,
        pr=None,
        context_lines=10,
        force_update=False,
        *,
        related_issue_max_hits=None,
    ) -> Dict[str, Any]:
        """
        Retrieves and processes the content of a pull request.

        :param number: The PR number.
        :param pr: Optional PR object.
        :param context_lines: Number of context lines for diffs.
        :param related_issue_max_hits: Maximum related issues to expand.
        :return: A dictionary containing detailed PR information.
        """
        if number not in self._prs or force_update:  # Check if issue has already been fetched
            with self._pr_lock:  # Locking for write action
                if number not in self._prs or force_update:  # Check if issue has already been fetched after get lock
                    try:
                        logger.debug(f"Processing PR #{number}")
                        if pr is None:
                            repo = self.repo
                            if repo is None:
                                return None
                            pr = repo.get_pull(number)
                        pr_files_result = self._bounded_retrieval(
                            "get_pr_files_with_status",
                            "get_pr_files",
                            repo_full_name=self.full_name,
                            pr_number=pr.number,
                        )
                        pr_comments_result = self._bounded_retrieval(
                            "get_pr_comments_with_status",
                            "get_pr_comments",
                            repo_full_name=self.full_name,
                            pr_number=pr.number,
                        )
                        pr_files = pr_files_result.items
                        pr_comments = pr_comments_result.items

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
                                "head_sha": pr.head.sha,
                            },
                            "related_issues": [],
                            "commits": [],
                            "file_changes": [],
                            "ci_cd_results": {
                                "state": None,
                                "statuses": [],
                                "check_runs": [],
                            },
                            "interactions": [],
                            "_retrieval_meta": {
                                "pr_files": pr_files_result.to_meta(),
                                "pr_comments": pr_comments_result.to_meta(),
                            }
                        }

                        ci_snapshot = self.get_ci_status_with_status(pr.head.sha)
                        pr_data["ci_cd_results"] = {
                            "state": ci_snapshot.state,
                            "statuses": ci_snapshot.statuses,
                            "check_runs": ci_snapshot.check_runs,
                        }
                        pr_data["_retrieval_meta"].update(ci_snapshot.retrieval_meta)
                        for source, meta in (
                            ("statuses", ci_snapshot.statuses_meta),
                            ("check_runs", ci_snapshot.check_runs_meta),
                        ):
                            if meta.get("error_type"):
                                logger.warning(
                                    "CI evidence retrieval incomplete pr_number=%s source=%s outcome=%s error_type=%s",
                                    number,
                                    source,
                                    meta.get("outcome"),
                                    meta.get("error_type"),
                                )

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

                        # Related-issue extraction is intentionally limited to
                        # PR title/body and top-level PR comments.  The comments
                        # have already been fetched, so include them before
                        # expansion instead of recursively scanning file diffs,
                        # CI output, SHAs, or other non-conversation fields.
                        related_issue_sources = {
                            "pr_metadata": pr_data["pr_metadata"],
                            "interactions": list(pr_interactions),
                        }
                        pr_data["related_issues"] = self._collect_related_issue_contents(
                            related_issue_sources,
                            pr.number,
                            max_issues=related_issue_max_hits,
                        )
                        pr_data["_retrieval_meta"]["related_issues"] = dict(
                            self._retrieval_meta["related_issues"]
                        )
                        related_issue_meta = {
                            str(issue_number): self._retrieval_meta[
                                f"issue:{issue_number}:comments"
                            ]
                            for issue_number in (
                                item["issue_number"]
                                for item in pr_data["related_issues"]
                            )
                            if f"issue:{issue_number}:comments" in self._retrieval_meta
                        }
                        if related_issue_meta:
                            pr_data["_retrieval_meta"]["related_issue_comments"] = (
                                related_issue_meta
                            )

                        # Process reviews and inline comments
                        reviews_result = self._bounded_github_call(
                            pr.get_reviews, 100
                        )
                        review_comments_result = self._bounded_github_call(
                            pr.get_review_comments, 200
                        )
                        reviews = reviews_result.items
                        review_comments = review_comments_result.items
                        pr_data["_retrieval_meta"].update(
                            {
                                "reviews": reviews_result.to_meta(),
                                "review_comments": review_comments_result.to_meta(),
                            }
                        )
                        for review in reviews:
                            review_raw = (
                                review.raw_data
                                if isinstance(getattr(review, "raw_data", None), dict)
                                else {}
                            )
                            pr_interactions.append({
                                "type": "review",
                                "author": review.user.login,
                                "author_association": review_raw.get("author_association"),
                                "content": review.body,
                                "state": review.state,
                                "created_at": self.to_isoformat(review.submitted_at),
                                "comment_id": review.id
                            })

                        # A review can own multiple inline comments. Preserve
                        # every comment as its own interaction instead of
                        # repeatedly overwriting the review summary with the
                        # last comment in that review.
                        for comment in review_comments:
                            comment_raw = (
                                comment.raw_data
                                if isinstance(getattr(comment, "raw_data", None), dict)
                                else {}
                            )
                            user = getattr(comment, "user", None)
                            pr_interactions.append({
                                "type": "inline_comment",
                                "author": getattr(user, "login", None),
                                "author_association": comment_raw.get(
                                    "author_association"
                                ),
                                "content": comment.body,
                                "path": comment.path,
                                "diff_hunk": comment.diff_hunk,
                                "created_at": self.to_isoformat(
                                    getattr(comment, "created_at", None)
                                ),
                                "comment_id": comment.id,
                                "review_id": comment.pull_request_review_id,
                            })

                        # Sort interactions by creation time
                        pr_interactions.sort(
                            key=lambda item: (
                                not bool(item.get("created_at")),
                                str(item.get("created_at") or ""),
                            )
                        )
                        pr_data["interactions"] = pr_interactions

                        # Fetch and process commits
                        try:
                            commits, commits_truncated = self._bounded_iterable(
                                pr.get_commits(), 250
                            )
                            pr_data["_retrieval_meta"]["commits"] = {
                                "outcome": "partial" if commits_truncated else (
                                    "ok" if commits else "no_hit"
                                ),
                                "item_count": len(commits),
                                "truncated": commits_truncated,
                            }
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
                            logger.error(
                                "Commit retrieval failed pr_number=%s error_type=%s",
                                number,
                                type(e).__name__,
                            )
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
                                        logger.error(
                                            "Diff fallback failed pr_number=%s error_type=%s",
                                            number,
                                            type(e).__name__,
                                        )
                                        custom_diff = ''

                            # Categorize code changes
                            change_categories = CodeAnalyzer.categorize_change(custom_diff)

                            # Extract imports from head content
                            related_modules = (
                                CodeAnalyzer.extract_imports(head_content)
                                if head_content
                                and CodeAnalyzer.is_python_file(file_path, language)
                                else []
                            )

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
                            logger.debug("Processed PR file")

                        self._prs[number] = pr_data
                        logger.debug("Collected details for PR #%s", number)
                    except Exception as e:
                        logger.error(
                            "PR-content retrieval failed pr_number=%s error_type=%s",
                            number,
                            type(e).__name__,
                        )
        self.update_last_read_time()
        return self._prs.get(number)

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
            self._bounded_text_reads = {}
        with self._issue_lock:
            self._issues = {}  # Reset the issue contents cache
        with self._readme_lock:
            self._readme = None  # Reset the README content cache
        with self._repo_lock:
            self._repo = None  # Reset the Repo cache
        with self._pr_lock:
            self._prs = {}  # Reset the PR cache
        self._retrieval_meta = {}


class RepositoryPool:
    def __init__(
        self,
        github_instance,
        cleanup_interval=3600,
        max_idle_time=86400,
        cleanup_enabled=True,
    ):
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")
        if max_idle_time <= 0:
            raise ValueError("max_idle_time must be positive")
        self._locks_registry = {}
        self._registry_lock = Lock()
        self._pool = {}
        self.github_instance = github_instance
        self.cleanup_interval = cleanup_interval
        self.max_idle_time = max_idle_time
        self._stop_event = Event()
        self._cleanup_thread = None
        if cleanup_enabled:
            self._cleanup_thread = Thread(target=self._cleanup, daemon=True)
            self._cleanup_thread.start()

    def _cleanup_once(self, current_time=None):
        """Evict idle repositories and their per-repository locks from the pool."""
        current_time = current_time or datetime.now(timezone.utc)
        evicted = []
        with self._registry_lock:
            candidates = [
                (full_name, repo, self._locks_registry.get(full_name))
                for full_name, repo in self._pool.items()
            ]

        for full_name, candidate, repo_lock in candidates:
            if repo_lock is None or not repo_lock.acquire(blocking=False):
                continue
            try:
                with self._registry_lock:
                    repo = self._pool.get(full_name)
                    if repo is not candidate:
                        continue
                    idle_seconds = (current_time - repo.last_read_time).total_seconds()
                    age_seconds = (current_time - repo.creation_time).total_seconds()
                    should_evict = idle_seconds > self.max_idle_time or (
                        age_seconds > 7 * self.max_idle_time
                        and idle_seconds > (self.max_idle_time // 4) + 1
                    )
                    if should_evict:
                        evicted.append(self._pool.pop(full_name))
                        if self._locks_registry.get(full_name) is repo_lock:
                            self._locks_registry.pop(full_name, None)
            finally:
                repo_lock.release()

        for repo in evicted:
            repo.clear_cache()
        return len(evicted)

    def _cleanup(self):
        """Wait interruptibly between cleanup passes."""
        while not self._stop_event.wait(self.cleanup_interval):
            self._cleanup_once()

    def stop_cleanup(self):
        """Stops the cleanup thread."""
        self._stop_event.set()
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)

    close = stop_cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

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

        with self._registry_lock:
            repo = self._pool.get(full_name)
        if repo is not None:
            if github_instance is not None:
                repo.set_github(github_instance)
                self.github_instance = github_instance
            repo.update_last_read_time()
            return repo
        
        repo_lock = self._get_repo_lock(full_name)
        with repo_lock:
            with self._registry_lock:
                repo = self._pool.get(full_name)
            if repo is None:
                if github_instance is not None:
                    repo = Repository(full_name, github_instance, **kwargs)
                    self.github_instance = github_instance
                else:
                    repo = Repository(full_name, self.github_instance, **kwargs)
                with self._registry_lock:
                    self._pool[full_name] = repo

        return repo
