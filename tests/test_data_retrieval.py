import base64
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from github import GithubException

from llama_github.data_retrieval.github_api import GitHubAPIHandler
from llama_github.data_retrieval.github_entities import (
    BOUNDED_TEXT_SOURCE_MAX_BYTES,
    BoundedTextReadOptIn,
    BoundedTextReadOutcome,
    Repository,
    RepositoryPool,
)
from llama_github.github_integration.github_auth_manager import (
    RetrievalOutcome,
    RetrievalResult,
)


class TestRepository:
    def test_repository_initialization(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object

        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.full_name == "owner/test-repo"
        assert repo.id == 12345
        assert repo.language == "Python"

    def test_get_readme_caching(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        mock_readme = MagicMock()
        mock_readme.decoded_content = b"# Readme"
        mock_repo_object.get_readme.return_value = mock_readme

        content1 = repo.get_readme()
        content2 = repo.get_readme()

        assert content1 == "# Readme"
        assert content2 == "# Readme"
        assert mock_repo_object.get_readme.call_count == 1

    def test_get_file_content_base64(
        self, mock_github_instance, mock_repo_object, mock_content_file
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = mock_content_file

        repo = Repository("owner/test-repo", mock_github_instance)
        content = repo.get_file_content("src/test.py")

        assert content == 'print("hello")'

    def test_get_file_content_cap_rejects_declared_oversize_before_decode_or_download(
        self, mock_github_instance, mock_repo_object
    ):
        class OversizeContent:
            size = 129

            @property
            def encoding(self):
                raise AssertionError("oversize content must not be decoded")

        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = OversizeContent()
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("results/large.json", max_source_bytes=128) is None
        mock_repo_object.get_contents.assert_called_once_with("results/large.json")

    def test_get_file_content_cap_applies_to_preexisting_cache(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo._file_contents["src/large.txt/head"] = "x" * 129

        assert (
            repo.get_file_content("src/large.txt", sha="head", max_source_bytes=128)
            is None
        )
        mock_repo_object.get_contents.assert_not_called()

    def test_get_file_content_cap_rejects_obviously_large_cache_without_encoding(
        self, mock_github_instance, mock_repo_object
    ):
        class EncodeMustNotRun(str):
            def encode(self, *args, **kwargs):
                raise AssertionError("obviously oversized text must not be copied")

        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo._file_contents["src/large.txt/head"] = EncodeMustNotRun("x" * 129)

        assert (
            repo.get_file_content("src/large.txt", sha="head", max_source_bytes=128)
            is None
        )
        mock_repo_object.get_contents.assert_not_called()

    def test_get_file_content_cap_counts_utf8_bytes_for_cached_text(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo._file_contents["src/multibyte.txt/head"] = "ééé"

        assert (
            repo.get_file_content(
                "src/multibyte.txt",
                sha="head",
                max_source_bytes=5,
            )
            is None
        )
        mock_repo_object.get_contents.assert_not_called()

    @patch("llama_github.data_retrieval.github_entities.base64.b64decode")
    def test_get_file_content_cap_rejects_large_base64_before_decode(
        self, mock_decode, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding="base64",
            content=base64.b64encode(b"x" * 129).decode("ascii"),
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/large.txt", max_source_bytes=128) is None
        mock_decode.assert_not_called()
        assert "src/large.txt" not in repo._file_contents

    def test_get_file_content_cap_accepts_base64_at_exact_boundary(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding="base64",
            content=base64.b64encode(b"x" * 128).decode("ascii"),
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content(
            "src/exact.txt", max_source_bytes=128
        ) == "x" * 128

    def test_get_file_content_non_ascii_base64_fails_closed(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding="base64",
            content="éééé",
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/broken.txt", max_source_bytes=128) is None
        assert "src/broken.txt" not in repo._file_contents

    def test_get_file_content_malformed_base64_fails_closed(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding="base64",
            content="abc",
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/broken.txt", max_source_bytes=128) is None
        assert "src/broken.txt" not in repo._file_contents

    @patch("llama_github.data_retrieval.github_entities.requests.get")
    def test_get_file_content_cap_rejects_raw_content_length_before_streaming(
        self, mock_get, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding=None,
            download_url="https://example.invalid/raw",
        )
        response = MagicMock()
        response.headers = {"Content-Length": "129"}
        mock_get.return_value = response
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/large.txt", max_source_bytes=128) is None
        mock_get.assert_called_once_with(
            "https://example.invalid/raw",
            stream=True,
            timeout=(5, 30),
            headers={"Accept": "application/vnd.github.v3.raw"},
        )
        response.iter_content.assert_not_called()
        response.close.assert_called_once_with()

    @patch("llama_github.data_retrieval.github_entities.requests.get")
    def test_get_file_content_cap_stops_raw_stream_at_local_boundary(
        self, mock_get, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding="none",
            download_url="https://example.invalid/raw",
        )
        response = MagicMock()
        response.headers = {}
        response.iter_content.return_value = iter([b"a" * 64, b"b" * 65])
        mock_get.return_value = response
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/large.txt", max_source_bytes=128) is None
        response.iter_content.assert_called_once_with(chunk_size=129)
        response.close.assert_called_once_with()
        assert "src/large.txt" not in repo._file_contents

    @patch("llama_github.data_retrieval.github_entities.requests.get")
    def test_get_file_content_cap_caches_utf8_raw_stream_within_boundary(
        self, mock_get, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            size=None,
            encoding=None,
            download_url="https://example.invalid/raw",
        )
        response = MagicMock()
        response.headers = {}
        response.iter_content.return_value = iter(["héllo".encode("utf-8")])
        mock_get.return_value = response
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("src/small.txt", max_source_bytes=8) == "héllo"
        assert repo.get_file_content("src/small.txt", max_source_bytes=8) == "héllo"
        mock_get.assert_called_once()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("value", [0, -1, True, "128"])
    def test_get_file_content_cap_requires_a_positive_integer(
        self, mock_github_instance, mock_repo_object, value
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        with pytest.raises(ValueError, match="positive integer"):
            repo.get_file_content("src/test.py", max_source_bytes=value)

    def test_bounded_text_read_preserves_legacy_lockfile_exclusion(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        assert repo.get_file_content("package-lock.json") is None
        result = repo.read_text_file_bounded("package-lock.json")

        assert result.outcome is BoundedTextReadOutcome.EXCLUDED_BY_POLICY
        assert result.policy_class == "dependency_lock"
        mock_repo_object.get_contents.assert_not_called()

    @pytest.mark.parametrize(
        ("path", "opt_in", "expected_policy"),
        [
            ("uv.lock", BoundedTextReadOptIn.DEPENDENCY_LOCK, "dependency_lock"),
            ("go.sum", BoundedTextReadOptIn.DEPENDENCY_LOCK, "dependency_lock"),
            ("gradle.lockfile", BoundedTextReadOptIn.DEPENDENCY_LOCK, "dependency_lock"),
            ("packages.lock.json", BoundedTextReadOptIn.DEPENDENCY_LOCK, "dependency_lock"),
            (
                ".github/workflows/test.yml",
                BoundedTextReadOptIn.CI_CONFIG,
                "ci_config",
            ),
            ("appveyor.yml", BoundedTextReadOptIn.CI_CONFIG, "ci_config"),
        ],
    )
    def test_bounded_text_read_allows_only_exact_high_intent_opt_in(
        self,
        mock_github_instance,
        mock_repo_object,
        path,
        opt_in,
        expected_policy,
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        content = b"version = 1\n"
        file_obj = SimpleNamespace(
            type="file",
            size=len(content),
            encoding="base64",
            content=base64.b64encode(content).decode("ascii"),
        )
        mock_repo_object.get_contents.return_value = file_obj
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded(path, sha="head", opt_in=opt_in)

        assert result.outcome is BoundedTextReadOutcome.SUCCESS
        assert result.content == content.decode("utf-8")
        assert result.policy_class == expected_policy
        assert result.source_size_bytes == len(content)
        assert result.bytes_read == len(content)
        assert "content" not in result.to_meta()
        mock_repo_object.get_contents.assert_called_once_with(path, ref="head")

    @pytest.mark.parametrize(
        ("path", "opt_in"),
        [
            ("uv.lock", BoundedTextReadOptIn.CI_CONFIG),
            ("src/main.py", BoundedTextReadOptIn.CI_CONFIG),
            (".env", None),
            ("dist/app.min.js", None),
        ],
    )
    def test_bounded_text_read_rejects_mismatched_or_non_text_policy(
        self, mock_github_instance, mock_repo_object, path, opt_in
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded(path, opt_in=opt_in)

        assert result.outcome is BoundedTextReadOutcome.EXCLUDED_BY_POLICY
        mock_repo_object.get_contents.assert_not_called()

    def test_bounded_text_read_rejects_oversize_before_content_access(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        file_obj = MagicMock()
        file_obj.type = "file"
        file_obj.size = BOUNDED_TEXT_SOURCE_MAX_BYTES + 1
        mock_repo_object.get_contents.return_value = file_obj
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded("src/large.txt")

        assert result.outcome is BoundedTextReadOutcome.OVERSIZE
        assert result.source_size_bytes == BOUNDED_TEXT_SOURCE_MAX_BYTES + 1
        assert file_obj.content.call_count == 0

    @pytest.mark.parametrize(
        ("payload_size", "expected"),
        [
            (BOUNDED_TEXT_SOURCE_MAX_BYTES, BoundedTextReadOutcome.SUCCESS),
            (BOUNDED_TEXT_SOURCE_MAX_BYTES + 1, BoundedTextReadOutcome.OVERSIZE),
        ],
    )
    def test_bounded_text_read_enforces_local_cap_after_decode(
        self, mock_github_instance, mock_repo_object, payload_size, expected
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        payload = b"a" * payload_size
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            type="file",
            size=BOUNDED_TEXT_SOURCE_MAX_BYTES,
            encoding="base64",
            content=base64.b64encode(payload).decode("ascii"),
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded("src/large.txt")

        assert result.outcome is expected

    @pytest.mark.parametrize("payload", [b"\xff\xfe", b"text\x00tail"])
    def test_bounded_text_read_rejects_binary_or_non_utf8(
        self, mock_github_instance, mock_repo_object, payload
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            type="file",
            size=len(payload),
            encoding="base64",
            content=base64.b64encode(payload).decode("ascii"),
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded("src/data.txt")

        assert result.outcome is BoundedTextReadOutcome.BINARY_OR_NON_UTF8

    def test_bounded_text_read_distinguishes_directory_not_found_and_error(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        mock_repo_object.get_contents.return_value = []
        assert (
            repo.read_text_file_bounded("src").outcome
            is BoundedTextReadOutcome.DIRECTORY
        )

        mock_repo_object.get_contents.side_effect = GithubException(404, {"message": "missing"})
        assert (
            repo.read_text_file_bounded("missing.txt").outcome
            is BoundedTextReadOutcome.NOT_FOUND
        )

        mock_repo_object.get_contents.side_effect = GithubException(503, {"message": "down"})
        assert (
            repo.read_text_file_bounded("unavailable.txt").outcome
            is BoundedTextReadOutcome.ERROR
        )

    def test_bounded_text_read_maps_raw_transport_outcomes(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_contents.return_value = SimpleNamespace(
            type="file",
            size=None,
            encoding="none",
            url="https://api.github.test/contents/src/config.txt",
            download_url=None,
        )
        mock_github_instance._request_bounded_bytes.return_value = SimpleNamespace(
            data=b"tail = true\n",
            bytes_read=12,
            declared_size_bytes=12,
            status_code=200,
            error_type=None,
            oversize=False,
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        result = repo.read_text_file_bounded("src/config.txt")

        assert result.outcome is BoundedTextReadOutcome.SUCCESS
        assert result.content == "tail = true\n"
        mock_github_instance._request_bounded_bytes.assert_called_once_with(
            "https://api.github.test/contents/src/config.txt",
            max_bytes=BOUNDED_TEXT_SOURCE_MAX_BYTES,
            operation="bounded_text_read",
        )

    def test_get_pr_content_includes_current_head_sha(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files.return_value = []
        mock_github_instance.get_pr_comments.return_value = []
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.ERROR,
            status_code=503,
            error_type="http_503",
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 7
        pr.title = "Bounded change"
        pr.body = "No linked issue"
        pr.user.login = "author"
        pr.raw_data = {"author_association": "MEMBER"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha-123"
        latest_commit = MagicMock()
        latest_commit.get_statuses.return_value = []
        latest_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = latest_commit
        pr.get_commits.return_value = []
        pr.get_reviews.return_value = []
        pr.get_review_comments.return_value = []

        repo = Repository("owner/test-repo", mock_github_instance)
        result = repo.get_pr_content(7, pr=pr)

        assert result["pr_metadata"]["head_sha"] == "head-sha-123"
        assert result["_retrieval_meta"]["pr_files"]["outcome"] == "error"
        assert result["_retrieval_meta"]["pr_files"]["error_type"] == "http_503"
        assert result["_retrieval_meta"]["pr_comments"]["outcome"] == "no_hit"
        assert result["_retrieval_meta"]["related_issues"] == {
            "outcome": "no_hit",
            "discovered_count": 0,
            "eligible_count": 0,
            "item_count": 0,
            "attempted_count": 0,
            "successful_count": 0,
            "max_items": 20,
            "truncated": False,
            "excluded_current_pr": False,
            "error_type": None,
        }
        assert result["_retrieval_meta"]["ci_statuses"]["outcome"] == "no_hit"
        assert result["_retrieval_meta"]["ci_aggregate"]["outcome"] == "no_hit"
        mock_repo_object.get_commit.assert_called_once_with(sha="head-sha-123")
        assert pr.get_commits.call_count == 1

    def test_get_pr_content_bounded_source_uses_api_patch_without_full_file_drift(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            items=[
                {
                    "filename": "results/large.json",
                    "status": "modified",
                    "language": "JSON",
                    "additions": 1,
                    "deletions": 1,
                    "changes": 2,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ],
            outcome=RetrievalOutcome.OK,
            pages_fetched=1,
            status_code=200,
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 9
        pr.title = "Bound generated source reads"
        pr.body = "No linked issue"
        pr.user.login = "author"
        pr.raw_data = {"author_association": "MEMBER"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha"
        pr.get_commits.return_value = []
        pr.get_reviews.return_value = []
        pr.get_review_comments.return_value = []
        head_commit = MagicMock()
        head_commit.get_statuses.return_value = []
        head_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = head_commit

        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_file_content = MagicMock(return_value=None)
        result = repo.get_pr_content(
            9,
            pr=pr,
            source_file_max_bytes=128,
            source_total_max_bytes=256,
        )

        assert result["file_changes"][0]["diff"] == "@@ -1 +1 @@\n-old\n+new"
        assert repo.get_file_content.call_args_list == [
            call(file_path="results/large.json", sha="base-sha", max_source_bytes=128),
            call(file_path="results/large.json", sha="head-sha", max_source_bytes=128),
        ]
        assert result["_retrieval_meta"]["file_content_budget"] == {
            "mode": "bounded",
            "file_max_bytes": 128,
            "total_max_bytes": 256,
            "attempted_reads": 2,
            "retained_reads": 0,
            "retained_source_bytes": 0,
            "unavailable_or_oversize_reads": 2,
            "budget_exhausted_reads": 0,
            "api_patch_fallbacks": 1,
            "missing_patch_fallbacks": 0,
            "outcome": "partial",
        }

    def test_get_pr_content_bounded_source_stops_at_total_budget(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            items=[
                {
                    "filename": name,
                    "status": "added",
                    "language": "Text",
                    "additions": 1,
                    "deletions": 0,
                    "changes": 1,
                    "patch": f"@@ -0,0 +1 @@\n+{name}",
                }
                for name in ("first.txt", "second.txt")
            ],
            outcome=RetrievalOutcome.OK,
            pages_fetched=1,
            status_code=200,
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 10
        pr.title = "Bound total source"
        pr.body = "No linked issue"
        pr.user.login = "author"
        pr.raw_data = {"author_association": "MEMBER"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha"
        pr.get_commits.return_value = []
        pr.get_reviews.return_value = []
        pr.get_review_comments.return_value = []
        head_commit = MagicMock()
        head_commit.get_statuses.return_value = []
        head_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = head_commit

        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_file_content = MagicMock(return_value="12345678")
        result = repo.get_pr_content(
            10,
            pr=pr,
            source_file_max_bytes=8,
            source_total_max_bytes=8,
        )

        repo.get_file_content.assert_called_once_with(
            file_path="first.txt", sha="head-sha", max_source_bytes=8
        )
        budget = result["_retrieval_meta"]["file_content_budget"]
        assert budget["retained_source_bytes"] == 8
        assert budget["budget_exhausted_reads"] == 1
        assert budget["api_patch_fallbacks"] == 1
        assert result["file_changes"][1]["diff"].endswith("+second.txt")

    @pytest.mark.parametrize(
        ("file_cap", "total_cap"),
        [(0, None), (128, 0), (None, 128)],
    )
    def test_get_pr_content_bounded_source_validates_caps(
        self, mock_github_instance, mock_repo_object, file_cap, total_cap
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        with pytest.raises(ValueError):
            repo.get_pr_content(
                1,
                source_file_max_bytes=file_cap,
                source_total_max_bytes=total_cap,
            )

    def test_get_pr_content_keeps_bounded_and_legacy_cache_identities_separate(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo._prs[12] = {"mode": "legacy"}
        repo._prs[(12, 128, 256)] = {"mode": "bounded"}

        assert repo.get_pr_content(12) == {"mode": "legacy"}
        assert repo.get_pr_content(
            12,
            source_file_max_bytes=128,
            source_total_max_bytes=256,
        ) == {"mode": "bounded"}

    def test_get_pr_content_preserves_statuses_when_check_runs_fail(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 8
        pr.title = "Keep independent CI evidence"
        pr.body = "No linked issue"
        pr.user.login = "author"
        pr.raw_data = {"author_association": "MEMBER"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha-456"
        pr.get_commits.return_value = []
        pr.get_reviews.return_value = []
        pr.get_review_comments.return_value = []

        failed_status = MagicMock()
        failed_status.context = "unit-tests"
        failed_status.state = "failure"
        failed_status.description = "tests failed"
        failed_status.target_url = "https://example.invalid/check"
        failed_status.created_at = datetime.now(timezone.utc)
        failed_status.updated_at = datetime.now(timezone.utc)
        head_commit = MagicMock()
        head_commit.get_statuses.return_value = [failed_status]
        head_commit.get_check_runs.side_effect = GithubException(
            503,
            {"message": "temporarily unavailable"},
        )
        mock_repo_object.get_commit.return_value = head_commit

        repo = Repository("owner/test-repo", mock_github_instance)
        result = repo.get_pr_content(8, pr=pr)

        assert result["ci_cd_results"]["statuses"][0]["state"] == "failure"
        assert result["ci_cd_results"]["check_runs"] == []
        assert result["_retrieval_meta"]["ci_statuses"]["outcome"] == "ok"
        assert result["_retrieval_meta"]["ci_check_runs"]["outcome"] == "error"
        assert result["_retrieval_meta"]["ci_check_runs"]["status_code"] == 503
        assert result["_retrieval_meta"]["ci_check_runs"]["error_type"] == "GithubException"
        assert result["_retrieval_meta"]["ci_aggregate"]["outcome"] == "partial"

    def test_get_pr_content_expands_issue_reference_from_pr_comment(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            items=[
                {
                    "id": 101,
                    "body": "Fixes #42",
                    "created_at": "2026-07-10T12:00:00Z",
                    "user": {"login": "maintainer"},
                    "author_association": "MEMBER",
                }
            ],
            outcome=RetrievalOutcome.OK,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 7
        pr.title = "Use the new parser"
        pr.body = "No issue reference in the PR body."
        pr.user.login = "author"
        pr.raw_data = {"author_association": "CONTRIBUTOR"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha"
        pr.get_reviews.return_value = []
        pr.get_review_comments.return_value = []
        pr.get_commits.return_value = []
        head_commit = MagicMock()
        head_commit.get_statuses.return_value = []
        head_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = head_commit

        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_issue_content = MagicMock(return_value="issue 42")

        result = repo.get_pr_content(7, pr=pr)

        assert result["related_issues"] == [
            {"issue_number": 42, "issue_content": "issue 42"}
        ]
        repo.get_issue_content.assert_called_once_with(42)

    def test_get_pr_content_preserves_review_and_each_inline_comment(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_github_instance.get_pr_files_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        mock_github_instance.get_pr_comments_with_status.return_value = RetrievalResult(
            outcome=RetrievalOutcome.NO_HIT,
            pages_fetched=1,
            status_code=200,
        )
        pr = MagicMock()
        pr.number = 8
        pr.title = "Preserve review context"
        pr.body = "No linked issue"
        pr.user.login = "author"
        pr.raw_data = {"author_association": "CONTRIBUTOR"}
        pr.created_at = datetime.now(timezone.utc)
        pr.updated_at = datetime.now(timezone.utc)
        pr.merged_at = None
        pr.state = "open"
        pr.base.ref = "main"
        pr.base.sha = "base-sha"
        pr.head.ref = "feature"
        pr.head.sha = "head-sha"
        pr.get_commits.return_value = []

        review = MagicMock()
        review.id = 501
        review.user.login = "reviewer"
        review.raw_data = {"author_association": "MEMBER"}
        review.body = "Overall review summary"
        review.state = "CHANGES_REQUESTED"
        review.submitted_at = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
        pr.get_reviews.return_value = [review]

        comments = []
        for comment_id, minute, body, path in (
            (601, 1, "First inline finding", "src/one.py"),
            (602, 2, "Second inline finding", "src/two.py"),
        ):
            comment = MagicMock()
            comment.id = comment_id
            comment.pull_request_review_id = review.id
            comment.user.login = "reviewer"
            comment.raw_data = {"author_association": "MEMBER"}
            comment.body = body
            comment.path = path
            comment.diff_hunk = "@@ -1 +1 @@"
            comment.created_at = datetime(
                2026, 7, 10, 12, minute, tzinfo=timezone.utc
            )
            comments.append(comment)
        pr.get_review_comments.return_value = comments

        head_commit = MagicMock()
        head_commit.get_statuses.return_value = []
        head_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = head_commit

        repo = Repository("owner/test-repo", mock_github_instance)
        result = repo.get_pr_content(8, pr=pr)

        assert [item["type"] for item in result["interactions"]] == [
            "review",
            "inline_comment",
            "inline_comment",
        ]
        assert [item["content"] for item in result["interactions"]] == [
            "Overall review summary",
            "First inline finding",
            "Second inline finding",
        ]
        assert [
            item.get("path") for item in result["interactions"]
        ] == [None, "src/one.py", "src/two.py"]

    def test_ci_status_helper_rejects_empty_head_sha(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        with pytest.raises(ValueError, match="head_sha is required"):
            repo.get_ci_status_with_status("")

    def test_ci_status_helper_returns_typed_error_for_missing_head(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        mock_repo_object.get_commit.side_effect = GithubException(
            404,
            {"message": "not found"},
        )
        repo = Repository("owner/test-repo", mock_github_instance)

        snapshot = repo.get_ci_status_with_status("missing-head")

        assert snapshot.outcome is RetrievalOutcome.ERROR
        assert snapshot.statuses == []
        assert snapshot.check_runs == []
        assert snapshot.statuses_meta["status_code"] == 404
        assert snapshot.check_runs_meta["error_type"] == "GithubException"

    def test_ci_status_helper_keeps_only_latest_state_per_status_context(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        old_failure = MagicMock(
            context="build",
            state="failure",
            description="old run failed",
            target_url="https://example.invalid/run/1",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        latest_success = MagicMock(
            context="build",
            state="success",
            description="rerun passed",
            target_url="https://example.invalid/run/2",
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        head_commit = MagicMock()
        head_commit.get_statuses.return_value = [old_failure, latest_success]
        head_commit.get_check_runs.return_value = []
        mock_repo_object.get_commit.return_value = head_commit
        repo = Repository("owner/test-repo", mock_github_instance)

        snapshot = repo.get_ci_status_with_status("head-sha")

        assert snapshot.state == "success"
        assert [item["state"] for item in snapshot.statuses] == ["success"]
        assert snapshot.statuses_meta["item_count"] == 2
        assert snapshot.retrieval_meta["ci_statuses"]["current_item_count"] == 1

    def test_related_issue_collection_makes_no_calls_for_zero_references(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_issue_content = MagicMock()

        contents = repo._collect_related_issue_contents(
            {"pr_metadata": {"description": "No linked issue"}},
            pr_number=7,
            max_issues=3,
        )

        assert contents == []
        repo.get_issue_content.assert_not_called()
        assert repo._retrieval_meta["related_issues"] == {
            "outcome": "no_hit",
            "discovered_count": 0,
            "eligible_count": 0,
            "item_count": 0,
            "attempted_count": 0,
            "successful_count": 0,
            "max_items": 3,
            "truncated": False,
            "excluded_current_pr": False,
            "error_type": None,
        }

    def test_related_issue_extraction_ignores_refs_outside_pr_conversation(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        issue_numbers = repo.extract_related_issues(
            {
                "pr_metadata": {
                    "title": "Refactor parser",
                    "description": "No linked issue",
                },
                "file_changes": [
                    {
                        "file_path": "src/parser.py",
                        "diff": "+ #123 is a data-format marker",
                    }
                ],
            }
        )

        assert issue_numbers == []

    def test_related_issue_extraction_keeps_bare_ref_in_long_pr_body(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        issue_numbers = repo.extract_related_issues(
            {
                "pr_metadata": {
                    "description": (
                        "This description intentionally contains enough context to "
                        "use the conservative matching path. "
                        + ("detail " * 30)
                        + "Additional context is available in #77."
                    )
                }
            }
        )

        assert issue_numbers == [77]

    @pytest.mark.parametrize(
        "description",
        [
            "See upstream/project#432 before merging.",
            "Upstream issue: https://github.com/upstream/project/issues/432",
            "[Upstream fix #432](https://github.com/upstream/project/pull/432)",
            (
                '<a href="https://redirect.github.com/upstream/project/pull/432">'
                "#432</a>"
            ),
        ],
    )
    def test_related_issue_extraction_ignores_cross_repo_scoped_references(
        self, mock_github_instance, mock_repo_object, description
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == []

    def test_related_issue_extraction_keeps_local_ref_next_to_external_release_notes(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        description = (
            "Fixes #77.\n"
            "<details><summary>Upstream release notes</summary>\n"
            "- [Improve parser (#431)](https://github.com/upstream/project/pull/431)\n"
            '<a href="https://redirect.github.com/upstream/project/issues/432">'
            "#432</a>\n"
            "</details>\n"
            "<details><summary>Upstream commits</summary>\n"
            '<a href="https://github.com/upstream/project/commit/abc123">abc123</a> '
            "Port PR 182675 to the release branch.\n"
            "</details>\n"
            + ("release detail " * 20)
        )

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == [77]

    def test_related_issue_extraction_keeps_refs_in_same_repo_details_block(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        description = (
            "<details><summary>Local migration notes</summary>\n"
            '<a href="https://github.com/owner/test-repo/blob/main/README.md">'
            "migration guide</a>\n"
            "Tracks #81.\n"
            "</details>"
        )

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == [81]

    def test_related_issue_extraction_keeps_local_hash_ref_in_external_details(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        description = (
            "<details><summary>Mixed maintainer notes</summary>\n"
            '<a href="https://github.com/upstream/project/releases">'
            "upstream release notes</a> mention issue 432.\n"
            "Fixes #77 in this repository.\n"
            "</details>"
        )

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == [77]

    def test_related_issue_extraction_scopes_short_unqualified_external_ref(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        description = (
            '<details><a href="https://github.com/upstream/project/releases">'
            "notes</a>: Fixes 432.</details> Fixes #77."
        )

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == [77]

    def test_related_issue_extraction_does_not_treat_app_link_as_repo_provenance(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        description = (
            "<details><summary>Automation notes</summary>\n"
            '<a href="https://github.com/apps/dependabot">Dependabot</a> '
            "tracks #82.\n"
            "</details>"
        )

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == [82]

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("owner/test-repo#41", [41]),
            (
                "[Local issue](https://github.com/owner/test-repo/issues/42)",
                [42],
            ),
            (
                '<a href="https://github.com/OWNER/TEST-REPO/pull/43">#43</a>',
                [43],
            ),
            ("https://github.com/owner/test-repo/issues/44", [44]),
        ],
    )
    def test_related_issue_extraction_keeps_same_repo_scoped_references(
        self, mock_github_instance, mock_repo_object, description, expected
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": description}}
        )

        assert issue_numbers == expected

    def test_related_issue_extraction_keeps_plain_same_repo_shorthand(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)

        issue_numbers = repo.extract_related_issues(
            {"pr_metadata": {"description": "Fixes #5 and relates to #2"}}
        )

        assert issue_numbers == [2, 5]

    def test_related_issue_collection_expands_all_references_under_limit(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_issue_content = MagicMock(
            side_effect=lambda number: f"issue {number}"
        )

        contents = repo._collect_related_issue_contents(
            {"pr_metadata": {"description": "Fixes #5 and relates to #2"}},
            pr_number=7,
            max_issues=3,
        )

        assert contents == [
            {"issue_number": 2, "issue_content": "issue 2"},
            {"issue_number": 5, "issue_content": "issue 5"},
        ]
        assert repo.get_issue_content.call_args_list == [call(2), call(5)]
        meta = repo._retrieval_meta["related_issues"]
        assert meta["outcome"] == "ok"
        assert meta["discovered_count"] == 2
        assert meta["eligible_count"] == 2
        assert meta["attempted_count"] == 2
        assert meta["successful_count"] == 2
        assert meta["truncated"] is False

    def test_related_issue_collection_excludes_pr_before_truncating(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_issue_content = MagicMock(
            side_effect=lambda number: f"issue {number}"
        )
        description = " ".join(f"Fixes #{number}" for number in range(1, 26))

        contents = repo._collect_related_issue_contents(
            {"pr_metadata": {"description": description}},
            pr_number=1,
            max_issues=3,
        )

        assert [item["issue_number"] for item in contents] == [2, 3, 4]
        assert repo.get_issue_content.call_args_list == [call(2), call(3), call(4)]
        meta = repo._retrieval_meta["related_issues"]
        assert meta["outcome"] == "partial"
        assert meta["discovered_count"] == 25
        assert meta["eligible_count"] == 24
        assert meta["item_count"] == 3
        assert meta["attempted_count"] == 3
        assert meta["successful_count"] == 3
        assert meta["max_items"] == 3
        assert meta["truncated"] is True
        assert meta["excluded_current_pr"] is True

    def test_related_issue_collection_uses_configured_default_limit(
        self, mock_github_instance, mock_repo_object
    ):
        mock_github_instance.get_repo.return_value = mock_repo_object
        repo = Repository("owner/test-repo", mock_github_instance)
        repo.get_issue_content = MagicMock(return_value="issue")
        description = " ".join(f"Fixes #{number}" for number in range(1, 26))

        contents = repo._collect_related_issue_contents(
            {"pr_metadata": {"description": description}},
            pr_number=99,
        )

        assert len(contents) == 20
        assert repo.get_issue_content.call_count == 20
        meta = repo._retrieval_meta["related_issues"]
        assert meta["discovered_count"] == 25
        assert meta["max_items"] == 20
        assert meta["truncated"] is True


class TestRepositoryPool:
    def test_pool_is_instance_scoped(self, mock_github_instance):
        pool1 = RepositoryPool(mock_github_instance)
        pool2 = RepositoryPool(mock_github_instance)

        try:
            assert pool1 is not pool2
        finally:
            pool1.stop_cleanup()
            pool2.stop_cleanup()

    def test_pool_can_disable_cleanup_thread(self, mock_github_instance):
        pool = RepositoryPool(mock_github_instance, cleanup_enabled=False)

        assert pool._cleanup_thread is None
        pool.close()

    def test_get_repository_caching(self, mock_github_instance, mock_repo_object):
        mock_github_instance.get_repo.return_value = mock_repo_object
        pool = RepositoryPool(mock_github_instance)

        try:
            repo1 = pool.get_repository("owner/test-repo")
            repo2 = pool.get_repository("owner/test-repo")

            assert repo1 is repo2
            assert len(pool._pool) == 1
        finally:
            pool.stop_cleanup()

    def test_cleanup_logic(self, mock_github_instance):
        pool = RepositoryPool(
            mock_github_instance,
            cleanup_interval=0.1,
            max_idle_time=0.1,
            cleanup_enabled=False,
        )

        mock_repo = MagicMock()
        mock_repo.last_read_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        mock_repo.creation_time = datetime(2000, 1, 1, tzinfo=timezone.utc)

        with pool._registry_lock:
            pool._pool["expired/repo"] = mock_repo
            pool._locks_registry["expired/repo"] = MagicMock()

        evicted = pool._cleanup_once(datetime(2025, 1, 1, tzinfo=timezone.utc))

        mock_repo.clear_cache.assert_called()
        assert evicted == 1
        assert "expired/repo" not in pool._pool
        assert "expired/repo" not in pool._locks_registry


class TestGitHubAPIHandler:
    def test_search_code_integration(self, mock_github_instance):
        pool = RepositoryPool(mock_github_instance, cleanup_enabled=False)
        handler = GitHubAPIHandler(mock_github_instance, pool=pool)

        mock_code_result = {
            "name": "test.py",
            "path": "test.py",
            "repository": {"full_name": "owner/repo"},
            "html_url": "http://url",
        }

        mock_github_instance.search_code.return_value = [mock_code_result]

        with patch.object(handler, "get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_file_content.return_value = "content"
            mock_repo.stargazers_count = 1
            mock_repo.watchers_count = 1
            mock_repo.language = "Python"
            mock_repo.description = "desc"
            mock_repo.updated_at = datetime.now(timezone.utc)
            mock_get_repo.return_value = mock_repo

            results = handler.search_code("query")

            assert len(results) == 1
            assert results[0]["content"] == "content"

    def test_code_search_expansion_failure_is_not_reported_as_ok(
        self, mock_github_instance
    ):
        pool = RepositoryPool(mock_github_instance, cleanup_enabled=False)
        handler = GitHubAPIHandler(mock_github_instance, pool=pool)
        mock_github_instance.search_code_with_status.return_value = RetrievalResult(
            items=[
                {
                    "name": "test.py",
                    "path": "test.py",
                    "repository": {"full_name": "owner/repo"},
                    "html_url": "https://example.invalid/test.py",
                }
            ],
            outcome=RetrievalOutcome.OK,
            pages_fetched=1,
            status_code=200,
        )

        with patch.object(
            handler,
            "_get_file_content_through_repository",
            return_value=(MagicMock(), None),
        ):
            result = handler.search_code_with_status("query")

        assert result.items == []
        assert result.outcome is RetrievalOutcome.PARTIAL
        assert result.error_type == "content_fetch_incomplete"

    def test_issue_search_expansion_failure_is_not_reported_as_ok(
        self, mock_github_instance
    ):
        pool = RepositoryPool(mock_github_instance, cleanup_enabled=False)
        handler = GitHubAPIHandler(mock_github_instance, pool=pool)
        mock_github_instance.search_issues_with_status.return_value = RetrievalResult(
            items=[
                {
                    "url": "https://api.github.com/repos/owner/repo/issues/1",
                    "body": "issue body",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ],
            outcome=RetrievalOutcome.OK,
            pages_fetched=1,
            status_code=200,
        )

        with patch.object(
            handler,
            "_get_issue_content_through_repository",
            return_value=None,
        ):
            result = handler.search_issues_with_status("query")

        assert result.items == []
        assert result.outcome is RetrievalOutcome.PARTIAL
        assert result.error_type == "content_fetch_incomplete"
