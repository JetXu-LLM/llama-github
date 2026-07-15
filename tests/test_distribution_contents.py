from pathlib import Path
import zipfile

import pytest

from scripts.verify_distribution_contents import (
    expected_package_files,
    verify_wheel,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_wheel(path: Path, *, extra_files=()) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(expected_package_files(ROOT)):
            archive.writestr(name, b"")
        archive.writestr("llama_github-0.4.2.dist-info/METADATA", b"")
        for name in extra_files:
            archive.writestr(name, b"")


def test_wheel_content_verifier_accepts_exact_source_package(tmp_path):
    wheel = tmp_path / "llama_github-0.4.2-py3-none-any.whl"
    _write_wheel(wheel)

    verify_wheel(wheel, source_root=ROOT)


def test_wheel_content_verifier_rejects_stale_duplicate_module(tmp_path):
    wheel = tmp_path / "llama_github-0.4.2-py3-none-any.whl"
    _write_wheel(
        wheel,
        extra_files=("llama_github/data_retrieval/github_entities 2.py",),
    )

    with pytest.raises(ValueError, match="unexpected package files"):
        verify_wheel(wheel, source_root=ROOT)
