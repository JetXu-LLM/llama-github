#!/usr/bin/env python3
"""Fail a release when a wheel contains stale or unexplained package files."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


PACKAGE_ROOT = "llama_github/"
ALLOWED_DATA_FILES = {"llama_github/config/config.json"}


def expected_package_files(source_root: Path) -> set[str]:
    package = source_root / "llama_github"
    expected = {
        path.relative_to(source_root).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    expected.update(
        name for name in ALLOWED_DATA_FILES if (source_root / name).is_file()
    )
    return expected


def verify_wheel(wheel_path: Path, *, source_root: Path) -> None:
    expected = expected_package_files(source_root)
    with zipfile.ZipFile(wheel_path) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}

    package_files = {name for name in names if name.startswith(PACKAGE_ROOT)}
    metadata_files = {
        name
        for name in names
        if ".dist-info/" in name and not name.startswith(PACKAGE_ROOT)
    }
    unexpected_top_level = names - package_files - metadata_files
    unexpected_package = package_files - expected
    missing_package = expected - package_files
    suspicious_names = {
        name
        for name in package_files
        if any(character.isspace() for character in Path(name).name)
        or Path(name).suffix.lower() in {".so", ".dylib", ".dll", ".pyd"}
    }

    problems = []
    if unexpected_package:
        problems.append(f"unexpected package files: {sorted(unexpected_package)}")
    if missing_package:
        problems.append(f"missing package files: {sorted(missing_package)}")
    if unexpected_top_level:
        problems.append(f"unexpected top-level files: {sorted(unexpected_top_level)}")
    if suspicious_names:
        problems.append(f"suspicious package filenames: {sorted(suspicious_names)}")
    if problems:
        raise ValueError(f"{wheel_path.name}: " + "; ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    for wheel in args.wheels:
        verify_wheel(wheel.resolve(), source_root=args.source_root.resolve())
        print(f"verified wheel contents: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
