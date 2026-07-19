# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5] - 2026-07-20

### Added
- Added optional per-file and cumulative source limits to `get_pr_content()` for memory-bounded consumers while preserving the historical unbounded default and a separate cache identity
- Added an optional caller-owned source limit to `get_file_content()`, including size-before-decode checks and locally capped raw-response streaming
- Added content-free PR retrieval metadata for retained reads, budget exhaustion, API-patch fallback, and missing-patch fallback

### Fixed
- Prevented large changed-file base/head content from being materialized or accumulated before callers can apply their own memory boundary
- Reused GitHub's changed-file patch when bounded source is unavailable instead of treating the file as analyzable full source

## [0.4.4] - 2026-07-19

### Fixed
- Bound explicit GitHub issue and pull-request references to their named repository before applying same-repository `#123` shorthand matching
- Prevented external Markdown/HTML links and unqualified issue phrases in repository-scoped embedded release-note blocks, including Dependabot-generated notes, from being expanded as unrelated local issues

## [0.4.3] - 2026-07-15

### Fixed
- Aligned the `dependency_lock` opt-in with the complete standard lockfile family used by bounded review planning, including `go.sum`, `gradle.lockfile`, `packages.lock.json`, and ecosystem peers
- Aligned the `ci_config` opt-in with the existing bounded CI path family, including AppVeyor configuration

## [0.4.2] - 2026-07-15

### Added
- Added a typed, fixed-cap repository text-read API with distinct success, policy, missing, oversize, binary, directory, and transport outcomes
- Added explicit `dependency_lock` and `ci_config` opt-ins without changing the legacy `get_file_content()` exclusion or cache contract

### Security
- Enforced the 2 MiB product cap both from GitHub size metadata and while streaming raw bytes
- Kept binary, generated, minified, and sensitive paths outside the new opt-in surface
- Raised the tested `setuptools` and `transformers` lower bounds to versions that clear the current dependency-audit security gate

## [0.4.1] - 2026-07-12

### Added
- Added status-aware code/issue search APIs that distinguish `ok`, `no_hit`, `partial`, and `error`
- Added bounded pagination metadata for PR files and comments, plus the captured PR `head_sha`
- Added exact-head, bounded CI refresh with independently typed status/check-run evidence
- Added explicit repository-pool shutdown and an option to disable its cleanup worker

### Changed
- Made the public `GithubRAG` import lazy so retrieval-only consumers do not import the ML/RAG stack
- Added connect/read timeouts, bounded REST pagination, privacy-safe logging, and immutable revisions for built-in Jina models
- Bounded related-issue extraction and expansion with configurable limits and truncation metadata
- Limited related-issue discovery to PR title/body and top-level PR comments, while retaining bare `#123` references in long descriptions
- Preserved review summaries and every inline review comment as separate chronological interactions
- Reduced commit-status history to the newest result per logical context while preserving fetched/current counts
- Raised dependency security floors while keeping the full install and public high-level RAG API compatible
- Hardened release and secret-scanning workflows with pinned actions, full-history scanning, checksum verification, and job-scoped PyPI OIDC
- Added a release artifact allowlist that rejects stale or unexplained files in the built wheel

### Fixed
- Prevented Python AST parsing from being selected for non-Python paths based only on repository language metadata
- Fixed repository-pool cleanup so expired entries are actually evicted instead of retaining empty objects
- Fixed `get_pr_content()` so already-fetched PR comments participate in related-issue discovery instead of being processed too late

## [0.4.0] - 2026-03-25

### Changed
- Modernized packaging to `pyproject.toml` and raised the runtime floor to Python 3.10+
- Reworked LLM and GitHub integration layers to reduce stale compatibility paths and deprecated APIs
- Updated repository retrieval docs to reflect the actual return shapes and maintenance-focused project positioning

### Fixed
- Corrected `simple_mode` behavior so it no longer depends on embedding and reranker initialization
- Fixed message composition order for chat history vs context handling
- Stabilized test execution and added runnable mock/real examples

## [0.3.3] - 2025-08-24

### Optimized
- Enhance diff hunk header

## [0.3.2] - 2025-06-23

### Optimized
- Upgrade to extract_related_issues method

## [0.3.1] - 2025-05-25

### Optimized
- Upgrade to mistral medium & devstral small

## [0.2.8] - 2025-03-19

### Optimized
- Upgrade to mistral-small-latest

## [0.2.3] - 2024-11-24

### Optimized
- Upgrade to mistral-large-2411

## [0.2.2] - 2024-11-19

### Optimized
- more precise model specification (stick to mistral-large-2407 and mistral-nemo-latest)

## [0.2.1] - 2024-11-16

### Optimized
- approperately handle more file types when calculate file changes in PR

## [0.2.0] - 2024-11-16

### Optimized
- fix bugs for generate repo from pool by using Github_install_id

## [0.1.9] - 2024-11-04

### Optimized
- fix bugs for get pr content

## [0.1.8] - 2024-11-03

### Optimized
- fix bugs for get pr content file diff calculate logic

## [0.1.7] - 2024-10-31

### Optimized
- fix bugs for get pr content

## [0.1.6] - 2024-10-30

### New Features
- Enhanced PR content analysis with detailed commit information extraction
- Improved issue linking detection with support for multiple reference formats
  - Full GitHub URLs, #references, and keyword-based references
  - Added validation for issue numbers

### Improvements
- Added detailed commit metadata extraction including stats and file changes
- Enhanced error handling for commit fetching

## [0.1.5] - 2024-10-14

### Optimized
- requirements.txt updated to more precise list

## [0.1.4] - 2024-10-14

### Improved
- Optimized `simple_mode`:
  - Removed dependencies on `Torch` and `Transformers` libraries
  - Reduced memory footprint
  - Eliminated related imports
  - Enhanced compatibility with AWS Lambda environment

## [0.1.3] - 2024-10-14

### Added
- Modified `LLMManager` class to skip loading embedding and reranker models when `simple_mode` is enabled
- Updated `retrieve_context` method to use instance's `simple_mode` by default, with option to override

### Improved
- Faster initialization process when `simple_mode` is enabled, skipping embedding and reranker model loading
- More flexible usage of `simple_mode` in `retrieve_context`, allowing per-call customization

### Developer Notes
- When using `simple_mode=True` during GithubRAG initialization, be aware that embedding and reranking functionalities will not be available
- The `retrieve_context` method now uses a late binding approach for `simple_mode` parameter

## [0.1.2] - 2024-10-09

### Added
- New `get_pr_content` method in `Repository` class for comprehensive PR data retrieval
- Singleton pattern implementation for efficient PR data caching
- Support for LLM-assisted PR analysis and Q&A capabilities
- Automatic caching mechanism to reduce API calls and improve performance
- Threaded comment and review retrieval functionality

### Changed
- Improved PR data fetching process to include metadata, file changes, and comments

### Optimized
- Reduced API calls through intelligent caching of PR data

### Developer Notes
- First call to `get_pr_content` fetches data from GitHub API, subsequent calls use cached data
- Cache automatically refreshes when PR is updated

## [0.1.1] - 2024-08-23

### Added
- Implemented `answer_with_context` method for direct answer generation (closes #6)
- Added support for Mistral AI LLM provider
- Enhanced `retrieve_context` function to include metadata (e.g., URLs) with each context string (closes #2)

### Changed
- Improved reranking with jina-reranker-v2 for better context retrieval
- Updated return type of `retrieve_context` to accommodate metadata

### Fixed
- Resolved warning during context retrieval (closes #3)

### Improved
- Enhanced overall context retrieval process
- Expanded LLM support for more versatile use cases

## [0.1.0] - 2024-08-15

### Added
- Initial release of llama-github
- Basic functionality for retrieving context from GitHub repositories
- Integration with LLM for processing and generating responses

[0.1.4]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.3...v0.1.4
[0.4.5]: https://github.com/JetXu-LLM/llama-github/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/JetXu-LLM/llama-github/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/JetXu-LLM/llama-github/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/JetXu-LLM/llama-github/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/JetXu-LLM/llama-github/compare/v0.4.0...v0.4.1
[0.1.3]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/JetXu-LLM/llama-github/releases/tag/v0.1.0
