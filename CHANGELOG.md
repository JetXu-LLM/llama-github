# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
[0.1.3]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/JetXu-LLM/llama-github/releases/tag/v0.1.0
