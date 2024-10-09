# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/JetXu-LLM/llama-github/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/JetXu-LLM/llama-github/releases/tag/v0.1.0