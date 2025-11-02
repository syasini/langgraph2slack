# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2025-11-01
### Added
- Automated release system with `scripts/release.sh` (supports patch/minor/major)
- Python script `scripts/update_changelog.py` to automatically update CHANGELOG.md
- GitHub Actions workflow `.github/workflows/release.yml` to create releases on tag push
- CHANGELOG.md following Keep a Changelog format
- Automated test running in CI before creating releases

### Fixed
- README.md image URLs now use absolute GitHub URLs for proper PyPI display
- README.md example links now point to GitHub repository
- Added missing `langsmith>=0.1.0` dependency (required for feedback tracking)

### Changed
- Release process now fully automated with version bumping, changelog updates, and PyPI publishing
## [0.1.4] - 2025-11-01

### Added
- Comprehensive test suite with 141 tests (96%+ coverage of core logic)
  - `test_utils.py`: 69 tests for markdown, images, event detection, feedback
  - `test_config.py`: 30 tests for MessageContext and BotConfig validation
  - `test_transformers.py`: 17 tests for async transformer chains
  - `test_handlers_base.py`: 25 tests for thread ID generation and block creation
- Test infrastructure: `pytest.ini`, `conftest.py` with 14 fixtures
- Added `aiohttp>=3.8.0` to dependencies (required by slack-bolt async support)

### Fixed
- **CRITICAL**: Wikipedia URLs with parentheses now parse correctly
  - Example: `https://en.wikipedia.org/wiki/File:Monstera_(plant).jpg`
  - Implemented balanced parentheses regex pattern: `(?:[^()]|\([^()]*\))+`
  - Affects both `clean_markdown()` and `extract_markdown_images()`
- Empty link text in markdown (`[](url)`) now handled correctly
  - Changed regex from `[^\]]+` to `[^\]]*` to allow zero characters
- Multiple images on same line now extract properly
  - Switched from greedy match to `re.findall()` with balanced parentheses pattern

### Changed
- Updated README with absolute GitHub URLs for images (fixes PyPI display)
- Simplified plant_bot example README
- Enhanced documentation with demo GIFs and setup screenshots

## [0.1.3] - 2024-10-19

### Added
- GitHub Actions workflow for automated PyPI publishing
- Dynamic version tracking using `importlib.metadata.version()`

### Changed
- Updated README with comprehensive documentation
- Updated plant_bot agent implementation

## [0.1.2] - 2024-10-14

### Added
- Optional message type processing
- Processing emoji reactions for user feedback
- Metadata passing to LangGraph runs

### Fixed
- No comment feedback bug
- Metadata and image extraction features
- Double streaming issue
- Transformation error for non-streaming mode
- Markdown rendering issues

### Changed
- Updated system prompt to avoid double streaming

## [0.1.1] - 2024-10-13

### Added
- plant_bot example project demonstrating full integration
- `.env.example` and `slack_manifest.yaml` templates
- Comprehensive README documentation

## [0.1.0] - 2024-10-13

### Added
- Initial release of lg2slack package
- Core SlackBot class for LangGraph-to-Slack integration
- Streaming and non-streaming message handlers
- Markdown to Slack mrkdwn conversion
- Image extraction from markdown and rendering as Slack blocks
- Feedback buttons with LangSmith integration
- Thread ID management using deterministic UUID5
- Input/output transformer chains
- MessageContext for passing Slack metadata
- BotConfig with environment variable loading via pydantic-settings
- Support for DMs, channel mentions, and thread participation

[Unreleased]: https://github.com/syasini/lg2slack/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/syasini/lg2slack/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syasini/lg2slack/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syasini/lg2slack/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syasini/lg2slack/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syasini/lg2slack/releases/tag/v0.1.0
