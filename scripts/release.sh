#!/bin/bash
# Release script for lg2slack
# Usage: ./scripts/release.sh [patch|minor|major]
#
# This script will:
# 1. Run tests locally to ensure everything works
# 2. Validate you're on main branch with clean state
# 3. Bump the version using `uv version --bump`
# 4. Update CHANGELOG.md ([Unreleased] -> [X.Y.Z])
# 5. Commit the changes (pyproject.toml + CHANGELOG.md)
# 6. Create and push a git tag
# 7. GitHub Actions will automatically create a release and publish to PyPI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print header
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   lg2slack Release Script${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Check if bump type provided
if [ -z "$1" ]; then
  echo -e "${RED}Error: Missing version bump type${NC}"
  echo ""
  echo "Usage: $0 [patch|minor|major]"
  echo ""
  echo "Examples:"
  echo "  $0 patch   # 0.1.4 -> 0.1.5 (bug fixes)"
  echo "  $0 minor   # 0.1.4 -> 0.2.0 (new features)"
  echo "  $0 major   # 0.1.4 -> 1.0.0 (breaking changes)"
  exit 1
fi

BUMP_TYPE=$1

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
  echo -e "${RED}Error: Invalid bump type '$BUMP_TYPE'${NC}"
  echo "Must be one of: patch, minor, major"
  exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo -e "${RED}Error: uv is not installed${NC}"
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}Error: python3 is not installed${NC}"
  exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
  echo -e "${RED}Error: pyproject.toml not found${NC}"
  echo "Run this script from the project root or scripts directory"
  exit 1
fi

# Check if on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo -e "${RED}Error: You must be on 'main' branch to release${NC}"
  echo "Current branch: $CURRENT_BRANCH"
  echo ""
  echo "Switch to main with: git checkout main"
  exit 1
fi

# Check if there are uncommitted changes
if ! git diff --quiet || ! git diff --quiet --cached; then
  echo -e "${RED}Error: You have uncommitted changes${NC}"
  echo "Please commit or stash them before releasing"
  echo ""
  git status --short
  exit 1
fi

# Check if local main is up to date with remote
git fetch origin main
LOCAL=$(git rev-parse main)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
  echo -e "${YELLOW}Warning: Your local 'main' branch is not in sync with origin/main${NC}"
  echo ""
  echo "Local:  $LOCAL"
  echo "Remote: $REMOTE"
  echo ""
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run: git pull origin main"
    exit 1
  fi
fi

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
  echo -e "${RED}Error: CHANGELOG.md not found${NC}"
  echo "Please create CHANGELOG.md with an [Unreleased] section"
  exit 1
fi

echo -e "${GREEN}Step 1/7: Run tests${NC}"
echo "Running test suite to ensure everything works..."
if uv run python3 -m pytest tests/ -v --tb=short; then
  echo -e "${GREEN}✅ All tests passed!${NC}"
  echo ""
else
  echo -e "${RED}❌ Tests failed!${NC}"
  echo "Please fix failing tests before releasing"
  exit 1
fi

echo -e "${GREEN}Step 2/7: Get current version${NC}"
# Get current version
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"
echo ""

echo -e "${GREEN}Step 3/7: Bump version ($BUMP_TYPE)${NC}"
# Bump version using uv
uv version --bump $BUMP_TYPE

# Get new version
NEW_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "New version: $NEW_VERSION"
echo ""

echo -e "${GREEN}Step 4/7: Update CHANGELOG.md${NC}"
# Update CHANGELOG.md using Python script
if python3 "$SCRIPT_DIR/update_changelog.py" "$NEW_VERSION"; then
  echo ""
else
  echo -e "${RED}Failed to update CHANGELOG.md${NC}"
  echo "Reverting version bump..."
  git restore pyproject.toml uv.lock
  exit 1
fi

# Show what changed in CHANGELOG
echo -e "${BLUE}Preview of CHANGELOG.md changes:${NC}"
git diff CHANGELOG.md | head -30
echo ""

echo -e "${GREEN}Step 5/7: Confirm release${NC}"
echo -e "${YELLOW}This will:${NC}"
echo "  1. Commit version bump: $CURRENT_VERSION -> $NEW_VERSION"
echo "  2. Commit CHANGELOG.md update"
echo "  3. Create tag: v$NEW_VERSION"
echo "  4. Push to GitHub (branch: main)"
echo "  5. Trigger automated release and PyPI publish"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Release cancelled"
  echo "Reverting changes..."
  git restore pyproject.toml uv.lock CHANGELOG.md
  echo "✅ All changes reverted"
  exit 0
fi
echo ""

echo -e "${GREEN}Step 6/7: Commit and tag${NC}"
# Commit version bump and changelog
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v$NEW_VERSION

- Bump version to $NEW_VERSION
- Update CHANGELOG.md"

# Create tag
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
echo "Created tag: v$NEW_VERSION"
echo ""

echo -e "${GREEN}Step 7/7: Push to GitHub${NC}"
# Push changes and tag
git push origin main
git push origin "v$NEW_VERSION"
echo ""

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Release v$NEW_VERSION initiated!${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""
echo "GitHub Actions will now:"
echo "  1. Run tests"
echo "  2. Create GitHub release (using CHANGELOG.md)"
echo "  3. Publish to PyPI"
echo ""
echo "Check progress at:"
echo -e "  ${BLUE}https://github.com/syasini/lg2slack/actions${NC}"
echo ""
echo "The release will appear at:"
echo -e "  ${BLUE}https://github.com/syasini/lg2slack/releases/tag/v$NEW_VERSION${NC}"
echo ""
echo "PyPI package (after publish completes):"
echo -e "  ${BLUE}https://pypi.org/project/lg2slack/$NEW_VERSION/${NC}"
echo ""
