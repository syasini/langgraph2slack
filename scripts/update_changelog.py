#!/usr/bin/env python3
"""
Update CHANGELOG.md to add a new version section.

This script:
1. Finds the [Unreleased] section
2. Creates a new version section [X.Y.Z] - YYYY-MM-DD
3. Copies content from [Unreleased] to the new version
4. Clears [Unreleased] section

Usage:
    python scripts/update_changelog.py 0.1.5
"""

import sys
import re
from datetime import date
from pathlib import Path


def update_changelog(version: str, changelog_path: Path = Path("CHANGELOG.md")):
    """Update CHANGELOG.md with new version section.

    Args:
        version: Version number (e.g., "0.1.5")
        changelog_path: Path to CHANGELOG.md file

    Returns:
        bool: True if successful, False otherwise
    """
    if not changelog_path.exists():
        print(f"Error: {changelog_path} not found")
        return False

    # Read current changelog
    content = changelog_path.read_text()

    # Check if [Unreleased] section exists
    if "## [Unreleased]" not in content:
        print("Error: No [Unreleased] section found in CHANGELOG.md")
        return False

    # Get today's date
    today = date.today().isoformat()  # YYYY-MM-DD format

    # Find the [Unreleased] section and extract its content
    # Match from ## [Unreleased] to the next ## [ or end of file
    unreleased_pattern = r"## \[Unreleased\](.*?)(?=\n## \[|\Z)"
    match = re.search(unreleased_pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not parse [Unreleased] section")
        return False

    unreleased_content = match.group(1).strip()

    # Check if there's any content in [Unreleased]
    # (excluding empty lines and common headers)
    content_lines = [
        line for line in unreleased_content.split('\n')
        if line.strip() and not line.strip().startswith('###')
    ]

    if not content_lines:
        print("Warning: [Unreleased] section is empty")
        print("Please add changes to CHANGELOG.md before releasing")
        return False

    # Create new version section
    new_version_section = f"## [{version}] - {today}\n{unreleased_content}"

    # Replace [Unreleased] section with:
    # 1. Empty [Unreleased] section
    # 2. New version section
    replacement = f"## [Unreleased]\n\n{new_version_section}"

    # Perform the replacement
    updated_content = re.sub(
        unreleased_pattern,
        replacement,
        content,
        count=1,  # Only replace first occurrence
        flags=re.DOTALL
    )

    # Write back to file
    changelog_path.write_text(updated_content)

    print(f"✅ Updated CHANGELOG.md:")
    print(f"   Created section: [{version}] - {today}")
    print(f"   Moved content from [Unreleased]")

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_changelog.py <version>")
        print("Example: python scripts/update_changelog.py 0.1.5")
        sys.exit(1)

    version = sys.argv[1]

    # Validate version format (basic check)
    if not re.match(r'^\d+\.\d+\.\d+$', version):
        print(f"Error: Invalid version format: {version}")
        print("Expected format: X.Y.Z (e.g., 0.1.5)")
        sys.exit(1)

    success = update_changelog(version)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
