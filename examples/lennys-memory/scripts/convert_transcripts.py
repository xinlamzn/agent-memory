#!/usr/bin/env python3
"""Convert Lenny's Data markdown transcripts to the expected .txt format.

Input format (markdown with YAML frontmatter):
    ---
    title: "..."
    guest: "Stewart Butterfield"
    ---

    **Stewart Butterfield** (00:00:00):
    Content here...

Output format (.txt):
    Stewart Butterfield (00:00:00):
    Content here...
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


def extract_guest_name(content: str) -> Optional[str]:
    """Extract guest name from YAML frontmatter."""
    match = re.search(r'^guest:\s*"(.+?)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def convert_transcript(md_content: str) -> str:
    """Convert markdown transcript to plain text format.

    Strips YAML frontmatter and removes ** bold markers from speaker names.
    """
    # Remove YAML frontmatter
    if md_content.startswith("---"):
        end = md_content.find("---", 3)
        if end != -1:
            md_content = md_content[end + 3:].strip()

    # Remove ** bold markers around speaker names
    # Pattern: **Speaker Name** (HH:MM:SS): -> Speaker Name (HH:MM:SS):
    md_content = re.sub(r"\*\*(.+?)\*\*\s*(\(\d{2}:\d{2}:\d{2}\):)", r"\1 \2", md_content)

    return md_content


def get_output_filename(md_path: Path, content: str) -> str:
    """Determine output filename from guest name or markdown filename."""
    guest = extract_guest_name(content)
    if guest:
        return f"{guest}.txt"
    # Fallback: convert slug to title case
    name = md_path.stem.replace("-", " ").title()
    return f"{name}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lenny's Data markdown transcripts to .txt format"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing .md transcript files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write .txt files to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing files",
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Error: Source directory not found: {args.source_dir}")
        sys.exit(1)

    md_files = sorted(args.source_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {args.source_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        output_name = get_output_filename(md_path, content)
        output_path = args.output_dir / output_name

        if args.dry_run:
            print(f"  {md_path.name} -> {output_name}")
            converted += 1
            continue

        if output_path.exists():
            skipped += 1
            continue

        txt_content = convert_transcript(content)
        output_path.write_text(txt_content, encoding="utf-8")
        converted += 1

    print(f"Converted: {converted}, Skipped: {skipped}, Total: {len(md_files)}")


if __name__ == "__main__":
    main()
