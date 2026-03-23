#!/usr/bin/env python3
"""Convert Lenny's Data newsletter markdown files to the transcript .txt format.

Newsletters are prose articles, not dialogues. This script converts them into
the speaker-turn format expected by load_transcripts.py by:
- Extracting the title and author from YAML frontmatter
- Splitting content into meaningful chunks (sections/paragraphs)
- Assigning each chunk as a turn from the author with synthetic timestamps
- Stripping markdown images and formatting artifacts

Input format (markdown with YAML frontmatter):
    ---
    title: "How Duolingo reignited user growth"
    date: "2023-02-28"
    type: "newsletter"
    ---
    Content here...

Output format (.txt):
    Lenny Rachitsky (00:00:00):
    How Duolingo reignited user growth

    Lenny Rachitsky (00:00:30):
    First paragraph of content...
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


def extract_frontmatter(content: str) -> tuple:
    """Extract YAML frontmatter fields and remaining content."""
    title = None
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end]
            content = content[end + 3:].strip()

            match = re.search(r'^title:\s*"(.+?)"', frontmatter, re.MULTILINE)
            if match:
                title = match.group(1)

    return title, content


def clean_markdown(text: str) -> str:
    """Remove markdown artifacts that don't make sense in plain text."""
    # Remove image lines
    text = re.sub(r'!\[.*?\]\(.*?\)\n?', '', text)
    # Remove standalone links that are just URLs (keep inline links text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Remove heading markers but keep text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_chunks(content: str) -> list:
    """Split newsletter content into meaningful chunks.

    Groups content by sections (headings) or merges short consecutive paragraphs.
    Each chunk becomes one speaker turn.
    """
    content = clean_markdown(content)

    # Split on double newlines (paragraph boundaries)
    raw_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Merge short paragraphs together, split on section-like boundaries
    chunks = []
    current_chunk = []
    current_len = 0

    for para in raw_paragraphs:
        # Skip very short lines that are likely artifacts
        if len(para) < 20 and not para.endswith(':'):
            if current_chunk:
                current_chunk.append(para)
                current_len += len(para)
            continue

        # Start new chunk if this paragraph is long enough on its own
        # or if accumulating would exceed a reasonable size
        if current_len + len(para) > 1500 and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_len = len(para)
        else:
            current_chunk.append(para)
            current_len += len(para)

        # If current chunk is substantial, flush it
        if current_len > 800:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_len = 0

    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def format_timestamp(seconds: int) -> str:
    """Format seconds as HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert_newsletter(md_content: str) -> Optional[str]:
    """Convert a newsletter markdown file to transcript format."""
    title, content = extract_frontmatter(md_content)
    if not content:
        return None

    chunks = split_into_chunks(content)
    if not chunks:
        return None

    author = "Lenny Rachitsky"
    lines = []
    timestamp_seconds = 0

    # First turn: the title
    if title:
        lines.append(f"{author} ({format_timestamp(timestamp_seconds)}):")
        lines.append(title)
        lines.append("")
        timestamp_seconds += 30

    # Each chunk becomes a speaker turn
    for chunk in chunks:
        lines.append(f"{author} ({format_timestamp(timestamp_seconds)}):")
        lines.append(chunk)
        lines.append("")
        # Estimate ~30 seconds per chunk (reading time)
        timestamp_seconds += 30

    return '\n'.join(lines)


def get_output_filename(md_path: Path, md_content: str) -> str:
    """Determine output filename from title or filename."""
    title, _ = extract_frontmatter(md_content)
    if title:
        # Clean title for filename
        clean = re.sub(r'[^\w\s-]', '', title)
        clean = clean.strip()[:80]  # Truncate long titles
        return f"{clean}.txt"
    # Fallback: convert slug to title case
    name = md_path.stem.replace("-", " ").title()
    return f"{name}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lenny's newsletter markdown files to transcript .txt format"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing .md newsletter files",
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
            title, body = extract_frontmatter(content)
            chunks = split_into_chunks(clean_markdown(body))
            print(f"  {md_path.name} -> {output_name} ({len(chunks)} turns)")
            converted += 1
            continue

        if output_path.exists():
            skipped += 1
            continue

        txt_content = convert_newsletter(content)
        if txt_content:
            output_path.write_text(txt_content, encoding="utf-8")
            converted += 1
        else:
            print(f"  Warning: Could not convert {md_path.name}")

    print(f"Converted: {converted}, Skipped: {skipped}, Total: {len(md_files)}")


if __name__ == "__main__":
    main()
