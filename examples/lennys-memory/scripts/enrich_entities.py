#!/usr/bin/env python3
"""Enrich entities with Wikipedia data.

This script enriches Entity nodes in Memory Store with data from Wikipedia/Wikimedia:
- Description/summary from Wikipedia
- Wikipedia URL
- Wikidata ID
- Thumbnail image URL

Features:
- Uses Memory Store API (query_nodes, update_node) for data access
- Real-time progress bars with ETA
- Configurable rate limiting (respects Wikimedia ToS)
- Skip already-enriched entities
- Deduplicates by entity name (enrich once per unique name)
- Filter by entity type
- Detailed statistics on completion
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from backend directory
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    MemoryClient,
    MemorySettings,
    MemoryStoreConfig,
)
from neo4j_agent_memory.enrichment.wikimedia import WikimediaProvider
from neo4j_agent_memory.enrichment.base import EnrichmentStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"


def supports_color() -> bool:
    """Check if terminal supports colors."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLORS = supports_color()


def color(text: str, color_code: str) -> str:
    """Apply color to text if supported."""
    if USE_COLORS:
        return f"{color_code}{text}{Colors.RESET}"
    return text


@dataclass
class EnrichmentStats:
    """Statistics for enrichment run."""
    total: int = 0
    enriched: int = 0
    not_found: int = 0
    skipped: int = 0
    errors: int = 0
    rate_limited: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.enriched + self.not_found + self.errors
        if processed == 0:
            return 0.0
        return self.enriched / processed * 100


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_progress(
    current: int,
    total: int,
    stats: EnrichmentStats,
    start_time: float,
    current_entity: str = "",
) -> None:
    """Print progress bar with stats."""
    if total == 0:
        return

    # Calculate progress
    progress = current / total
    bar_width = 30
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Calculate ETA
    elapsed = time.time() - start_time
    if current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = format_time(eta)
    else:
        eta_str = "calculating..."

    # Truncate entity name
    entity_display = current_entity[:30] + "..." if len(current_entity) > 30 else current_entity

    # Build status line
    status = (
        f"\r{color('Progress', Colors.CYAN)} [{bar}] "
        f"{current}/{total} ({progress*100:.1f}%) "
        f"ETA: {eta_str} "
        f"| {color('✓', Colors.GREEN)}{stats.enriched} "
        f"{color('✗', Colors.RED)}{stats.not_found} "
        f"{color('!', Colors.YELLOW)}{stats.errors} "
        f"| {entity_display:<35}"
    )

    print(status, end="", flush=True)


def _get_graph_backend(client: MemoryClient):
    """Access the graph backend from a MemoryClient."""
    return client._bundle.graph


async def get_all_entities(client: MemoryClient) -> list[dict]:
    """Load all Entity nodes from Memory Store via query_nodes.

    The Memory Store API caps top_k at 10,000. To get all entities we
    query in batches by entity type.

    Returns:
        List of entity dicts with id, name, type, and any enrichment properties.
    """
    graph = _get_graph_backend(client)

    # First try a single large query (works if <=10K entities)
    entities = await graph.query_nodes("Entity", limit=10000)

    if len(entities) < 10000:
        return entities

    # If we hit the cap, query by entity type to get them all
    logger.info("Hit 10K cap, loading entities by type...")
    seen_ids: set[str] = set()
    all_entities: list[dict] = []

    entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "OBJECT", "EVENT",
                    "CONCEPT", "PRODUCT", "TECHNOLOGY", "WORK_OF_ART"]

    for etype in entity_types:
        batch = await graph.query_nodes("Entity", filters={"type": etype}, limit=10000)
        for e in batch:
            eid = e.get("id", "")
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_entities.append(e)
        if batch:
            logger.info(f"  {etype}: {len(batch)} entities")

    # Also grab any entities without a type or with unusual types
    remaining = await graph.query_nodes("Entity", limit=10000)
    for e in remaining:
        eid = e.get("id", "")
        if eid not in seen_ids:
            seen_ids.add(eid)
            all_entities.append(e)

    return all_entities


def filter_unenriched_entities(
    all_entities: list[dict],
    entity_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Filter entities that haven't been enriched yet.

    Args:
        all_entities: Pre-loaded list of all entities
        entity_types: Filter by entity types (e.g., ["PERSON", "ORGANIZATION"])
        limit: Maximum number of entities to return

    Returns:
        List of entity dicts with id, name, type (deduplicated by name)
    """
    # Filter for unenriched entities
    unenriched = []
    for e in all_entities:
        if e.get("enriched_description") or e.get("enrichment_error"):
            continue
        if entity_types:
            if e.get("type", "").upper() not in [t.upper() for t in entity_types]:
                continue
        unenriched.append(e)

    # Deduplicate by name - keep first occurrence per unique name
    seen_names: set[str] = set()
    deduped = []
    for e in sorted(unenriched, key=lambda x: x.get("name", "")):
        name = e.get("name", "").strip().lower()
        if name and name not in seen_names:
            seen_names.add(name)
            deduped.append(e)

    if limit:
        deduped = deduped[:limit]

    return deduped


def compute_enrichment_status(all_entities: list[dict]) -> dict:
    """Compute enrichment status from a pre-loaded entity list.

    Returns:
        Dict with counts: total, enriched, pending, errors
    """
    total = len(all_entities)
    enriched = sum(1 for e in all_entities if e.get("enriched_description"))
    errors = sum(1 for e in all_entities if e.get("enrichment_error"))
    pending = total - enriched - errors

    return {
        "total": total,
        "enriched": enriched,
        "pending": pending,
        "errors": errors,
    }


async def _upsert_entity_with_extra_props(
    graph, entity: dict, extra_props: dict,
) -> None:
    """Re-upsert an entity with additional properties merged in.

    The Memory Store ``_update`` endpoint's ``set`` field does not persist
    into the nested ``properties`` object.  ``_upsert`` replaces the full
    properties dict, so we merge the enrichment data into the existing
    entity properties and re-upsert.
    """
    # Build merged properties (exclude internal fields)
    merged = {
        k: v for k, v in entity.items()
        if k not in ("_labels", "_score", "embedding")
    }
    merged.update(extra_props)

    # Extract embedding if present for kNN indexing
    embedding = entity.get("embedding") if isinstance(entity.get("embedding"), list) else None
    if embedding:
        merged["embedding"] = embedding

    await graph.upsert_node("Entity", id=entity["id"], properties=merged)


async def update_entities_by_name(
    client: MemoryClient,
    entity_name: str,
    all_entities: list[dict],
    enrichment_data: dict,
) -> int:
    """Update all entities sharing a name with the same enrichment data.

    Returns:
        Number of entities updated.
    """
    graph = _get_graph_backend(client)
    from datetime import datetime, timezone

    extra = {
        "enriched_description": enrichment_data.get("description"),
        "wikipedia_url": enrichment_data.get("wikipedia_url"),
        "wikidata_id": enrichment_data.get("wikidata_id"),
        "image_url": enrichment_data.get("image_url"),
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "enrichment_provider": "wikimedia",
    }

    count = 0
    name_lower = entity_name.strip().lower()
    for e in all_entities:
        if e.get("name", "").strip().lower() == name_lower:
            await _upsert_entity_with_extra_props(graph, e, extra)
            count += 1
    return count


async def mark_entities_not_found_by_name(
    client: MemoryClient,
    entity_name: str,
    all_entities: list[dict],
    reason: str,
) -> int:
    """Mark all entities with matching name as not found.

    Returns:
        Number of entities marked.
    """
    graph = _get_graph_backend(client)
    from datetime import datetime, timezone

    extra = {
        "enrichment_error": reason,
        "enrichment_attempted_at": datetime.now(timezone.utc).isoformat(),
    }

    count = 0
    name_lower = entity_name.strip().lower()
    for e in all_entities:
        if e.get("name", "").strip().lower() == name_lower:
            await _upsert_entity_with_extra_props(graph, e, extra)
            count += 1
    return count


async def enrich_entities(
    client: MemoryClient,
    all_entities: list[dict],
    entity_types: list[str] | None = None,
    limit: int | None = None,
    rate_limit: float = 0.5,
    dry_run: bool = False,
) -> EnrichmentStats:
    """Enrich entities with Wikipedia data.

    Uses Memory Store API for all data access. Deduplicates by entity name
    so each unique name is only looked up on Wikipedia once, then all
    matching nodes are updated.

    Args:
        client: Memory client
        all_entities: Pre-loaded list of all entities
        entity_types: Filter by entity types
        limit: Maximum unique entities to process
        rate_limit: Seconds between API calls (default 0.5 = 2 req/sec)
        dry_run: If True, don't actually update entities

    Returns:
        EnrichmentStats with counts
    """
    stats = EnrichmentStats()

    # Get deduplicated unenriched entities
    print(f"\n{color('Filtering unenriched entities...', Colors.CYAN)}")
    entities = filter_unenriched_entities(all_entities, entity_types, limit)
    stats.total = len(entities)

    if stats.total == 0:
        print(f"{color('All entities are already enriched!', Colors.GREEN)}")
        return stats

    print(f"Found {color(str(stats.total), Colors.BOLD)} unique entities to enrich")

    if dry_run:
        print(f"{color('DRY RUN - no changes will be made', Colors.YELLOW)}")
        for entity in entities[:10]:
            print(f"  Would enrich: {entity.get('name', '?')} ({entity.get('type', '?')})")
        if stats.total > 10:
            print(f"  ... and {stats.total - 10} more")
        return stats

    # Create Wikimedia provider
    provider = WikimediaProvider(
        rate_limit=rate_limit,
        language="en",
    )

    print(f"\n{color('Starting enrichment...', Colors.CYAN)}")
    print(f"Rate limit: {1/rate_limit:.1f} requests/second")
    print()

    start_time = time.time()

    for i, entity in enumerate(entities):
        entity_name = entity.get("name", "")
        entity_type = entity.get("type", "")
        entity_id = entity["id"]

        # Print progress
        print_progress(i, stats.total, stats, start_time, entity_name)

        try:
            # Call Wikimedia API
            result = await provider.enrich(
                entity_name,
                entity_type,
                context=entity.get("description"),
            )

            if result.status == EnrichmentStatus.SUCCESS and result.has_data():
                # Update all entities with this name
                enrichment_data = {
                    "description": result.description,
                    "wikipedia_url": result.wikipedia_url,
                    "wikidata_id": result.wikidata_id,
                    "image_url": result.image_url,
                }
                updated = await update_entities_by_name(
                    client, entity_name, all_entities, enrichment_data,
                )
                stats.enriched += 1
                if updated > 1:
                    logger.debug(f"Updated {updated} nodes for '{entity_name}'")

            elif result.status == EnrichmentStatus.NOT_FOUND:
                await mark_entities_not_found_by_name(
                    client, entity_name, all_entities, "Not found in Wikipedia",
                )
                stats.not_found += 1

            elif result.status == EnrichmentStatus.RATE_LIMITED:
                stats.rate_limited += 1
                # Wait longer and retry
                await asyncio.sleep(rate_limit * 5)
                continue

            else:
                await mark_entities_not_found_by_name(
                    client, entity_name, all_entities,
                    result.error_message or "Unknown error",
                )
                stats.errors += 1

        except Exception as e:
            logger.error(f"Error enriching {entity_name}: {e}")
            await mark_entities_not_found_by_name(
                client, entity_name, all_entities, str(e),
            )
            stats.errors += 1

        # Rate limiting
        await asyncio.sleep(rate_limit)

    # Final progress
    print_progress(stats.total, stats.total, stats, start_time, "Complete!")
    print()  # New line after progress bar

    return stats


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich entities with Wikipedia data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enrich all entities
  python enrich_entities.py

  # Enrich only PERSON and ORGANIZATION entities
  python enrich_entities.py --types PERSON ORGANIZATION

  # Enrich with slower rate limit (1 req/sec)
  python enrich_entities.py --rate-limit 1.0

  # Preview what would be enriched
  python enrich_entities.py --dry-run

  # Check enrichment status
  python enrich_entities.py --status
        """,
    )

    parser.add_argument(
        "--types", "-t",
        nargs="+",
        help="Entity types to enrich (e.g., PERSON ORGANIZATION LOCATION)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of entities to enrich",
    )
    parser.add_argument(
        "--rate-limit", "-r",
        type=float,
        default=0.5,
        help="Seconds between API calls (default: 0.5 = 2 req/sec)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be enriched without making changes",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current enrichment status and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print()
    print(color("═" * 60, Colors.PURPLE))
    print(color("  Wikipedia Entity Enrichment", Colors.BOLD))
    print(color("═" * 60, Colors.PURPLE))

    # Get settings from environment
    import os
    from pydantic import SecretStr

    memory_store_username = os.getenv("MEMORY_STORE_USERNAME")
    memory_store_password = os.getenv("MEMORY_STORE_PASSWORD")
    memory_store_verify_ssl = os.getenv("MEMORY_STORE_VERIFY_SSL", "true").lower() not in (
        "false",
        "0",
        "no",
    )

    settings = MemorySettings(
        backend="memory_store",
        memory_store=MemoryStoreConfig(
            endpoint=os.getenv("MEMORY_STORE_ENDPOINT", "https://localhost:9200"),
            username=memory_store_username,
            password=SecretStr(memory_store_password) if memory_store_password else None,
            verify_ssl=memory_store_verify_ssl,
            user_id=memory_store_username or "default",
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.BEDROCK,
            model="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            aws_region=os.getenv("AWS_REGION", "us-west-2"),
        ),
    )

    print(f"\n{color('Connecting to Memory Store...', Colors.CYAN)}")

    async with MemoryClient(settings) as client:
        print(f"{color('Connected', Colors.GREEN)}")

        # Load all entities once
        print(f"\n{color('Loading entities from Memory Store...', Colors.CYAN)}")
        all_entities = await get_all_entities(client)
        print(f"  Loaded {len(all_entities):,} entity nodes")

        # Show status
        status = compute_enrichment_status(all_entities)
        print(f"\n{color('Current Status:', Colors.BOLD)}")
        print(f"  Total entities:    {status['total']:,}")
        print(f"  {color('Enriched:', Colors.GREEN)}        {status['enriched']:,}")
        print(f"  {color('Pending:', Colors.YELLOW)}         {status['pending']:,}")
        print(f"  {color('Errors:', Colors.RED)}          {status['errors']:,}")

        if status['total'] > 0:
            pct = status['enriched'] / status['total'] * 100
            print(f"  Coverage:          {pct:.1f}%")

        if args.status:
            return

        if status['pending'] == 0:
            print(f"\n{color('All entities are already enriched!', Colors.GREEN)}")
            return

        # Count unique unenriched names for time estimate
        unique_unenriched = filter_unenriched_entities(all_entities, args.types, args.limit)
        pending = len(unique_unenriched)
        estimated_time = pending * args.rate_limit
        print(f"\n  Unique names to enrich: {pending:,}")
        print(f"  {color('Estimated time:', Colors.DIM)} {format_time(estimated_time)}")

        # Run enrichment
        stats = await enrich_entities(
            client,
            all_entities,
            entity_types=args.types,
            limit=args.limit,
            rate_limit=args.rate_limit,
            dry_run=args.dry_run,
        )

        # Print summary
        print()
        print(color("═" * 60, Colors.PURPLE))
        print(color("  Enrichment Complete", Colors.BOLD))
        print(color("═" * 60, Colors.PURPLE))
        print(f"  Processed:     {stats.total:,}")
        print(f"  {color('Enriched:', Colors.GREEN)}     {stats.enriched:,}")
        print(f"  {color('Not found:', Colors.YELLOW)}    {stats.not_found:,}")
        print(f"  {color('Errors:', Colors.RED)}       {stats.errors:,}")
        if stats.rate_limited > 0:
            print(f"  {color('Rate limited:', Colors.YELLOW)} {stats.rate_limited:,}")
        print(f"  Success rate:  {stats.success_rate:.1f}%")
        print()


if __name__ == "__main__":
    asyncio.run(main())
