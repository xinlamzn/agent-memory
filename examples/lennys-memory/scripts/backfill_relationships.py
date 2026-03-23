#!/usr/bin/env python3
"""Backfill RELATED_TO relationships for existing entities in the database.

This script runs GLiREL relationship extraction on messages that already have
extracted entities, and creates RELATED_TO relationships between Entity nodes.

This is useful for databases that were populated before relationship extraction
was implemented, or when relationship extraction was disabled during initial load.

Features:
- Processes messages that have entities but no relationships extracted yet
- Uses GLiREL for relationship extraction (no LLM required)
- Batch processing with progress tracking
- Resume capability (tracks which messages have been processed)
- Rate limiting to avoid overwhelming the database
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

# Suppress warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    MemoryClient,
    MemorySettings,
    MemoryStoreConfig,
)
from neo4j_agent_memory.extraction.gliner_extractor import (
    GLiRELExtractor,
    is_glirel_available,
)
from neo4j_agent_memory.extraction.base import ExtractedEntity

# Load .env file from backend directory
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLORS = supports_color()


def color(text: str, color_code: str) -> str:
    if USE_COLORS:
        return f"{color_code}{text}{Colors.RESET}"
    return text


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ---------------------------------------------------------------------------
# OpenSearch helpers (for bulk data loading without Cypher)
# ---------------------------------------------------------------------------

def _os_request(endpoint: str, path: str, body: dict | None = None,
                method: str = "POST", username: str | None = None,
                password: str | None = None, verify_ssl: bool = True) -> dict:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if username and password:
        import base64
        creds = base64.b64encode(f"{username}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {creds}")
    ctx = None
    if not verify_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    resp = urllib.request.urlopen(req, context=ctx)
    return json.loads(resp.read().decode())


def _scroll_all(endpoint: str, index: str, query: dict,
                fields: list[str] | None = None,
                username: str | None = None, password: str | None = None,
                verify_ssl: bool = True, batch_size: int = 5000) -> list[dict]:
    body: dict = {"query": query, "size": batch_size}
    if fields:
        body["_source"] = fields
    result = _os_request(endpoint, f"{index}/_search?scroll=2m",
                         body=body, username=username, password=password,
                         verify_ssl=verify_ssl)
    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    all_docs = list(hits)
    while hits:
        result = _os_request(endpoint, "_search/scroll",
                             body={"scroll": "2m", "scroll_id": scroll_id},
                             username=username, password=password,
                             verify_ssl=verify_ssl)
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        all_docs.extend(hits)
    if scroll_id:
        try:
            _os_request(endpoint, "_search/scroll",
                        body={"scroll_id": scroll_id}, method="DELETE",
                        username=username, password=password,
                        verify_ssl=verify_ssl)
        except Exception:
            pass
    return all_docs


def load_messages_with_entities(endpoint: str, username: str | None = None,
                                password: str | None = None,
                                verify_ssl: bool = True) -> list[dict]:
    """Load all messages that have 2+ entities, along with entity details.

    Returns list of dicts: {message_id, content, entities: [{id, name, type, subtype}]}
    """
    # 1. Get all MENTIONS edges → group by message
    print("  Loading MENTIONS edges...", end=" ", flush=True)
    mentions_docs = _scroll_all(
        endpoint, "memory-lpg-edges",
        {"term": {"type": "MENTIONS"}},
        fields=["source", "target"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    msg_to_entity_ids: dict[str, set[str]] = defaultdict(set)
    for doc in mentions_docs:
        src = doc["_source"]
        msg_to_entity_ids[src["source"]].add(src["target"])
    print(f"{len(mentions_docs)} edges, {len(msg_to_entity_ids)} messages")

    # Filter to messages with 2+ entities
    multi_entity_msgs = {
        mid: eids for mid, eids in msg_to_entity_ids.items()
        if len(eids) >= 2
    }
    print(f"  Messages with 2+ entities: {len(multi_entity_msgs)}")

    if not multi_entity_msgs:
        return []

    # 2. Collect all needed entity IDs and message IDs
    all_entity_ids = set()
    for eids in multi_entity_msgs.values():
        all_entity_ids.update(eids)

    # 3. Load entity details
    print("  Loading entity details...", end=" ", flush=True)
    entity_docs = _scroll_all(
        endpoint, "memory-lpg-nodes",
        {"term": {"labels": "Entity"}},
        fields=["id", "properties.name", "properties.type", "properties.subtype", "properties.id"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    entity_map = {}  # node_id → {id, name, type, subtype}
    for doc in entity_docs:
        src = doc["_source"]
        props = src["properties"]
        entity_map[src["id"]] = {
            "id": props.get("id", src["id"]),
            "name": props.get("name"),
            "type": props.get("type"),
            "subtype": props.get("subtype"),
        }
    print(f"{len(entity_map)} entities")

    # 4. Load message content for relevant messages
    print("  Loading message content...", end=" ", flush=True)
    msg_docs = _scroll_all(
        endpoint, "memory-lpg-nodes",
        {"term": {"labels": "Message"}},
        fields=["id", "properties.content", "properties.id"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    msg_content = {}  # node_id → {id, content}
    for doc in msg_docs:
        src = doc["_source"]
        node_id = src["id"]
        if node_id in multi_entity_msgs:
            msg_content[node_id] = {
                "id": src["properties"].get("id", node_id),
                "content": src["properties"].get("content", ""),
            }
    print(f"{len(msg_content)} messages loaded")

    # 5. Assemble results
    results = []
    for msg_node_id, entity_node_ids in multi_entity_msgs.items():
        msg = msg_content.get(msg_node_id)
        if not msg or not msg["content"]:
            continue
        entities = []
        for eid in entity_node_ids:
            e = entity_map.get(eid)
            if e and e["name"]:
                entities.append(e)
        if len(entities) >= 2:
            results.append({
                "message_id": msg["id"],
                "content": msg["content"],
                "entities": entities,
            })

    return results


def get_existing_related_to(endpoint: str, username: str | None = None,
                            password: str | None = None,
                            verify_ssl: bool = True) -> set[tuple[str, str]]:
    """Get set of (source_id, target_id) for existing RELATED_TO edges."""
    docs = _scroll_all(
        endpoint, "memory-lpg-edges",
        {"term": {"type": "RELATED_TO"}},
        fields=["source", "target"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    pairs = set()
    for doc in docs:
        src = doc["_source"]
        pairs.add((src["source"], src["target"]))
        pairs.add((src["target"], src["source"]))  # bidirectional check
    return pairs


def count_related_to(endpoint: str, username: str | None = None,
                     password: str | None = None,
                     verify_ssl: bool = True) -> int:
    """Count RELATED_TO edges."""
    result = _os_request(
        endpoint, "memory-lpg-edges/_count",
        body={"query": {"term": {"type": "RELATED_TO"}}},
        username=username, password=password, verify_ssl=verify_ssl,
    )
    return result.get("count", 0)


# ---------------------------------------------------------------------------
# LLM entity filtering via Bedrock
# ---------------------------------------------------------------------------

FILTER_PROMPT = """You are filtering named entity recognition results. Given a text excerpt and a list of extracted entities, return ONLY the real named entities — specific proper nouns that refer to actual people, companies, products, technologies, or places.

REMOVE:
- Common nouns (product, company, startup, engineer, founder, customer, team, model, agent)
- Generic terms (AI, LLM, three, five, world, space, market, business)
- Temporal phrases (the end of the day, these days, daily, hours)
- Roles/titles used generically (CEO, PM, engineer, founder)
- Vague references (stuff, things, way)

KEEP:
- Real person names (Lenny Rachitsky, Marc Andreessen)
- Real company names (Google, Stripe, Anthropic)
- Real product names (ChatGPT, Cursor, Slack, Duolingo)
- Real technology names (React, Next.js, PostgreSQL)
- Real place names (San Francisco, Silicon Valley)
- Specific named concepts (Series A, Y Combinator)

Text:
{text}

Extracted entities:
{entities}

Return a JSON array of ONLY the entity names that are real named entities. Example: ["Google", "Marc Andreessen", "ChatGPT"]
Return just the JSON array, nothing else."""


class BedrockEntityFilter:
    """Filter entities using Bedrock Claude to identify real named entities."""

    def __init__(self, model_id: str, region: str):
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id

    def filter_entities(self, text: str, entity_names: list[str]) -> set[str]:
        """Return set of entity names that are real named entities."""
        if not entity_names:
            return set()

        # Deduplicate for the prompt
        unique_names = sorted(set(entity_names))
        entities_str = "\n".join(f"- {name}" for name in unique_names)

        # Truncate text if too long
        max_text = 2000
        if len(text) > max_text:
            text = text[:max_text] + "..."

        prompt = FILTER_PROMPT.format(text=text, entities=entities_str)

        try:
            response = self._client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }),
            )
            result = json.loads(response["body"].read())
            content = result["content"][0]["text"].strip()

            # Parse JSON array from response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            kept_names = set(json.loads(content))
            return kept_names

        except Exception as e:
            logger.warning(f"LLM filter failed: {e}")
            # Fallback: keep all entities (skip filtering for this message)
            return set(entity_names)


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

async def process_message(
    memory: MemoryClient,
    extractor: GLiRELExtractor,
    message: dict,
    entity_filter: BedrockEntityFilter | None = None,
) -> dict:
    """Process a single message to extract and store relationships."""
    content = message["content"]
    entities_data = message["entities"]

    # LLM-based entity filtering
    if entity_filter:
        all_names = [e["name"] for e in entities_data]
        kept_names = entity_filter.filter_entities(content, all_names)
        entities_data = [e for e in entities_data if e["name"] in kept_names]

    # Convert to ExtractedEntity objects for GLiREL
    entities = []
    entity_id_map = {}

    for e in entities_data:
        entity = ExtractedEntity(
            name=e["name"],
            type=e["type"],
            subtype=e.get("subtype"),
            confidence=1.0,
        )
        entities.append(entity)
        entity_id_map[e["name"].lower()] = e["id"]

    if len(entities) < 2:
        return {"relations_extracted": 0, "relations_stored": 0,
                "entities_before": len(message["entities"]),
                "entities_after": len(entities)}

    # Extract relations using GLiREL
    try:
        relations = await extractor.extract_relations(content, entities)
    except Exception as e:
        logger.warning(f"Failed to extract relations: {e}")
        return {"relations_extracted": 0, "relations_stored": 0, "error": str(e)}

    if not relations:
        return {"relations_extracted": 0, "relations_stored": 0,
                "entities_before": len(message["entities"]),
                "entities_after": len(entities)}

    # Store relations using the graph client API
    client = memory.short_term._client
    stored = 0
    from datetime import datetime

    for rel in relations:
        source_name = rel.source.lower().strip()
        target_name = rel.target.lower().strip()
        source_id = entity_id_map.get(source_name)
        target_id = entity_id_map.get(target_name)

        if source_id and target_id:
            try:
                await client.link_nodes(
                    "Entity", source_id,
                    "Entity", target_id,
                    "RELATED_TO",
                    properties={
                        "relation_type": rel.relation_type,
                        "confidence": rel.confidence,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
                stored += 1
            except Exception as e:
                logger.debug(f"Failed to create relation: {e}")

    return {
        "relations_extracted": len(relations),
        "relations_stored": stored,
        "entities_before": len(message["entities"]),
        "entities_after": len(entities),
    }


async def backfill_relationships(
    memory: MemoryClient,
    extractor: GLiRELExtractor,
    messages: list[dict],
    entity_filter: BedrockEntityFilter | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    """Backfill RELATED_TO relationships for existing entities."""
    total_messages = len(messages)
    if limit:
        total_messages = min(total_messages, limit)
        messages = messages[:total_messages]

    print(f"\n{color('Relationship Backfill', Colors.BOLD + Colors.CYAN)}")
    print("=" * 50)
    print(f"  Messages to process: {total_messages:,}")
    if entity_filter:
        print(f"  Entity filtering: {color('LLM (Bedrock)', Colors.GREEN)}")
    if dry_run:
        print(f"  {color('DRY RUN MODE', Colors.YELLOW)}")
    print()

    if total_messages == 0:
        print(color("No messages need relationship extraction!", Colors.GREEN))
        return {"messages_processed": 0, "relations_extracted": 0, "relations_stored": 0}

    if dry_run:
        print("Sample of messages that would be processed:")
        for msg in messages[:5]:
            entity_names = [e["name"] for e in msg["entities"]]
            if entity_filter:
                kept = entity_filter.filter_entities(msg["content"], entity_names)
                filtered = [n for n in entity_names if n in kept]
                removed = [n for n in entity_names if n not in kept]
                print(f"  • {len(entity_names)} → {len(filtered)} entities")
                print(f"    Kept: {', '.join(filtered[:8])}")
                if removed:
                    print(f"    Removed: {', '.join(removed[:8])}")
            else:
                print(f"  • {len(msg['entities'])} entities: {', '.join(entity_names[:5])}")
                if len(entity_names) > 5:
                    print(f"    ... and {len(entity_names) - 5} more")
        print()
        return {"messages_processed": 0, "relations_extracted": 0, "relations_stored": 0, "dry_run": True}

    # Process messages
    start_time = time.time()
    processed = 0
    total_extracted = 0
    total_stored = 0
    total_entities_before = 0
    total_entities_after = 0
    errors = 0

    for msg in messages:
        result = await process_message(memory, extractor, msg, entity_filter=entity_filter)

        total_extracted += result.get("relations_extracted", 0)
        total_stored += result.get("relations_stored", 0)
        total_entities_before += result.get("entities_before", 0)
        total_entities_after += result.get("entities_after", 0)
        if result.get("error"):
            errors += 1

        processed += 1

        if processed % 10 == 0 or processed == total_messages:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_messages - processed) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  Progress: {processed}/{total_messages} "
                f"({processed/total_messages*100:.0f}%) "
                f"| Relations: {total_stored:,} stored "
                f"| {rate:.1f} msg/s "
                f"| ETA: {format_duration(eta)}"
                "\033[K"
            )
            sys.stdout.flush()

    print()  # Newline after progress

    elapsed = time.time() - start_time

    print()
    print(color("═" * 50, Colors.CYAN))
    print(color("  Backfill Complete!", Colors.BOLD + Colors.GREEN))
    print(color("═" * 50, Colors.CYAN))
    print()
    print(f"  {color('Messages processed:', Colors.DIM)} {processed:,}")
    if entity_filter and total_entities_before > 0:
        print(f"  {color('Entities before filter:', Colors.DIM)} {total_entities_before:,}")
        print(f"  {color('Entities after filter:', Colors.DIM)} {total_entities_after:,}")
        pct = (1 - total_entities_after / total_entities_before) * 100 if total_entities_before else 0
        print(f"  {color('Filtered out:', Colors.DIM)} {pct:.0f}%")
    print(f"  {color('Relations extracted:', Colors.DIM)} {total_extracted:,}")
    print(f"  {color('Relations stored:', Colors.DIM)} {total_stored:,}")
    print(f"  {color('Errors:', Colors.DIM)} {errors}")
    print(f"  {color('Elapsed time:', Colors.DIM)} {format_duration(elapsed)}")
    if elapsed > 0:
        print(f"  {color('Throughput:', Colors.DIM)} {processed / elapsed:.1f} msg/s")
    print()

    return {
        "messages_processed": processed,
        "relations_extracted": total_extracted,
        "relations_stored": total_stored,
        "errors": errors,
        "elapsed_seconds": elapsed,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill RELATED_TO relationships for existing entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  %(prog)s --status

  # Run backfill (skip already processed)
  %(prog)s

  # Reprocess all messages (including already processed)
  %(prog)s --reprocess

  # Process only 100 messages
  %(prog)s --limit 100

  # Preview without making changes
  %(prog)s --dry-run
""",
    )
    parser.add_argument(
        "--memory-store-endpoint",
        default=os.getenv("MEMORY_STORE_ENDPOINT", "https://localhost:9200"),
        help="Memory Store endpoint",
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", "us-west-2"),
        help="AWS region for Bedrock embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of messages per batch (default: 50)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess all messages, not just pending ones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit total messages to process",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for relations (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run GLiREL on (default: cpu)",
    )
    parser.add_argument(
        "--filter-model",
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Bedrock model ID for LLM entity filtering (default: Sonnet 4.5)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable LLM entity filtering (use raw NER entities)",
    )

    args = parser.parse_args()

    endpoint = args.memory_store_endpoint
    username = os.getenv("MEMORY_STORE_USERNAME")
    password = os.getenv("MEMORY_STORE_PASSWORD")
    verify_ssl = os.getenv("MEMORY_STORE_VERIFY_SSL", "true").lower() not in (
        "false", "0", "no",
    )

    # Handle --status without needing GLiREL
    if args.status:
        rel_count = count_related_to(endpoint, username, password, verify_ssl)
        messages = load_messages_with_entities(endpoint, username, password, verify_ssl)
        print()
        print(color("Relationship Extraction Status", Colors.BOLD + Colors.CYAN))
        print("=" * 50)
        print()
        print(f"  {color('Messages with 2+ entities:', Colors.DIM)} {len(messages):,}")
        print(f"  {color('Total RELATED_TO relationships:', Colors.DIM)} {rel_count:,}")
        print()
        return

    # Check GLiREL availability
    if not is_glirel_available():
        print(color("Error: GLiREL is not installed.", Colors.RED))
        print("Install it with: pip install glirel")
        sys.exit(1)

    # Load data from Memory Store
    print()
    print(color("Loading data from Memory Store...", Colors.DIM))
    messages = load_messages_with_entities(endpoint, username, password, verify_ssl)

    if not messages:
        print(color("No messages with 2+ entities found!", Colors.YELLOW))
        return

    # Connect via MemoryClient (for link_nodes API)
    from pydantic import SecretStr

    settings = MemorySettings(
        backend="memory_store",
        memory_store=MemoryStoreConfig(
            endpoint=endpoint,
            username=username,
            password=SecretStr(password) if password else None,
            verify_ssl=verify_ssl,
            user_id=username or "default",
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.BEDROCK,
            model="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            aws_region=args.aws_region,
        ),
    )

    print(color("Connecting to Memory Store...", Colors.DIM), end=" ", flush=True)

    try:
        memory_client = MemoryClient(settings)
        async with memory_client as memory:
            print(color("Connected!", Colors.GREEN))

            # Initialize LLM entity filter
            entity_filter = None
            if not args.no_filter:
                print(color("Setting up LLM entity filter...", Colors.DIM), end=" ", flush=True)
                entity_filter = BedrockEntityFilter(
                    model_id=args.filter_model,
                    region=args.aws_region,
                )
                print(color(f"Ready ({args.filter_model})", Colors.GREEN))

            # Initialize GLiREL extractor
            print(color("Loading GLiREL model...", Colors.DIM), end=" ", flush=True)
            extractor = GLiRELExtractor.for_poleo(
                threshold=args.threshold,
                device=args.device,
            )
            # Force model load
            _ = extractor.model
            print(color("Loaded!", Colors.GREEN))

            # Run backfill
            await backfill_relationships(
                memory,
                extractor,
                messages,
                entity_filter=entity_filter,
                dry_run=args.dry_run,
                limit=args.limit,
            )

    except KeyboardInterrupt:
        print()
        print(color("\nInterrupted by user.", Colors.YELLOW))
        sys.exit(130)
    except Exception as e:
        print(color(f"Error: {e}", Colors.RED))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
