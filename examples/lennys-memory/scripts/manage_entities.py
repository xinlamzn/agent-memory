#!/usr/bin/env python3
"""Export and import the full entity graph for Lenny's Podcast data.

Exports entities, MENTIONS edges, RELATED_TO relationship edges, and all
enrichment data (Wikipedia descriptions, geocoding, etc.) in a portable
JSON format keyed by session_id and turn_index.

This allows the complete entity graph — including backfilled relationships
and enrichment — to be restored into a fresh database without re-running
the NER extraction, relationship backfill, or enrichment pipelines.

Format version 2 adds:
  - Full entity properties (enrichment, geocoding, etc.)
  - RELATED_TO edges keyed by entity name (stable across re-imports)

Usage:
    # Export full graph to JSON
    python manage_entities.py export entities.json

    # Import full graph from JSON
    python manage_entities.py import entities.json

    # Dry-run import (show what would be created)
    python manage_entities.py import entities.json --dry-run
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from backend directory
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")


def _build_url(endpoint: str, path: str) -> str:
    return f"{endpoint.rstrip('/')}/{path.lstrip('/')}"


def _make_request(endpoint: str, path: str, body: dict | None = None,
                  method: str = "GET", username: str | None = None,
                  password: str | None = None, verify_ssl: bool = True) -> dict:
    """Make an HTTP request to OpenSearch."""
    url = _build_url(endpoint, path)
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


def _scroll_all(endpoint: str, index: str, query: dict, fields: list[str] | None = None,
                username: str | None = None, password: str | None = None,
                verify_ssl: bool = True, batch_size: int = 5000) -> list[dict]:
    """Scroll through all documents matching a query."""
    body: dict = {"query": query, "size": batch_size}
    if fields:
        body["_source"] = fields

    # Initial search with scroll
    result = _make_request(
        endpoint, f"{index}/_search?scroll=2m",
        body=body, method="POST",
        username=username, password=password, verify_ssl=verify_ssl,
    )

    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    all_docs = list(hits)

    # Continue scrolling
    while hits:
        result = _make_request(
            endpoint, "_search/scroll",
            body={"scroll": "2m", "scroll_id": scroll_id},
            method="POST",
            username=username, password=password, verify_ssl=verify_ssl,
        )
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        all_docs.extend(hits)

    # Clear scroll
    if scroll_id:
        try:
            _make_request(
                endpoint, "_search/scroll",
                body={"scroll_id": scroll_id},
                method="DELETE",
                username=username, password=password, verify_ssl=verify_ssl,
            )
        except Exception:
            pass

    return all_docs


def export_entities(args):
    """Export entities from the graph to a JSON file."""
    endpoint = args.endpoint
    database = args.database
    username = os.getenv("MEMORY_STORE_USERNAME")
    password = os.getenv("MEMORY_STORE_PASSWORD")
    verify_ssl = os.getenv("MEMORY_STORE_VERIFY_SSL", "true").lower() not in ("false", "0", "no")
    nodes_index = f"{database}-lpg-nodes"
    edges_index = f"{database}-lpg-edges"

    print("Exporting entities from Memory Store...")
    print(f"  Endpoint: {endpoint}")
    print(f"  Database: {database}")
    start_time = time.time()

    # Step 1: Fetch all Conversation nodes → {node_id: session_id}
    print("  Fetching conversations...", end=" ", flush=True)
    conv_docs = _scroll_all(
        endpoint, nodes_index,
        {"term": {"labels": "Conversation"}},
        fields=["id", "properties.session_id"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    conv_map = {}  # node_id → session_id
    for doc in conv_docs:
        src = doc["_source"]
        conv_map[src["id"]] = src["properties"]["session_id"]
    print(f"{len(conv_map)} conversations")

    # Step 2: Fetch all HAS_MESSAGE edges → {message_node_id: conv_node_id}
    print("  Fetching HAS_MESSAGE edges...", end=" ", flush=True)
    has_msg_docs = _scroll_all(
        endpoint, edges_index,
        {"term": {"type": "HAS_MESSAGE"}},
        fields=["source", "target"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    msg_to_conv = {}  # message_node_id → conv_node_id
    for doc in has_msg_docs:
        src = doc["_source"]
        # HAS_MESSAGE: Conversation → Message
        msg_to_conv[src["target"]] = src["source"]
    print(f"{len(msg_to_conv)} edges")

    # Step 3: Fetch all Message nodes → {node_id: turn_index}
    print("  Fetching messages...", end=" ", flush=True)
    msg_docs = _scroll_all(
        endpoint, nodes_index,
        {"term": {"labels": "Message"}},
        fields=["id", "properties.metadata"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    msg_info = {}  # node_id → turn_index
    for doc in msg_docs:
        src = doc["_source"]
        metadata = src["properties"].get("metadata")
        if metadata:
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    continue
            turn_index = metadata.get("turn_index")
            if turn_index is not None:
                msg_info[src["id"]] = int(turn_index)
    print(f"{len(msg_info)} messages with turn_index")

    # Step 4: Fetch all Entity nodes → {node_id: entity_data}
    # Include full properties so enrichment/geocoding data is preserved.
    print("  Fetching entities...", end=" ", flush=True)
    entity_docs = _scroll_all(
        endpoint, nodes_index,
        {"term": {"labels": "Entity"}},
        fields=["id", "labels", "properties"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    entity_map = {}  # node_id → entity_data
    # Properties to export (skip large/internal fields)
    _SKIP_PROPS = {"id", "embedding", "task_embedding"}
    for doc in entity_docs:
        src = doc["_source"]
        props = src["properties"]
        entity_data = {k: v for k, v in props.items() if k not in _SKIP_PROPS and v is not None}
        entity_data["labels"] = src.get("labels", [])
        entity_map[src["id"]] = entity_data
    print(f"{len(entity_map)} entities")

    # Step 5: Fetch all MENTIONS edges → message_node_id → [(entity_node_id, props)]
    print("  Fetching MENTIONS edges...", end=" ", flush=True)
    mentions_docs = _scroll_all(
        endpoint, edges_index,
        {"term": {"type": "MENTIONS"}},
        fields=["source", "target", "properties"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    print(f"{len(mentions_docs)} edges")

    # Step 6: Fetch all RELATED_TO edges → relationships between entities
    print("  Fetching RELATED_TO edges...", end=" ", flush=True)
    rel_docs = _scroll_all(
        endpoint, edges_index,
        {"term": {"type": "RELATED_TO"}},
        fields=["source", "target", "properties"],
        username=username, password=password, verify_ssl=verify_ssl,
    )
    # Resolve to entity names for portability across re-imports
    relationships = []
    rel_skipped = 0
    for doc in rel_docs:
        src = doc["_source"]
        source_entity = entity_map.get(src["source"])
        target_entity = entity_map.get(src["target"])
        if not source_entity or not target_entity:
            rel_skipped += 1
            continue
        rel_props = src.get("properties", {})
        relationships.append({
            "source_name": source_entity["name"],
            "source_type": source_entity.get("type"),
            "target_name": target_entity["name"],
            "target_type": target_entity.get("type"),
            "relation_type": rel_props.get("relation_type"),
            "confidence": rel_props.get("confidence"),
            "created_at": rel_props.get("created_at"),
        })
    print(f"{len(rel_docs)} edges ({len(relationships)} resolved, {rel_skipped} skipped)")

    # Step 7: Assemble into export format
    print("  Assembling export data...", end=" ", flush=True)
    sessions: dict[str, dict[int, list[dict]]] = {}
    skipped = 0

    for doc in mentions_docs:
        src = doc["_source"]
        msg_node_id = src["source"]   # MENTIONS: Message → Entity
        ent_node_id = src["target"]
        mention_props = src.get("properties", {})

        # Resolve message → (session_id, turn_index)
        conv_node_id = msg_to_conv.get(msg_node_id)
        if not conv_node_id:
            skipped += 1
            continue
        session_id = conv_map.get(conv_node_id)
        if not session_id:
            skipped += 1
            continue
        turn_index = msg_info.get(msg_node_id)
        if turn_index is None:
            skipped += 1
            continue

        # Resolve entity
        entity_data = entity_map.get(ent_node_id)
        if not entity_data:
            skipped += 1
            continue

        # Build entry — include all entity properties for full restore
        entry = dict(entity_data)
        entry["mention"] = {
            "confidence": mention_props.get("confidence"),
            "start_pos": mention_props.get("start_pos"),
            "end_pos": mention_props.get("end_pos"),
        }

        if session_id not in sessions:
            sessions[session_id] = {}
        if turn_index not in sessions[session_id]:
            sessions[session_id][turn_index] = []
        sessions[session_id][turn_index].append(entry)

    # Convert turn_index keys to sorted lists for cleaner JSON
    export_data = {
        "version": 2,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats": {
            "sessions": len(sessions),
            "messages_with_entities": sum(len(turns) for turns in sessions.values()),
            "total_mentions": len(mentions_docs),
            "unique_entities": len(entity_map),
            "skipped_mentions": skipped,
            "relationships": len(relationships),
            "skipped_relationships": rel_skipped,
        },
        "sessions": {},
    }

    for session_id in sorted(sessions):
        turns = sessions[session_id]
        export_data["sessions"][session_id] = []
        for turn_index in sorted(turns):
            export_data["sessions"][session_id].append({
                "turn_index": turn_index,
                "entities": turns[turn_index],
            })

    export_data["relationships"] = relationships

    # Write JSON
    output_path = Path(args.output_file)
    output_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - start_time
    print("Done!")
    print()
    print(f"  Sessions: {export_data['stats']['sessions']}")
    print(f"  Messages with entities: {export_data['stats']['messages_with_entities']}")
    print(f"  Total mentions: {export_data['stats']['total_mentions']}")
    print(f"  Unique entities: {export_data['stats']['unique_entities']}")
    print(f"  Relationships: {export_data['stats']['relationships']}")
    if skipped:
        print(f"  Skipped mentions: {skipped}")
    if rel_skipped:
        print(f"  Skipped relationships: {rel_skipped}")
    print(f"  Output: {output_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


def import_entities(args):
    """Import entities from a JSON file into the graph."""
    # Defer heavy imports to avoid slow startup for export-only usage
    import asyncio
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    from pydantic import SecretStr

    from neo4j_agent_memory import (
        MemoryClient,
        MemorySettings,
        MemoryStoreConfig,
    )

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"Loading entities from {input_path}...")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    version = data.get("version", 1)
    if version not in (1, 2):
        print(f"Warning: Unknown format version {version}, attempting import anyway")

    stats = data.get("stats", {})
    sessions = data.get("sessions", {})
    relationships = data.get("relationships", [])
    print(f"  Version: {version}")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Total mentions: {stats.get('total_mentions', 'unknown')}")
    if relationships:
        print(f"  Relationships: {len(relationships)}")

    if args.dry_run:
        print("\nDry run — showing summary per session:")
        for session_id in sorted(sessions):
            turns = sessions[session_id]
            total_entities = sum(len(t["entities"]) for t in turns)
            print(f"  {session_id}: {len(turns)} messages, {total_entities} entities")
        total = sum(sum(len(t["entities"]) for t in turns) for turns in sessions.values())
        print(f"\nTotal: {total} entity mentions across {len(sessions)} sessions")
        if relationships:
            # Summarise relationship types
            type_counts: dict[str, int] = {}
            for rel in relationships:
                rt = rel.get("relation_type", "UNKNOWN")
                type_counts[rt] = type_counts.get(rt, 0) + 1
            print(f"\nRelationships: {len(relationships)} total")
            for rt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"  {rt}: {count}")
        return

    # Build MemoryClient settings
    endpoint = args.endpoint
    database = args.database
    username = os.getenv("MEMORY_STORE_USERNAME")
    password = os.getenv("MEMORY_STORE_PASSWORD")
    verify_ssl = os.getenv("MEMORY_STORE_VERIFY_SSL", "true").lower() not in ("false", "0", "no")

    settings = MemorySettings(
        backend="memory_store",
        memory_store=MemoryStoreConfig(
            endpoint=endpoint,
            database=database,
            username=username,
            password=SecretStr(password) if password else None,
            verify_ssl=verify_ssl,
            user_id=username or "default",
        ),
    )

    async def _import():
        memory_client = MemoryClient(settings)
        async with memory_client as memory:
            print(f"Connected to Memory Store at {endpoint}")
            client = memory.short_term._client

            total_sessions = len(sessions)
            total_created = 0
            total_linked = 0
            total_skipped = 0
            total_rels_created = 0
            total_rels_skipped = 0
            name_to_ids: dict[tuple[str, str | None], list[str]] = {}
            start_time = time.time()

            for sess_idx, session_id in enumerate(sorted(sessions), 1):
                turns = sessions[session_id]

                # Find the Conversation node for this session
                conv_node = await client.get_node(
                    "Conversation", filters={"session_id": session_id}
                )
                if not conv_node:
                    print(f"  [{sess_idx}/{total_sessions}] {session_id}: conversation not found, skipping")
                    total_skipped += sum(len(t["entities"]) for t in turns)
                    continue

                conv_id = conv_node["id"]

                # Get all messages for this conversation
                messages = await client.traverse(
                    "Conversation", conv_id,
                    relationship_types=["HAS_MESSAGE"],
                    target_labels=["Message"],
                    direction="outgoing",
                )
                if not messages:
                    print(f"  [{sess_idx}/{total_sessions}] {session_id}: no messages found, skipping")
                    total_skipped += sum(len(t["entities"]) for t in turns)
                    continue

                # Build turn_index → message_id mapping
                turn_to_msg = {}
                for msg in messages:
                    metadata = msg.get("metadata")
                    if metadata:
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                continue
                        ti = metadata.get("turn_index")
                        if ti is not None:
                            turn_to_msg[int(ti)] = msg["id"]

                session_created = 0
                session_skipped = 0

                for turn_data in turns:
                    turn_index = turn_data["turn_index"]
                    entities = turn_data["entities"]
                    message_id = turn_to_msg.get(turn_index)

                    if not message_id:
                        session_skipped += len(entities)
                        continue

                    for entity in entities:
                        from uuid import uuid4
                        entity_id = str(uuid4())

                        # Build additional labels from type/subtype
                        additional_labels = []
                        if entity.get("type"):
                            additional_labels.append(entity["type"])
                        if entity.get("subtype"):
                            additional_labels.append(entity["subtype"])

                        # Build properties — restore all exported fields
                        # v2 exports all entity properties directly; v1 has just name/type/subtype
                        _IMPORT_SKIP = {"labels", "mention"}
                        props = {k: v for k, v in entity.items() if k not in _IMPORT_SKIP and v is not None}
                        # Ensure minimum required fields
                        props.setdefault("name", entity.get("name", ""))
                        props.setdefault("canonical_name", entity.get("name", ""))

                        # Create Entity node
                        await client.upsert_node(
                            "Entity",
                            id=entity_id,
                            properties=props,
                            additional_labels=additional_labels if additional_labels else None,
                        )
                        total_created += 1

                        # Track name → id for relationship recreation
                        ent_name = entity.get("name", "")
                        ent_type = entity.get("type")
                        name_key = (ent_name, ent_type)
                        if name_key not in name_to_ids:
                            name_to_ids[name_key] = []
                        name_to_ids[name_key].append(entity_id)

                        mention = entity.get("mention", {})
                        link_props = {
                            "confidence": mention.get("confidence", entity.get("confidence", 0.85)),
                            "start_pos": mention.get("start_pos"),
                            "end_pos": mention.get("end_pos"),
                        }

                        # Create EXTRACTED_FROM edge (Entity → Message)
                        await client.link_nodes(
                            "Entity", entity_id,
                            "Message", message_id,
                            "EXTRACTED_FROM",
                            properties=link_props,
                        )

                        # Create MENTIONS edge (Message → Entity)
                        await client.link_nodes(
                            "Message", message_id,
                            "Entity", entity_id,
                            "MENTIONS",
                            properties=link_props,
                        )
                        total_linked += 1
                        session_created += 1

                elapsed = time.time() - start_time
                rate = total_created / elapsed if elapsed > 0 else 0
                print(
                    f"\r  [{sess_idx}/{total_sessions}] "
                    f"{session_id}: {session_created} entities"
                    f"{f', {session_skipped} skipped' if session_skipped else ''}"
                    f"  (total: {total_created}, {rate:.0f}/s)\033[K",
                    end="", flush=True,
                )
                total_skipped += session_skipped

            # Recreate RELATED_TO edges from the relationships section
            if relationships:
                print(f"\n  Importing {len(relationships)} relationships...")
                for rel_idx, rel in enumerate(relationships, 1):
                    src_key = (rel["source_name"], rel.get("source_type"))
                    tgt_key = (rel["target_name"], rel.get("target_type"))
                    src_ids = name_to_ids.get(src_key, [])
                    tgt_ids = name_to_ids.get(tgt_key, [])

                    if not src_ids or not tgt_ids:
                        total_rels_skipped += 1
                        continue

                    # Link the first matching entity pair
                    rel_props = {}
                    if rel.get("relation_type"):
                        rel_props["relation_type"] = rel["relation_type"]
                    if rel.get("confidence") is not None:
                        rel_props["confidence"] = rel["confidence"]
                    if rel.get("created_at"):
                        rel_props["created_at"] = rel["created_at"]

                    await client.link_nodes(
                        "Entity", src_ids[0],
                        "Entity", tgt_ids[0],
                        "RELATED_TO",
                        properties=rel_props,
                    )
                    total_rels_created += 1

                    if rel_idx % 50 == 0 or rel_idx == len(relationships):
                        print(
                            f"\r    Relationships: {rel_idx}/{len(relationships)}"
                            f" ({total_rels_created} created, {total_rels_skipped} skipped)\033[K",
                            end="", flush=True,
                        )
                print()

            elapsed = time.time() - start_time
            print()
            print(f"  Entities created: {total_created}")
            print(f"  Mention edges created: {total_linked * 2} (MENTIONS + EXTRACTED_FROM)")
            if total_rels_created:
                print(f"  Relationship edges created: {total_rels_created}")
            if total_skipped:
                print(f"  Skipped entities: {total_skipped}")
            if total_rels_skipped:
                print(f"  Skipped relationships: {total_rels_skipped}")
            print(f"  Elapsed: {elapsed:.1f}s")

    asyncio.run(_import())


def main():
    parser = argparse.ArgumentParser(
        description="Export and import extracted entities for Lenny's Podcast data"
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MEMORY_STORE_ENDPOINT", "https://localhost:9200"),
        help="Memory Store endpoint",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("MEMORY_STORE_DATABASE", "memory"),
        help="Memory Store database name (index prefix)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export entities to JSON")
    export_parser.add_argument("output_file", type=str, help="Output JSON file path")

    # Import subcommand
    import_parser = subparsers.add_parser("import", help="Import entities from JSON")
    import_parser.add_argument("input_file", type=str, help="Input JSON file path")
    import_parser.add_argument("--dry-run", action="store_true",
                               help="Show what would be imported without writing")

    args = parser.parse_args()

    if args.command == "export":
        export_entities(args)
    elif args.command == "import":
        import_entities(args)


if __name__ == "__main__":
    main()
