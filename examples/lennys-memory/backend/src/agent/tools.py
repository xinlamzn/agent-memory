"""Podcast search tools for the chat agent.

This module provides tools for:
- Podcast content search (transcripts, speakers, episodes)
- Entity queries (people, organizations, topics)
- Location queries with geospatial features
- User preferences and reasoning memory
"""

import json
import logging

logger = logging.getLogger(__name__)
import math
import re
from typing import Any

from pydantic_ai import RunContext

from src.agent.dependencies import AgentDeps


def _guest_to_session_id(guest_name: str) -> str:
    """Convert a guest name to a session_id format.

    Handles Unicode characters by normalizing to ASCII equivalents
    (e.g., "Tobi Lütke" -> "tobi-lutke", not "tobi-l-tke").
    """
    import unicodedata

    # Normalize Unicode characters to their ASCII equivalents
    # NFD decomposes characters (ü -> u + combining umlaut)
    # Then we strip combining characters
    normalized = unicodedata.normalize("NFD", guest_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return f"lenny-podcast-{slug}"


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


async def search_podcast_content(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 10,
    context_messages: int = 0,
) -> list[dict[str, Any]]:
    """Search podcast transcripts using semantic search.

    Args:
        ctx: The agent run context.
        query: Search terms or topic to find.
        limit: Maximum number of results to return.
        context_messages: Number of surrounding messages to include for context (0-3).
            When > 0, includes messages before/after each result for fuller context.

    Returns:
        List of matching transcript segments with speaker, episode, and content.
        When context_messages > 0, includes surrounding conversation context.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    # Clamp context_messages to reasonable range
    context_messages = max(0, min(3, context_messages))

    try:
        messages = await ctx.deps.client.short_term.search_messages(
            query=query,
            limit=limit,
            threshold=0.5,
            metadata_filters={"source": "lenny_podcast"},
        )

        results = []
        for msg in messages:
            result = {
                "content": (msg.content[:500] + "..." if len(msg.content) > 500 else msg.content),
                "speaker": msg.metadata.get("speaker", "Unknown"),
                "episode_guest": msg.metadata.get("episode_guest", "Unknown"),
                "timestamp": msg.metadata.get("timestamp", ""),
                "relevance": round(msg.metadata.get("similarity", 0), 3),
            }

            # Fetch surrounding context if requested
            if context_messages > 0 and hasattr(msg, "id") and msg.id:
                try:
                    context = await _get_message_context(ctx, str(msg.id), context_messages)
                    if context:
                        result["context_before"] = context.get("before", [])
                        result["context_after"] = context.get("after", [])
                except Exception:
                    pass  # Silently skip context on error

            results.append(result)

        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


async def _get_message_context(
    ctx: RunContext[AgentDeps],
    message_id: str,
    context_size: int,
) -> dict[str, list[dict[str, str]]]:
    """Get surrounding messages for context.

    Uses NEXT_MESSAGE traversal when available.  Falls back gracefully
    for backends that do not support the relationship.
    """
    try:
        graph = ctx.deps.client.backend.graph
        neighbours = await graph.traverse(
            "Message",
            message_id,
            relationship_types=["NEXT_MESSAGE"],
            direction="both",
            target_labels=["Message"],
            depth=context_size,
            limit=context_size * 2,
        )
        before: list[dict[str, str]] = []
        after: list[dict[str, str]] = []
        for n in neighbours:
            content = (n.get("content") or "")[:200]
            meta = n.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            speaker = meta.get("speaker", "Unknown")
            # Without directional info from traverse, just return as context
            after.append({"content": content, "speaker": speaker})
        return {"before": before, "after": after[:context_size]}
    except Exception:
        return {"before": [], "after": []}


async def search_by_speaker(
    ctx: RunContext[AgentDeps],
    speaker: str,
    topic: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for what a specific speaker said.

    Args:
        ctx: The agent run context.
        speaker: Name of the speaker (e.g., "Brian Chesky", "Lenny").
        topic: Optional topic to filter by.
        limit: Maximum number of results to return.

    Returns:
        List of transcript segments from the specified speaker.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # If topic is provided, use vector search with speaker filter for best results
        if topic:
            messages = await ctx.deps.client.short_term.search_messages(
                query=topic,
                limit=limit * 2,  # Fetch extra to filter by speaker
                threshold=0.5,  # Lower threshold for better recall
                metadata_filters={"source": "lenny_podcast"},
            )

            # Filter by speaker name (case-insensitive, partial match)
            speaker_lower = speaker.lower()
            results = []
            for msg in messages:
                msg_speaker = msg.metadata.get("speaker", "").lower()
                if speaker_lower in msg_speaker or msg_speaker in speaker_lower:
                    results.append(
                        {
                            "content": (
                                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                            ),
                            "speaker": msg.metadata.get("speaker", "Unknown"),
                            "episode_guest": msg.metadata.get("episode_guest", "Unknown"),
                            "relevance": round(msg.metadata.get("similarity", 0), 3),
                        }
                    )
                    if len(results) >= limit:
                        break

            if results:
                return results

        # Fallback: broad semantic search including speaker name
        fallback_query = f"{speaker} {topic}" if topic else speaker
        messages = await ctx.deps.client.short_term.search_messages(
            query=fallback_query,
            limit=limit,
            threshold=0.4,
            metadata_filters={"source": "lenny_podcast"},
        )
        return [
            {
                "content": (
                    msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                ),
                "speaker": msg.metadata.get("speaker", "Unknown"),
                "episode_guest": msg.metadata.get("episode_guest", "Unknown"),
            }
            for msg in messages
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


async def search_by_episode(
    ctx: RunContext[AgentDeps],
    guest_name: str,
    topic: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search within a specific episode by guest name.

    Args:
        ctx: The agent run context.
        guest_name: Name of the podcast guest.
        topic: Optional topic to search for within the episode.
        limit: Maximum number of results to return.

    Returns:
        List of transcript segments from the specified episode.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # Convert guest name to session_id format (handles Unicode like "Lütke" -> "lutke")
        session_id = _guest_to_session_id(guest_name)

        if topic:
            # Search within episode for specific topic
            messages = await ctx.deps.client.short_term.search_messages(
                query=topic,
                session_id=session_id,
                limit=limit,
            )
        else:
            # Get conversation from episode
            conv = await ctx.deps.client.short_term.get_conversation(
                session_id=session_id,
                limit=limit,
            )
            messages = conv.messages if conv else []

        return [
            {
                "content": (msg.content[:500] + "..." if len(msg.content) > 500 else msg.content),
                "speaker": msg.metadata.get("speaker", "Unknown"),
                "timestamp": msg.metadata.get("timestamp", ""),
            }
            for msg in messages
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


async def get_episode_list(ctx: RunContext[AgentDeps]) -> list[dict[str, Any]]:
    """Get list of all podcast episodes.

    Args:
        ctx: The agent run context.

    Returns:
        List of episodes with guest names and message counts.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        sessions = await ctx.deps.client.short_term.list_sessions(
            prefix="lenny-podcast-",
            limit=500,
        )

        return [
            {
                "guest": s.session_id.replace("lenny-podcast-", "").replace("-", " ").title(),
                "session_id": s.session_id,
                "message_count": s.message_count,
            }
            for s in sessions
        ]
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]


async def get_speaker_list(ctx: RunContext[AgentDeps]) -> list[dict[str, Any]]:
    """Get list of all unique speakers across episodes.

    Derives speakers from session IDs: Lenny is the host in all episodes,
    and each session corresponds to a guest.

    Args:
        ctx: The agent run context.

    Returns:
        List of speakers with their appearance counts.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        sessions = await ctx.deps.client.short_term.list_sessions(
            prefix="lenny-podcast-",
            limit=500,
        )

        # Lenny appears in every episode; each guest appears once
        speakers: dict[str, int] = {"Lenny Rachitsky": len(sessions)}
        for s in sessions:
            guest = s.session_id.replace("lenny-podcast-", "").replace("-", " ").title()
            speakers[guest] = speakers.get(guest, 0) + 1

        return [
            {"speaker": name, "appearance_count": count}
            for name, count in sorted(speakers.items(), key=lambda x: -x[1])
        ]
    except Exception as e:
        return [{"error": f"Query failed: {str(e)}"}]


async def get_memory_stats(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """Get statistics about the loaded podcast data.

    Args:
        ctx: The agent run context.

    Returns:
        Dictionary with counts of episodes, messages, speakers, etc.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        stats = await ctx.deps.client.get_stats()

        # Handle both Neo4j format (flat keys) and Memory Store format (nested)
        conversations = stats.get("conversations", 0)
        messages = stats.get("messages", 0)
        entities = stats.get("entities", 0)

        # Memory Store returns stats grouped by namespace/label as bucket lists
        if not conversations and "by_labels" in stats:
            by_labels = stats["by_labels"]
            # Handle both dict format and bucket list format
            if isinstance(by_labels, dict) and "buckets" in by_labels:
                label_map = {b["key"]: b["count"] for b in by_labels["buckets"]}
            elif isinstance(by_labels, dict):
                label_map = by_labels
            else:
                label_map = {}
            conversations = label_map.get("Conversation", 0)
            messages = label_map.get("Message", 0)
            entities = label_map.get("Entity", 0)

        return {
            "conversations": conversations,
            "messages": messages,
            "entities": entities,
            "note": "These are podcast transcript segments loaded into memory",
        }
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}


# =============================================================================
# Entity Query Tools
# =============================================================================


async def search_entities(
    ctx: RunContext[AgentDeps],
    query: str,
    entity_type: str | None = None,
    limit: int = 10,
    collapse_duplicates: bool = True,
) -> list[dict[str, Any]]:
    """Search for entities (people, organizations, topics) mentioned in podcasts.

    Args:
        ctx: The agent run context.
        query: Search term (e.g., "product-market fit", "Y Combinator").
        entity_type: Filter by type - PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT.
        limit: Maximum number of results to return.
        collapse_duplicates: If True, collapse entities linked via SAME_AS relationships
            to show only the canonical entity. This prevents duplicate results.

    Returns:
        List of entities with descriptions and mention counts.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        entities = await ctx.deps.client.long_term.search_entities(
            query=query,
            entity_types=[entity_type.upper()] if entity_type else None,
            limit=limit * 2 if collapse_duplicates else limit,
            threshold=0.5,
        )

        if collapse_duplicates and entities:
            entities = await _collapse_duplicate_entities(ctx, entities, limit)

        results = []
        for e in entities[:limit]:
            metadata = getattr(e, "metadata", {}) or {}
            enriched_desc = getattr(e, "enriched_description", None) or metadata.get(
                "enriched_description", ""
            )
            wiki_url = getattr(e, "wikipedia_url", None) or metadata.get("wikipedia_url")
            results.append(
                {
                    "name": e.name,
                    "type": e.type,
                    "subtype": e.subtype,
                    "description": enriched_desc or (e.description or ""),
                    "wikipedia_url": wiki_url,
                    "enriched": bool(enriched_desc),
                }
            )

        return results
    except Exception as e:
        return [{"error": f"Entity search failed: {str(e)}"}]


async def _collapse_duplicate_entity_results(
    ctx: RunContext[AgentDeps],
    results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Deduplicate entity results by name.

    Note: SAME_AS relationship support is not yet implemented in the database.
    This function currently just deduplicates by exact name match.
    """
    if not results:
        return results

    # Simple deduplication by name
    seen_names = set()
    deduped = []
    for r in results:
        name = r.get("name")
        if name and name not in seen_names:
            seen_names.add(name)
            deduped.append(r)
            if len(deduped) >= limit:
                break

    return deduped


async def _collapse_duplicate_entities(
    ctx: RunContext[AgentDeps],
    entities: list,
    limit: int,
) -> list:
    """Deduplicate entities by name.

    Note: SAME_AS relationship support is not yet implemented in the database.
    This function currently just deduplicates by exact name match.
    """
    if not entities:
        return entities

    # Simple deduplication by name
    seen_names = set()
    deduped = []
    for e in entities:
        name = getattr(e, "name", None)
        if name and name not in seen_names:
            seen_names.add(name)
            deduped.append(e)
            if len(deduped) >= limit:
                break

    return deduped


async def get_entity_context(
    ctx: RunContext[AgentDeps],
    entity_name: str,
) -> dict[str, Any]:
    """Get full context about an entity including enrichment data and status.

    Uses vector search to find entities even with partial or fuzzy name matches.

    Args:
        ctx: The agent run context.
        entity_name: Name of entity (e.g., "Brian Chesky", "Airbnb").

    Returns:
        Entity details, enrichment status, Wikipedia summary (if available),
        and podcast mentions. The enrichment_status field indicates:
        - "enriched": Entity has Wikipedia/external data
        - "pending": Enrichment was attempted but no data found
        - "not_attempted": Entity has not been enriched yet
    """
    logger.info(f"[get_entity_context] Called with entity_name='{entity_name}'")

    if not ctx.deps.client:
        logger.warning("[get_entity_context] Memory client not available!")
        return {"error": "Memory client not available"}

    try:
        # First try exact name match
        logger.info(f"[get_entity_context] Trying exact name match for '{entity_name}'")
        entity = await ctx.deps.client.long_term.get_entity_by_name(entity_name)
        logger.info(f"[get_entity_context] Exact match result: {entity is not None}")

        # If not found, try vector search for fuzzy matching
        if not entity:
            entities = await ctx.deps.client.long_term.search_entities(
                query=entity_name,
                limit=5,
                threshold=0.5,  # Lower threshold for name matching
            )
            if entities:
                # Use the best match
                entity = entities[0]

        if not entity:
            return {
                "error": f"Entity '{entity_name}' not found. Try searching with search_entities tool first."
            }

        # Get mentions via graph traversal (Entity <-[MENTIONS]- Message)
        graph = ctx.deps.client.backend.graph
        mention_nodes = await graph.traverse(
            "Entity",
            str(entity.id),
            relationship_types=["MENTIONS"],
            direction="incoming",
            target_labels=["Message"],
            limit=5,
        )

        mention_list = []
        for m in mention_nodes:
            metadata = m.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            content = m.get("content", "")
            session_id = metadata.get("session_id", "")
            episode_guest = metadata.get("episode_guest", "")
            if not episode_guest and session_id.startswith("lenny-podcast-"):
                episode_guest = session_id.replace("lenny-podcast-", "").replace("-", " ").title()
            mention_list.append(
                {
                    "content": content[:300] + "..." if len(content) > 300 else content,
                    "speaker": metadata.get("speaker", "Unknown"),
                    "episode": episode_guest or "Unknown",
                }
            )

        # Determine enrichment status
        enrichment_status = _get_enrichment_status(entity)

        # Get enrichment fields - check both direct attributes and metadata dict
        # The Entity model stores extra properties in metadata, not as direct attributes
        metadata = getattr(entity, "metadata", {}) or {}

        enriched_description = getattr(entity, "enriched_description", None) or metadata.get(
            "enriched_description"
        )
        wikipedia_url = getattr(entity, "wikipedia_url", None) or metadata.get("wikipedia_url")
        image_url = getattr(entity, "image_url", None) or metadata.get("image_url")
        enrichment_provider = getattr(entity, "enrichment_provider", None) or metadata.get(
            "enrichment_provider"
        )
        enriched_at = metadata.get("enriched_at")
        wikidata_id = getattr(entity, "wikidata_id", None) or metadata.get("wikidata_id")

        # Return in format expected by frontend EntityCard
        # Frontend expects: { entity: {...}, mentions: [...] } with enrichment fields at entity level
        return {
            "entity": {
                "id": str(entity.id) if hasattr(entity, "id") else entity.name,
                "name": entity.name,
                "type": entity.type,
                "subtype": entity.subtype,
                "description": entity.description,
                # Enrichment fields at top level for frontend compatibility
                "enriched_description": enriched_description,
                "wikipedia_url": wikipedia_url,
                "image_url": image_url,
                "wikidata_id": wikidata_id,
                # Additional enrichment metadata
                "enrichment_status": enrichment_status,
                "enrichment_provider": enrichment_provider,
                "enriched_at": str(enriched_at) if enriched_at else None,
            },
            "mentions": mention_list,
            "mention_count": len(mention_list),
        }
    except Exception as e:
        logger.exception(f"[get_entity_context] Exception for entity '{entity_name}': {e}")
        return {"error": f"Failed to get entity context: {str(e)}"}


def _get_enrichment_status(entity) -> str:
    """Determine the enrichment status of an entity."""
    # Get metadata dict for checking enrichment properties
    metadata = getattr(entity, "metadata", {}) or {}

    # Check if entity has enriched data
    enriched_description = getattr(entity, "enriched_description", None) or metadata.get(
        "enriched_description"
    )
    wikipedia_url = getattr(entity, "wikipedia_url", None) or metadata.get("wikipedia_url")

    if enriched_description or wikipedia_url:
        return "enriched"

    # Check if enrichment was attempted but failed
    if metadata.get("enrichment_attempted"):
        return "pending"  # Attempted but no data found

    return "not_attempted"


async def find_related_entities(
    ctx: RunContext[AgentDeps],
    entity_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find entities related to a given entity via the knowledge graph.

    Uses fuzzy name matching to find the entity, then returns co-occurring entities.

    Args:
        ctx: The agent run context.
        entity_name: Starting entity (e.g., "Airbnb").
        limit: Maximum number of related entities to return.

    Returns:
        List of related entities with co-occurrence counts.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # Resolve entity name using the high-level API
        resolved_name = await _resolve_entity_name(ctx, entity_name)
        if not resolved_name:
            return [{"error": f"Entity '{entity_name}' not found"}]

        # Use get_entity_relationships which uses backend-neutral traverse
        related = await ctx.deps.client.long_term.get_entity_relationships(resolved_name)

        results = []
        for entity, rel in related[:limit]:
            metadata = getattr(entity, "metadata", {}) or {}
            enriched_desc = getattr(entity, "enriched_description", None) or metadata.get(
                "enriched_description"
            )
            results.append(
                {
                    "name": entity.name,
                    "type": entity.type,
                    "subtype": entity.subtype,
                    "enriched_description": enriched_desc,
                    "co_occurrences": 1,  # Relationship-based (not counted by co-occurrence)
                }
            )

        return results
    except Exception as e:
        return [{"error": f"Failed to find related entities: {str(e)}"}]


async def _resolve_entity_name(
    ctx: RunContext[AgentDeps],
    entity_name: str,
) -> str | None:
    """Resolve an entity name using multiple matching strategies.

    Returns the canonical entity name or None if not found.
    """
    # Try exact match first
    entity = await ctx.deps.client.long_term.get_entity_by_name(entity_name)
    if entity:
        return entity.name

    # Try vector search for fuzzy matching
    entities = await ctx.deps.client.long_term.search_entities(
        query=entity_name,
        limit=3,
        threshold=0.4,  # Lower threshold for name resolution
    )
    if entities:
        return entities[0].name

    return None


async def get_most_mentioned_entities(
    ctx: RunContext[AgentDeps],
    entity_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get the most frequently mentioned entities across all podcasts.

    Args:
        ctx: The agent run context.
        entity_type: Filter by PERSON, ORGANIZATION, CONCEPT, LOCATION, etc.
        limit: Number of results (default 10).

    Returns:
        List of top entities with mention counts.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # Query entity nodes and sort by mention_count property
        graph = ctx.deps.client.backend.graph
        filters: dict[str, Any] = {}
        if entity_type:
            filters["type"] = entity_type.upper()

        entities = await graph.query_nodes(
            "Entity",
            filters=filters if filters else None,
            order_by="mention_count",
            order_dir="desc",
            limit=limit,
        )

        results = []
        for e in entities:
            metadata = e.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            enriched_desc = e.get("enriched_description") or metadata.get(
                "enriched_description", ""
            )
            wiki_url = e.get("wikipedia_url") or metadata.get("wikipedia_url")
            results.append(
                {
                    "name": e.get("name", "Unknown"),
                    "type": e.get("type", "Unknown"),
                    "subtype": e.get("subtype"),
                    "description": enriched_desc,
                    "wikipedia_url": wiki_url,
                    "mentions": e.get("mention_count", 0),
                }
            )

        return results
    except Exception as e:
        return [{"error": f"Failed to get top entities: {str(e)}"}]


# =============================================================================
# Location Query Tools (Map View Integration)
# =============================================================================


async def search_locations(
    ctx: RunContext[AgentDeps],
    query: str | None = None,
    episode_guest: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for locations mentioned in podcasts.

    Args:
        ctx: The agent run context.
        query: Optional search term (e.g., "Silicon Valley", "Europe").
        episode_guest: Optional guest name to filter by episode.
        limit: Maximum number of results to return.

    Returns:
        Locations mentioned in podcasts. Includes coordinates if available for map visualization.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        session_id = None
        if episode_guest:
            session_id = _guest_to_session_id(episode_guest)

        # Use the backend-neutral get_locations utility
        locations = await ctx.deps.client.get_locations(
            session_id=session_id,
            has_coordinates=True,
            limit=limit,
        )

        # Apply optional query filter client-side
        if query:
            query_lower = query.lower()
            locations = [
                loc for loc in locations
                if query_lower in loc.get("name", "").lower()
            ]

        return [
            {
                "name": loc.get("name"),
                "type": loc.get("type", "LOCATION"),
                "subtype": loc.get("subtype"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "description": loc.get("enriched_description") or loc.get("description") or "",
            }
            for loc in locations[:limit]
        ]
    except Exception as e:
        return [{"error": f"Location search failed: {str(e)}"}]


async def find_locations_near(
    ctx: RunContext[AgentDeps],
    location_name: str,
    radius_km: float = 500.0,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find other locations mentioned near a given location.

    Args:
        ctx: The agent run context.
        location_name: Reference location (e.g., "San Francisco").
        radius_km: Search radius in kilometers (default 500km).
        limit: Maximum number of results to return.

    Returns:
        Nearby locations with distance information.
        Results can be visualized on the map with the radius overlay.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # First get the reference location
        ref_locations = await ctx.deps.client.get_locations(has_coordinates=True, limit=500)
        ref_loc = None
        for loc in ref_locations:
            if location_name.lower() in loc.get("name", "").lower():
                ref_loc = loc
                break

        if not ref_loc or ref_loc.get("latitude") is None or ref_loc.get("longitude") is None:
            return [{"error": f"Location '{location_name}' not found with coordinates"}]

        # Use geospatial search
        nearby = await ctx.deps.client.search_locations_near(
            latitude=ref_loc["latitude"],
            longitude=ref_loc["longitude"],
            radius_km=radius_km,
            limit=limit,
        )

        return [
            {
                "name": loc.get("name"),
                "type": loc.get("type", "LOCATION"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "country": loc.get("country"),
                "distance_km": round(
                    _haversine_distance(
                        ref_loc["latitude"],
                        ref_loc["longitude"],
                        loc.get("latitude") or 0,
                        loc.get("longitude") or 0,
                    ),
                    1,
                )
                if loc.get("latitude") and loc.get("longitude")
                else None,
            }
            for loc in nearby
            if loc.get("name", "").lower()
            != ref_loc.get("name", "").lower()  # Exclude reference location
        ]
    except Exception as e:
        return [{"error": f"Nearby location search failed: {str(e)}"}]


async def get_episode_locations(
    ctx: RunContext[AgentDeps],
    episode_guest: str,
) -> list[dict[str, Any]]:
    """Get all locations mentioned in a specific episode.

    Args:
        ctx: The agent run context.
        episode_guest: Guest name (e.g., "Brian Chesky").

    Returns:
        A geographic profile of the episode's content with all locations mentioned.
        Includes coordinates if available for map visualization.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        session_id = _guest_to_session_id(episode_guest)

        # Use the backend-neutral get_locations utility with session filter
        locations = await ctx.deps.client.get_locations(
            session_id=session_id,
            has_coordinates=True,
            limit=100,
        )

        return [
            {
                "name": loc.get("name"),
                "type": loc.get("type", "LOCATION"),
                "subtype": loc.get("subtype"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "description": loc.get("enriched_description") or loc.get("description") or "",
                "mentions": loc.get("mention_count", 0),
            }
            for loc in locations
        ]
    except Exception as e:
        return [{"error": f"Failed to get episode locations: {str(e)}"}]


async def find_location_path(
    ctx: RunContext[AgentDeps],
    from_location: str,
    to_location: str,
) -> dict[str, Any]:
    """Find the shortest path between two locations in the knowledge graph.

    Args:
        ctx: The agent run context.
        from_location: Starting location name.
        to_location: Destination location name.

    Returns:
        The graph path showing how locations are connected through
        entities, messages, and conversations. The frontend map can visualize
        this as a polyline overlay connecting the locations.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        # Find both locations
        all_locs = await ctx.deps.client.get_locations(has_coordinates=True, limit=500)

        from_loc = None
        to_loc = None
        for loc in all_locs:
            name = loc.get("name", "").lower()
            if from_location.lower() in name and from_loc is None:
                from_loc = loc
            if to_location.lower() in name and to_loc is None:
                to_loc = loc

        if not from_loc:
            return {"error": f"Location '{from_location}' not found"}
        if not to_loc:
            return {"error": f"Location '{to_location}' not found"}

        # Calculate direct distance (graph path finding not available in backend-neutral API)
        dist = _haversine_distance(
            from_loc.get("latitude", 0),
            from_loc.get("longitude", 0),
            to_loc.get("latitude", 0),
            to_loc.get("longitude", 0),
        )

        return {
            "from_location": {
                "name": from_loc.get("name"),
                "latitude": from_loc.get("latitude"),
                "longitude": from_loc.get("longitude"),
            },
            "to_location": {
                "name": to_loc.get("name"),
                "latitude": to_loc.get("latitude"),
                "longitude": to_loc.get("longitude"),
            },
            "distance_km": round(dist, 1),
            "note": "Direct distance shown. Graph path traversal requires Neo4j backend.",
        }
    except Exception as e:
        return {"error": f"Path finding failed: {str(e)}"}


async def get_location_clusters(
    ctx: RunContext[AgentDeps],
    episode_guest: str | None = None,
) -> list[dict[str, Any]]:
    """Analyze geographic clusters of locations mentioned in podcasts.

    Args:
        ctx: The agent run context.
        episode_guest: Optional guest to filter by.

    Returns:
        Location density information useful for understanding
        which geographic regions are most discussed. Works with the
        frontend heatmap visualization.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        session_id = None
        if episode_guest:
            session_id = _guest_to_session_id(episode_guest)

        locations = await ctx.deps.client.get_locations(
            session_id=session_id,
            has_coordinates=True,
            limit=500,
        )

        # Group by country for cluster analysis
        country_clusters: dict[str, list[dict]] = {}
        for loc in locations:
            country = loc.get("country") or "Unknown"
            if country not in country_clusters:
                country_clusters[country] = []
            country_clusters[country].append(
                {
                    "name": loc.get("name"),
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                }
            )

        return [
            {
                "country": country,
                "location_count": len(locs),
                "locations": locs[:5],  # Top 5 locations per country
                "center_lat": sum(l["latitude"] for l in locs if l["latitude"]) / len(locs)
                if locs
                else None,
                "center_lon": sum(l["longitude"] for l in locs if l["longitude"]) / len(locs)
                if locs
                else None,
            }
            for country, locs in sorted(country_clusters.items(), key=lambda x: -len(x[1]))
        ]
    except Exception as e:
        return [{"error": f"Cluster analysis failed: {str(e)}"}]


async def calculate_location_distances(
    ctx: RunContext[AgentDeps],
    locations: list[str],
) -> list[dict[str, Any]]:
    """Calculate distances between multiple locations.

    Args:
        ctx: The agent run context.
        locations: List of location names to measure between.

    Returns:
        Pairwise distances in kilometers using great-circle calculation.
        Useful for understanding the geographic scope of discussions.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    if len(locations) < 2:
        return [{"error": "At least 2 locations are required"}]

    try:
        # Get all locations with coordinates
        all_locs = await ctx.deps.client.get_locations(has_coordinates=True, limit=500)

        # Find matching locations
        found_locs: list[dict] = []
        for loc_name in locations:
            for loc in all_locs:
                if (
                    loc_name.lower() in loc.get("name", "").lower()
                    and loc.get("latitude")
                    and loc.get("longitude")
                ):
                    found_locs.append(
                        {
                            "name": loc.get("name"),
                            "latitude": loc.get("latitude"),
                            "longitude": loc.get("longitude"),
                        }
                    )
                    break

        if len(found_locs) < 2:
            return [
                {"error": f"Need at least 2 locations with coordinates, found {len(found_locs)}"}
            ]

        # Calculate pairwise distances
        distances = []
        for i in range(len(found_locs)):
            for j in range(i + 1, len(found_locs)):
                loc1 = found_locs[i]
                loc2 = found_locs[j]
                dist = _haversine_distance(
                    loc1["latitude"],
                    loc1["longitude"],
                    loc2["latitude"],
                    loc2["longitude"],
                )
                distances.append(
                    {
                        "from": loc1["name"],
                        "to": loc2["name"],
                        "distance_km": round(dist, 1),
                    }
                )

        return sorted(distances, key=lambda x: x["distance_km"])
    except Exception as e:
        return [{"error": f"Distance calculation failed: {str(e)}"}]


# =============================================================================
# User Preferences and Reasoning Memory Tools
# =============================================================================


async def get_user_preferences(
    ctx: RunContext[AgentDeps],
) -> list[dict[str, Any]]:
    """Get the current user's stored preferences.

    Args:
        ctx: The agent run context.

    Returns:
        Preferences including content interests, format preferences,
        and historical patterns. Use this to personalize responses.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        preferences = await ctx.deps.client.long_term.search_preferences(
            query="",
            limit=20,
        )

        return [
            {
                "category": p.category,
                "preference": p.preference,
                "confidence": p.confidence,
                "source": p.source,
            }
            for p in preferences
        ]
    except Exception as e:
        return [{"error": f"Failed to get preferences: {str(e)}"}]


async def find_similar_past_queries(
    ctx: RunContext[AgentDeps],
    current_query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find similar queries from past conversations and their successful resolutions.

    Args:
        ctx: The agent run context.
        current_query: The current user query.
        limit: Maximum number of similar traces to return.

    Returns:
        Past reasoning traces that solved similar problems.
        Useful for learning from successful past interactions.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        traces = await ctx.deps.client.reasoning.get_similar_traces(
            task=current_query,
            limit=limit,
            success_only=True,
        )

        return [
            {
                "task": t.task,
                "outcome": t.outcome,
                "success": t.success,
                "steps_count": len(t.steps) if t.steps else 0,
            }
            for t in traces
        ]
    except Exception as e:
        return [{"error": f"Failed to find similar traces: {str(e)}"}]


# =============================================================================
# Enhanced Reasoning Memory Tools (NEW)
# =============================================================================


async def learn_from_similar_task(
    ctx: RunContext[AgentDeps],
    task_description: str,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Get full reasoning traces from similar past tasks for few-shot learning.

    Unlike find_similar_past_queries which returns summaries, this returns
    the complete reasoning steps so the agent can learn the approach.

    Args:
        ctx: The agent run context.
        task_description: Description of the current task.
        limit: Number of similar traces to return.

    Returns:
        Complete reasoning traces including all steps and tool calls.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        traces = await ctx.deps.client.reasoning.get_similar_traces(
            task=task_description,
            limit=limit,
            success_only=True,
            threshold=0.6,  # Lower threshold to find more potential matches
        )

        results = []
        for trace in traces:
            # Get the full trace with steps
            full_trace = await ctx.deps.client.reasoning.get_trace_with_steps(trace.id)
            if not full_trace:
                continue

            steps_data = []
            for step in full_trace.steps or []:
                step_info = {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                }
                # Include tool calls if present
                if step.tool_calls:
                    step_info["tool_calls"] = [
                        {
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "status": tc.status.value if tc.status else "unknown",
                            "duration_ms": tc.duration_ms,
                        }
                        for tc in step.tool_calls
                    ]
                steps_data.append(step_info)

            results.append(
                {
                    "task": full_trace.task,
                    "outcome": full_trace.outcome,
                    "success": full_trace.success,
                    "similarity": trace.metadata.get("similarity", 0) if trace.metadata else 0,
                    "steps": steps_data,
                }
            )

        return results
    except Exception as e:
        return [{"error": f"Failed to get similar traces: {str(e)}"}]


async def get_tool_usage_patterns(
    ctx: RunContext[AgentDeps],
    tool_name: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Analyze tool usage patterns to understand which tools are most effective.

    Returns statistics about tool calls including success rates, average durations,
    and common tool sequences.

    Args:
        ctx: The agent run context.
        tool_name: Optional specific tool to analyze.
        limit: Maximum number of tools to include in analysis.

    Returns:
        Tool usage statistics and patterns.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        # Get pre-aggregated tool stats
        stats = await ctx.deps.client.reasoning.get_tool_stats(tool_name=tool_name)

        tool_data = []
        for s in stats[:limit]:
            tool_info = {
                "tool_name": s.tool_name,
                "total_calls": s.total_calls,
                "success_count": s.success_count,
                "failure_count": s.failure_count,
                "success_rate": round(s.success_rate * 100, 1) if s.success_rate else 0,
                "avg_duration_ms": round(s.avg_duration_ms, 1) if s.avg_duration_ms else None,
            }
            tool_data.append(tool_info)

        # Sort by success rate and usage
        tool_data.sort(key=lambda x: (x["success_rate"], x["total_calls"]), reverse=True)

        return {
            "tools": tool_data,
            "total_tools_analyzed": len(tool_data),
            "recommendation": _get_tool_recommendation(tool_data),
        }
    except Exception as e:
        return {"error": f"Failed to get tool patterns: {str(e)}"}


def _get_tool_recommendation(tool_data: list[dict]) -> str:
    """Generate a recommendation based on tool usage patterns."""
    if not tool_data:
        return "No tool usage data available yet."

    # Find best performing tools
    best_tools = [t for t in tool_data if t["success_rate"] >= 90 and t["total_calls"] >= 5]
    if best_tools:
        top_tool = best_tools[0]
        return f"'{top_tool['tool_name']}' has highest success rate ({top_tool['success_rate']}%) with {top_tool['total_calls']} calls."

    # Find most used tools
    most_used = sorted(tool_data, key=lambda x: x["total_calls"], reverse=True)
    if most_used:
        return f"'{most_used[0]['tool_name']}' is most frequently used ({most_used[0]['total_calls']} calls)."

    return "Continue using tools to build usage patterns."


async def get_session_reasoning_history(
    ctx: RunContext[AgentDeps],
    session_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get reasoning traces from a session to understand the conversation history.

    Args:
        ctx: The agent run context.
        session_id: Session ID to query. Defaults to current session.
        limit: Maximum number of traces to return.

    Returns:
        List of reasoning traces from the session.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        target_session = session_id or ctx.deps.session_id
        traces = await ctx.deps.client.reasoning.get_session_traces(
            session_id=target_session,
            limit=limit,
        )

        return [
            {
                "id": str(t.id),
                "task": t.task,
                "outcome": t.outcome,
                "success": t.success,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "steps_count": len(t.steps) if t.steps else 0,
            }
            for t in traces
        ]
    except Exception as e:
        return [{"error": f"Failed to get session traces: {str(e)}"}]


# =============================================================================
# Entity Management Tools (NEW)
# =============================================================================


async def find_duplicate_entities(
    ctx: RunContext[AgentDeps],
    entity_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find potential duplicate entities that may need merging.

    Useful for data quality improvement and entity resolution.

    Args:
        ctx: The agent run context.
        entity_type: Filter by entity type (PERSON, ORGANIZATION, etc.)
        limit: Maximum number of duplicate pairs to return.

    Returns:
        Pairs of potentially duplicate entities with similarity scores.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # Use the library's deduplication capabilities
        duplicates = await ctx.deps.client.long_term.find_potential_duplicates(
            entity_type=entity_type,
            limit=limit,
        )

        return [
            {
                "entity1": {
                    "id": str(d.entity1.id),
                    "name": d.entity1.name,
                    "type": d.entity1.type,
                },
                "entity2": {
                    "id": str(d.entity2.id),
                    "name": d.entity2.name,
                    "type": d.entity2.type,
                },
                "similarity": round(d.similarity, 3),
                "status": d.status,
            }
            for d in duplicates
        ]
    except AttributeError:
        # Method may not exist in older library versions
        # Fallback: Use custom Cypher to find similar names
        return await _find_duplicates_fallback(ctx, entity_type, limit)
    except Exception as e:
        return [{"error": f"Failed to find duplicates: {str(e)}"}]


async def _find_duplicates_fallback(
    ctx: RunContext[AgentDeps],
    entity_type: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Fallback duplicate detection using name comparison."""
    try:
        filters: dict[str, Any] = {}
        if entity_type:
            filters["type"] = entity_type.upper()

        entities = await ctx.deps.client.backend.graph.query_nodes(
            "Entity",
            filters=filters if filters else None,
            limit=500,
        )

        # Simple client-side duplicate detection by substring matching
        pairs: list[dict[str, Any]] = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.get("type") != e2.get("type"):
                    continue
                n1 = (e1.get("name") or "").lower()
                n2 = (e2.get("name") or "").lower()
                if n1 and n2 and (n1 in n2 or n2 in n1):
                    pairs.append(
                        {
                            "entity1": {"id": e1.get("id", ""), "name": e1.get("name", ""), "type": e1.get("type", "")},
                            "entity2": {"id": e2.get("id", ""), "name": e2.get("name", ""), "type": e2.get("type", "")},
                            "similarity": 0.8,
                            "status": "pending",
                        }
                    )
                    if len(pairs) >= limit:
                        return pairs
        return pairs
    except Exception as e:
        return [{"error": f"Fallback duplicate detection failed: {str(e)}"}]


async def get_entity_provenance(
    ctx: RunContext[AgentDeps],
    entity_name: str,
) -> dict[str, Any]:
    """Get the provenance (source) information for an entity.

    Shows which messages the entity was extracted from and extraction confidence.

    Args:
        ctx: The agent run context.
        entity_name: Name of the entity to get provenance for.

    Returns:
        Source messages, extraction dates, and confidence scores.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        # Resolve entity name
        entity = await ctx.deps.client.long_term.get_entity_by_name(entity_name)
        if not entity:
            entities = await ctx.deps.client.long_term.search_entities(
                query=entity_name, limit=1, threshold=0.5,
            )
            if entities:
                entity = entities[0]

        if not entity:
            return {"error": f"Entity '{entity_name}' not found"}

        # Use the high-level provenance method
        provenance = await ctx.deps.client.long_term.get_entity_provenance(entity)
        metadata = getattr(entity, "metadata", {}) or {}

        sources = []
        for s in provenance.get("sources", [])[:5]:
            sources.append(
                {
                    "message_id": s.get("message_id"),
                    "content_preview": (s.get("content") or "")[:150],
                    "confidence": s.get("confidence"),
                }
            )

        return {
            "entity": {
                "name": entity.name,
                "type": entity.type,
                "created_at": entity.created_at.isoformat() if entity.created_at else None,
                "confidence": entity.confidence,
            },
            "enrichment": {
                "provider": getattr(entity, "enrichment_provider", None) or metadata.get("enrichment_provider"),
                "enriched_at": str(metadata.get("enriched_at")) if metadata.get("enriched_at") else None,
            },
            "sources": sources,
            "total_mentions": len(sources),
        }
    except Exception as e:
        return {"error": f"Failed to get entity provenance: {str(e)}"}


async def trigger_entity_enrichment(
    ctx: RunContext[AgentDeps],
    entity_name: str,
    provider: str = "wikimedia",
) -> dict[str, Any]:
    """Request enrichment for an entity from Wikipedia or other providers.

    Args:
        ctx: The agent run context.
        entity_name: Name of the entity to enrich.
        provider: Enrichment provider ("wikimedia" or "diffbot").

    Returns:
        Enrichment status and data if available.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        # First, find the entity
        entity = await ctx.deps.client.long_term.get_entity_by_name(entity_name)
        if not entity:
            return {"error": f"Entity '{entity_name}' not found"}

        # Get metadata dict for checking enrichment properties
        # Entity model stores extra properties in metadata, not properties
        metadata = getattr(entity, "metadata", {}) or {}

        # Check if already enriched - check both direct attributes and metadata
        enriched_description = getattr(entity, "enriched_description", None) or metadata.get(
            "enriched_description"
        )
        wikipedia_url = getattr(entity, "wikipedia_url", None) or metadata.get("wikipedia_url")
        image_url = getattr(entity, "image_url", None) or metadata.get("image_url")
        enrichment_provider = getattr(entity, "enrichment_provider", None) or metadata.get(
            "enrichment_provider"
        )
        enriched_at = metadata.get("enriched_at")

        if enriched_description:
            return {
                "status": "already_enriched",
                "entity_name": entity.name,
                "enrichment_provider": enrichment_provider,
                "enriched_at": str(enriched_at) if enriched_at else None,
                "description": enriched_description,
                "wikipedia_url": wikipedia_url,
                "image_url": image_url,
            }

        # Trigger enrichment if enrichment service is available
        # NOTE: This requires the enrichment service to be running
        # For now, return status indicating enrichment needed
        return {
            "status": "enrichment_needed",
            "entity_name": entity.name,
            "entity_type": entity.type,
            "message": f"Entity '{entity_name}' needs enrichment. Run the enrichment script or enable background enrichment.",
            "suggestion": "Use `make enrich` to enrich entities with Wikipedia data.",
        }
    except Exception as e:
        return {"error": f"Failed to trigger enrichment: {str(e)}"}


# =============================================================================
# Conversation & Summary Tools (NEW)
# =============================================================================


async def get_conversation_context(
    ctx: RunContext[AgentDeps],
    limit: int = 10,
    include_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Get recent conversation history for context.

    Args:
        ctx: The agent run context.
        limit: Maximum number of messages to return.
        include_tool_calls: Whether to include tool call details.

    Returns:
        Recent messages in the current conversation.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        conv = await ctx.deps.client.short_term.get_conversation(
            session_id=ctx.deps.session_id,
            limit=limit,
        )

        results = []
        for msg in conv.messages:
            msg_data = {
                "role": msg.role,
                "content": msg.content[:500] if len(msg.content) > 500 else msg.content,
                "timestamp": msg.metadata.get("timestamp") if msg.metadata else None,
            }
            if include_tool_calls and hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_data["tool_calls"] = [
                    {"name": tc.get("name"), "status": tc.get("status")} for tc in msg.tool_calls
                ]
            results.append(msg_data)

        return results
    except Exception as e:
        return [{"error": f"Failed to get conversation context: {str(e)}"}]


async def list_podcast_sessions(
    ctx: RunContext[AgentDeps],
    sort_by: str = "message_count",
    order: str = "desc",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List available podcast sessions with metadata.

    Args:
        ctx: The agent run context.
        sort_by: Sort field ("message_count", "created_at", "updated_at").
        order: Sort order ("asc" or "desc").
        limit: Maximum number of sessions.

    Returns:
        List of sessions with message counts and metadata.
    """
    if not ctx.deps.client:
        return [{"error": "Memory client not available"}]

    try:
        # Map sort_by to the API's expected field names
        valid_order_by = sort_by if sort_by in ("message_count", "created_at", "updated_at") else "message_count"
        valid_order_dir = order if order in ("asc", "desc") else "desc"

        sessions = await ctx.deps.client.short_term.list_sessions(
            prefix="lenny-podcast-",
            limit=limit,
            order_by=valid_order_by,
            order_dir=valid_order_dir,
        )
        return [
            {
                "session_id": s.session_id,
                "message_count": s.message_count,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                "preview": (s.first_message_preview or "")[:100] or None,
            }
            for s in sessions
        ]
    except Exception as e:
        return [{"error": f"Failed to list sessions: {str(e)}"}]


async def get_episode_summary(
    ctx: RunContext[AgentDeps],
    episode_guest: str,
) -> dict[str, Any]:
    """Get a summary of a podcast episode including key topics and entities.

    Args:
        ctx: The agent run context.
        episode_guest: Guest name (e.g., "Brian Chesky").

    Returns:
        Episode summary with key topics, entities, and highlights.
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        session_id = _guest_to_session_id(episode_guest)

        # Get conversation summary if available
        try:
            summary = await ctx.deps.client.short_term.get_conversation_summary(
                session_id=session_id,
            )
            if summary:
                return {
                    "episode_guest": episode_guest,
                    "session_id": session_id,
                    "summary": summary.summary,
                    "key_topics": summary.key_topics,
                    "key_entities": summary.key_entities,
                    "generated_at": summary.generated_at.isoformat()
                    if summary.generated_at
                    else None,
                }
        except AttributeError:
            pass

        # Fallback: Generate summary from session info and message search
        graph = ctx.deps.client.backend.graph
        conv_node = await graph.get_node("Conversation", filters={"session_id": session_id})
        if not conv_node:
            return {"error": f"Episode with guest '{episode_guest}' not found"}

        # Count messages via traverse
        msg_nodes = await graph.traverse(
            "Conversation",
            conv_node["id"],
            relationship_types=["HAS_MESSAGE"],
            target_labels=["Message"],
            direction="outgoing",
            limit=10,
        )

        # Get entity mentions from a sample of messages
        entity_names: list[str] = []
        for msg in msg_nodes[:5]:
            try:
                entities_for_msg = await ctx.deps.client.long_term.get_entities_from_message(
                    msg.get("id", "")
                )
                for ent, _ in entities_for_msg:
                    if ent.name not in entity_names:
                        entity_names.append(ent.name)
            except Exception:
                pass

        return {
            "episode_guest": episode_guest,
            "session_id": session_id,
            "title": conv_node.get("title"),
            "message_count": len(msg_nodes),
            "key_entities": entity_names[:10],
            "summary": None,
            "note": "Full AI summary requires conversation summarization feature.",
        }
    except Exception as e:
        return {"error": f"Failed to get episode summary: {str(e)}"}


# =============================================================================
# Memory Graph Search Tool (NEW)
# =============================================================================


async def memory_graph_search(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 10,
    include_related_entities: bool = True,
    max_related_per_entity: int = 5,
) -> dict[str, Any]:
    """Search memory using vector similarity, then traverse the graph to find entities.

    This tool combines semantic search with knowledge graph traversal:
    1. Vector search finds semantically similar messages
    2. Graph traversal finds entities mentioned in those messages
    3. Expands to find ALL entities related to the mentioned ones

    Use this tool when you want to explore the knowledge graph starting from a
    semantic concept rather than a specific entity name.

    Args:
        ctx: The agent run context.
        query: Natural language search query (e.g., "product-market fit", "scaling teams")
        limit: Maximum number of messages to find via vector search (default 10)
        include_related_entities: Whether to also find entities related to the mentioned ones
        max_related_per_entity: Maximum related entities to include per found entity (default 5)

    Returns:
        A graph structure with:
        - nodes: Messages found via vector search + Entities mentioned + Related entities
        - relationships: MENTIONS links (Message->Entity) + RELATED_TO links (Entity->Entity)
        - metadata: Search scores, entity types, co-occurrence counts
    """
    if not ctx.deps.client:
        return {"error": "Memory client not available"}

    try:
        # Step 1: Vector search for similar messages
        messages = await ctx.deps.client.short_term.search_messages(
            query=query,
            limit=limit,
            threshold=0.5,
            metadata_filters={"source": "lenny_podcast"},
        )

        if not messages:
            return {
                "query": query,
                "nodes": [],
                "relationships": [],
                "summary": {
                    "messages_found": 0,
                    "entities_found": 0,
                    "relationships_found": 0,
                },
                "message": f"No messages found matching '{query}'",
            }

        graph = ctx.deps.client.backend.graph

        # Build graph structure
        nodes: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        seen_rel_ids: set[str] = set()
        mentioned_entity_ids: list[str] = []

        # Add message nodes with full content
        for msg in messages:
            msg_id = f"msg-{msg.id}"
            if msg_id not in seen_node_ids:
                metadata = msg.metadata or {}
                content = msg.content
                if len(content) > 300:
                    content = content[:300] + "..."
                nodes.append(
                    {
                        "id": msg_id,
                        "label": metadata.get("speaker", "Message"),
                        "type": "Message",
                        "properties": {
                            "content": content,
                            "speaker": metadata.get("speaker"),
                            "episode": metadata.get("episode_guest"),
                            "similarity": round(metadata.get("similarity", 0), 3),
                        },
                    }
                )
                seen_node_ids.add(msg_id)

        # Step 2: Find entities mentioned in these messages via traverse
        for msg in messages:
            try:
                entities_for_msg = await ctx.deps.client.long_term.get_entities_from_message(
                    str(msg.id)
                )
                msg_node_id = f"msg-{msg.id}"
                for entity, extraction_info in entities_for_msg:
                    entity_id = f"entity-{entity.id}"
                    raw_entity_id = str(entity.id)

                    if entity_id not in seen_node_ids:
                        metadata = getattr(entity, "metadata", {}) or {}
                        enriched_desc = getattr(entity, "enriched_description", None) or metadata.get(
                            "enriched_description"
                        )
                        nodes.append(
                            {
                                "id": entity_id,
                                "label": entity.name,
                                "type": entity.type or "Entity",
                                "properties": {
                                    "subtype": entity.subtype,
                                    "enriched_description": enriched_desc,
                                },
                            }
                        )
                        seen_node_ids.add(entity_id)
                        mentioned_entity_ids.append(raw_entity_id)

                    # Add MENTIONS relationship
                    rel_id = f"mentions-{msg_node_id}-{entity_id}"
                    if rel_id not in seen_rel_ids:
                        relationships.append(
                            {
                                "id": rel_id,
                                "from": msg_node_id,
                                "to": entity_id,
                                "type": "MENTIONS",
                            }
                        )
                        seen_rel_ids.add(rel_id)
            except Exception:
                pass  # Skip messages where entity lookup fails

        # Step 3: Find related entities via traverse
        if include_related_entities and mentioned_entity_ids:
            for raw_eid in mentioned_entity_ids[:10]:  # Limit to avoid too many traversals
                try:
                    related_nodes = await graph.traverse(
                        "Entity",
                        raw_eid,
                        relationship_types=["RELATED_TO"],
                        direction="both",
                        target_labels=["Entity"],
                        include_edges=True,
                        limit=max_related_per_entity,
                    )

                    source_node_id = f"entity-{raw_eid}"
                    for related_row in related_nodes:
                        edge_data = related_row.get("_edge", {})
                        entity_data = {k: v for k, v in related_row.items() if k != "_edge"}
                        related_raw_id = entity_data.get("id")
                        if not related_raw_id:
                            continue

                        related_id = f"entity-{related_raw_id}"

                        if related_id not in seen_node_ids:
                            rel_metadata = entity_data.get("metadata", {})
                            if isinstance(rel_metadata, str):
                                try:
                                    rel_metadata = json.loads(rel_metadata)
                                except (json.JSONDecodeError, TypeError):
                                    rel_metadata = {}
                            enriched_desc = entity_data.get("enriched_description") or rel_metadata.get(
                                "enriched_description"
                            )
                            nodes.append(
                                {
                                    "id": related_id,
                                    "label": entity_data.get("name", "Unknown"),
                                    "type": entity_data.get("type", "Entity"),
                                    "properties": {
                                        "subtype": entity_data.get("subtype"),
                                        "enriched_description": enriched_desc,
                                    },
                                }
                            )
                            seen_node_ids.add(related_id)

                        # Add RELATED_TO relationship
                        rel_id = f"related-{source_node_id}-{related_id}"
                        reverse_rel_id = f"related-{related_id}-{source_node_id}"
                        if rel_id not in seen_rel_ids and reverse_rel_id not in seen_rel_ids:
                            relationships.append(
                                {
                                    "id": rel_id,
                                    "from": source_node_id,
                                    "to": related_id,
                                    "type": "RELATED_TO",
                                    "properties": {},
                                }
                            )
                            seen_rel_ids.add(rel_id)
                except Exception:
                    pass  # Skip entities where traversal fails

        # Step 4: Find additional messages for entities via traverse
        all_entity_ids = [n["id"].replace("entity-", "") for n in nodes if n["type"] != "Message"]
        existing_msg_ids = {n["id"].replace("msg-", "") for n in nodes if n["type"] == "Message"}

        for raw_eid in all_entity_ids[:15]:  # Limit traversals
            try:
                msg_nodes = await graph.traverse(
                    "Entity",
                    raw_eid,
                    relationship_types=["MENTIONS"],
                    direction="incoming",
                    target_labels=["Message"],
                    limit=2,
                )
                entity_id = f"entity-{raw_eid}"

                for msg_row in msg_nodes:
                    msg_raw_id = msg_row.get("id", "")
                    if msg_raw_id in existing_msg_ids:
                        continue

                    msg_id = f"msg-{msg_raw_id}"
                    if msg_id not in seen_node_ids:
                        metadata = msg_row.get("metadata", {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except (json.JSONDecodeError, TypeError):
                                metadata = {}

                        content = msg_row.get("content", "")
                        if len(content) > 300:
                            content = content[:300] + "..."

                        nodes.append(
                            {
                                "id": msg_id,
                                "label": metadata.get("speaker", "Message"),
                                "type": "Message",
                                "properties": {
                                    "content": content,
                                    "speaker": metadata.get("speaker"),
                                    "episode": metadata.get("episode_guest"),
                                    "source": "entity_connection",
                                },
                            }
                        )
                        seen_node_ids.add(msg_id)
                        existing_msg_ids.add(msg_raw_id)

                    # Add MENTIONS relationship
                    rel_id = f"mentions-{msg_id}-{entity_id}"
                    if rel_id not in seen_rel_ids:
                        relationships.append(
                            {
                                "id": rel_id,
                                "from": msg_id,
                                "to": entity_id,
                                "type": "MENTIONS",
                            }
                        )
                        seen_rel_ids.add(rel_id)
            except Exception:
                pass

        # Step 5: Remove disconnected nodes
        connected_node_ids: set[str] = set()
        for rel in relationships:
            connected_node_ids.add(rel["from"])
            connected_node_ids.add(rel["to"])

        nodes = [n for n in nodes if n["id"] in connected_node_ids]

        return {
            "query": query,
            "nodes": nodes,
            "relationships": relationships,
            "summary": {
                "messages_found": len([n for n in nodes if n["type"] == "Message"]),
                "entities_found": len([n for n in nodes if n["type"] != "Message"]),
                "relationships_found": len(relationships),
            },
        }

    except Exception as e:
        logger.exception(f"[memory_graph_search] Error for query '{query}': {e}")
        return {"error": f"Memory graph search failed: {str(e)}"}
