"""Tests for entity deduplication on ingest."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from neo4j_agent_memory.memory.long_term import (
    DeduplicationConfig,
    DeduplicationResult,
    DeduplicationStats,
    DuplicateCandidate,
    LongTermMemory,
)


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeduplicationConfig()

        assert config.enabled is True
        assert config.auto_merge_threshold == 0.95
        assert config.flag_threshold == 0.85
        assert config.use_fuzzy_matching is True
        assert config.fuzzy_threshold == 0.9
        assert config.max_candidates == 10
        assert config.match_same_type_only is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeduplicationConfig(
            enabled=False,
            auto_merge_threshold=0.98,
            flag_threshold=0.80,
            use_fuzzy_matching=False,
            fuzzy_threshold=0.85,
            max_candidates=20,
            match_same_type_only=False,
        )

        assert config.enabled is False
        assert config.auto_merge_threshold == 0.98
        assert config.flag_threshold == 0.80
        assert config.use_fuzzy_matching is False
        assert config.fuzzy_threshold == 0.85
        assert config.max_candidates == 20
        assert config.match_same_type_only is False

    def test_invalid_threshold_order(self):
        """Test that auto_merge_threshold must be >= flag_threshold."""
        with pytest.raises(ValueError, match="auto_merge_threshold must be >= flag_threshold"):
            DeduplicationConfig(
                auto_merge_threshold=0.80,
                flag_threshold=0.90,
            )

    def test_invalid_flag_threshold_range(self):
        """Test flag_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="flag_threshold must be between 0 and 1"):
            DeduplicationConfig(flag_threshold=1.5)

        with pytest.raises(ValueError, match="flag_threshold must be between 0 and 1"):
            DeduplicationConfig(flag_threshold=-0.1)

    def test_invalid_auto_merge_threshold_range(self):
        """Test auto_merge_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="auto_merge_threshold must be between 0 and 1"):
            DeduplicationConfig(auto_merge_threshold=1.5)

    def test_invalid_fuzzy_threshold_range(self):
        """Test fuzzy_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="fuzzy_threshold must be between 0 and 1"):
            DeduplicationConfig(fuzzy_threshold=-0.1)


class TestDeduplicationResult:
    """Tests for DeduplicationResult."""

    def test_default_result(self):
        """Test default result values."""
        result = DeduplicationResult()

        assert result.is_duplicate is False
        assert result.action == "none"
        assert result.matched_entity_id is None
        assert result.matched_entity_name is None
        assert result.similarity_score == 0.0
        assert result.match_type is None

    def test_merged_result(self):
        """Test result for merged entity."""
        entity_id = uuid4()
        result = DeduplicationResult(
            is_duplicate=True,
            action="merged",
            matched_entity_id=entity_id,
            matched_entity_name="John Smith",
            similarity_score=0.96,
            match_type="embedding",
        )

        assert result.is_duplicate is True
        assert result.action == "merged"
        assert result.matched_entity_id == entity_id
        assert result.matched_entity_name == "John Smith"
        assert result.similarity_score == 0.96
        assert result.match_type == "embedding"

    def test_flagged_result(self):
        """Test result for flagged entity."""
        entity_id = uuid4()
        result = DeduplicationResult(
            is_duplicate=True,
            action="flagged",
            matched_entity_id=entity_id,
            matched_entity_name="J. Smith",
            similarity_score=0.88,
            match_type="both",
        )

        assert result.is_duplicate is True
        assert result.action == "flagged"
        assert result.similarity_score == 0.88
        assert result.match_type == "both"


class TestDeduplicationStats:
    """Tests for DeduplicationStats."""

    def test_default_stats(self):
        """Test default stats values."""
        stats = DeduplicationStats()

        assert stats.total_entities == 0
        assert stats.merged_entities == 0
        assert stats.same_as_relationships == 0
        assert stats.pending_reviews == 0

    def test_custom_stats(self):
        """Test custom stats values."""
        stats = DeduplicationStats(
            total_entities=100,
            merged_entities=10,
            same_as_relationships=15,
            pending_reviews=5,
        )

        assert stats.total_entities == 100
        assert stats.merged_entities == 10
        assert stats.same_as_relationships == 15
        assert stats.pending_reviews == 5


class TestDuplicateCandidate:
    """Tests for DuplicateCandidate."""

    def test_duplicate_candidate(self):
        """Test DuplicateCandidate creation."""
        entity_id = uuid4()
        candidate = DuplicateCandidate(
            entity_id=entity_id,
            entity_name="John Smith",
            canonical_name="John Q. Smith",
            entity_type="PERSON",
            similarity_score=0.92,
            fuzzy_score=0.95,
            relationship_status="pending",
        )

        assert candidate.entity_id == entity_id
        assert candidate.entity_name == "John Smith"
        assert candidate.canonical_name == "John Q. Smith"
        assert candidate.entity_type == "PERSON"
        assert candidate.similarity_score == 0.92
        assert candidate.fuzzy_score == 0.95
        assert candidate.relationship_status == "pending"


def _make_graph_backend_mock():
    """Create a mock GraphBackend client with all async methods."""
    client = MagicMock()
    client.vector_search = AsyncMock(return_value=[])
    client.upsert_node = AsyncMock(return_value={})
    client.get_node = AsyncMock(return_value=None)
    client.query_nodes = AsyncMock(return_value=[])
    client.link_nodes = AsyncMock(return_value={})
    client.traverse = AsyncMock(return_value=[])
    client.count_nodes = AsyncMock(return_value=0)
    client.update_node = AsyncMock(return_value={})
    client.unlink_nodes = AsyncMock(return_value=True)
    return client


class TestLongTermMemoryDeduplication:
    """Tests for LongTermMemory deduplication functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphBackend client."""
        return _make_graph_backend_mock()

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 384)
        return embedder

    @pytest.fixture
    def memory(self, mock_client, mock_embedder):
        """Create a LongTermMemory instance with mocked dependencies."""
        return LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(),
        )

    @pytest.fixture
    def memory_disabled_dedup(self, mock_client, mock_embedder):
        """Create a LongTermMemory instance with deduplication disabled."""
        return LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(enabled=False),
        )

    @pytest.mark.asyncio
    async def test_add_entity_no_duplicates(self, memory, mock_client):
        """Test adding entity when no duplicates exist."""
        # vector_search returns empty list (no duplicates)
        mock_client.vector_search.return_value = []

        entity, dedup_result = await memory.add_entity(
            name="John Smith",
            entity_type="PERSON",
        )

        assert entity.name == "John Smith"
        assert entity.type == "PERSON"
        assert dedup_result.is_duplicate is False
        assert dedup_result.action == "none"

    @pytest.mark.asyncio
    async def test_add_entity_auto_merge(self, memory, mock_client):
        """Test adding entity that triggers auto-merge."""
        existing_entity_id = str(uuid4())

        # vector_search returns a high-score match (above auto_merge_threshold)
        mock_client.vector_search.return_value = [
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "canonical_name": "John Smith",
                "type": "PERSON",
                "subtype": None,
                "description": None,
                "embedding": [0.1] * 384,
                "confidence": 1.0,
                "metadata": None,
                "merged_into": None,
                "_score": 0.96,
            }
        ]

        # get_node is called twice:
        # 1) _get_entity_by_id for the matched entity (to return it)
        # 2) _get_entity_by_id inside _add_alias_to_entity
        mock_client.get_node.side_effect = [
            # First call: return existing entity
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "canonical_name": "John Smith",
                "type": "PERSON",
                "subtype": None,
                "description": None,
                "embedding": [0.1] * 384,
                "confidence": 1.0,
                "metadata": json.dumps({"aliases": []}),
            },
            # Second call: _add_alias_to_entity calls _get_entity_by_id
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "canonical_name": "John Smith",
                "type": "PERSON",
                "subtype": None,
                "description": None,
                "embedding": [0.1] * 384,
                "confidence": 1.0,
                "metadata": json.dumps({"aliases": []}),
            },
        ]

        entity, dedup_result = await memory.add_entity(
            name="Jon Smith",  # Slightly different name
            entity_type="PERSON",
        )

        assert dedup_result.is_duplicate is True
        assert dedup_result.action == "merged"
        assert dedup_result.matched_entity_id == UUID(existing_entity_id)
        assert dedup_result.similarity_score == 0.96
        # The returned entity should be the existing one
        assert entity.name == "John Smith"

    @pytest.mark.asyncio
    async def test_add_entity_flagged_for_review(self, memory, mock_client):
        """Test adding entity that gets flagged for review."""
        existing_entity_id = str(uuid4())

        # vector_search returns a medium-score match (between flag and auto_merge)
        mock_client.vector_search.return_value = [
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "canonical_name": "John Smith",
                "type": "PERSON",
                "subtype": None,
                "description": None,
                "embedding": [0.1] * 384,
                "confidence": 1.0,
                "metadata": None,
                "merged_into": None,
                "_score": 0.88,
            }
        ]

        entity, dedup_result = await memory.add_entity(
            name="J. Smith",
            entity_type="PERSON",
        )

        assert dedup_result.is_duplicate is True
        assert dedup_result.action == "flagged"
        assert dedup_result.matched_entity_id == UUID(existing_entity_id)
        assert dedup_result.similarity_score == 0.88

        # Entity should still be created
        assert entity.name == "J. Smith"

        # Verify SAME_AS relationship was created via link_nodes
        link_calls = mock_client.link_nodes.call_args_list
        same_as_calls = [c for c in link_calls if "SAME_AS" in str(c)]
        assert len(same_as_calls) == 1

    @pytest.mark.asyncio
    async def test_add_entity_deduplication_disabled(self, memory_disabled_dedup, mock_client):
        """Test adding entity with deduplication disabled."""
        entity, dedup_result = await memory_disabled_dedup.add_entity(
            name="John Smith",
            entity_type="PERSON",
        )

        assert entity.name == "John Smith"
        assert dedup_result.is_duplicate is False
        assert dedup_result.action == "none"

        # Should not check for duplicates (no vector_search call)
        mock_client.vector_search.assert_not_called()
        # Only the entity creation upsert should happen
        assert mock_client.upsert_node.call_count == 1

    @pytest.mark.asyncio
    async def test_add_entity_deduplicate_param_false(self, memory, mock_client):
        """Test adding entity with deduplicate=False parameter."""
        entity, dedup_result = await memory.add_entity(
            name="John Smith",
            entity_type="PERSON",
            deduplicate=False,
        )

        assert entity.name == "John Smith"
        assert dedup_result.is_duplicate is False
        assert dedup_result.action == "none"

    @pytest.mark.asyncio
    async def test_add_entity_skip_merged_entities(self, memory, mock_client):
        """Test that merged entities are skipped during deduplication."""
        merged_entity_id = str(uuid4())
        active_entity_id = str(uuid4())

        # vector_search returns both merged and active entities (flat dicts)
        mock_client.vector_search.return_value = [
            {
                "id": merged_entity_id,
                "name": "John Smith",
                "type": "PERSON",
                "merged_into": active_entity_id,  # This entity is merged
                "metadata": None,
                "_score": 0.98,
            },
            {
                "id": active_entity_id,
                "name": "John Q. Smith",
                "type": "PERSON",
                "merged_into": None,  # Active entity
                "metadata": None,
                "_score": 0.87,
            },
        ]

        entity, dedup_result = await memory.add_entity(
            name="Jon Smith",
            entity_type="PERSON",
        )

        # Should match the active entity, not the merged one
        assert dedup_result.matched_entity_id == UUID(active_entity_id)
        assert dedup_result.action == "flagged"  # 0.87 is between thresholds

    @pytest.mark.asyncio
    async def test_find_potential_duplicates(self, memory, mock_client):
        """Test finding potential duplicate pairs."""
        entity1_id = str(uuid4())
        entity2_id = str(uuid4())

        # query_nodes returns all entities
        mock_client.query_nodes.return_value = [
            {
                "id": entity1_id,
                "name": "John Smith",
                "type": "PERSON",
                "metadata": None,
            },
            {
                "id": entity2_id,
                "name": "Jon Smith",
                "type": "PERSON",
                "metadata": None,
            },
        ]

        # traverse returns SAME_AS neighbors with edge data
        # Called for each entity: entity1 sees entity2, entity2 sees entity1
        mock_client.traverse.side_effect = [
            # Traversal from entity1: finds entity2 with pending SAME_AS
            [
                {
                    "id": entity2_id,
                    "name": "Jon Smith",
                    "type": "PERSON",
                    "metadata": None,
                    "_edge": {"confidence": 0.88, "status": "pending", "match_type": "embedding"},
                }
            ],
            # Traversal from entity2: finds entity1 (same pair, will be deduped)
            [
                {
                    "id": entity1_id,
                    "name": "John Smith",
                    "type": "PERSON",
                    "metadata": None,
                    "_edge": {"confidence": 0.88, "status": "pending", "match_type": "embedding"},
                }
            ],
        ]

        duplicates = await memory.find_potential_duplicates(limit=10)

        assert len(duplicates) == 1
        entity1, entity2, confidence = duplicates[0]
        assert entity1.name == "John Smith"
        assert entity2.name == "Jon Smith"
        assert confidence == 0.88

    @pytest.mark.asyncio
    async def test_merge_duplicate_entities(self, memory, mock_client):
        """Test merging two duplicate entities."""
        source_id = uuid4()
        target_id = uuid4()

        # get_node is called multiple times:
        # 1) get source entity
        # 2) get target entity
        # 3) re-fetch source after merge
        # 4) re-fetch target after merge
        mock_client.get_node.side_effect = [
            # 1) source
            {
                "id": str(source_id),
                "name": "Jon Smith",
                "type": "PERSON",
                "metadata": None,
            },
            # 2) target
            {
                "id": str(target_id),
                "name": "John Smith",
                "type": "PERSON",
                "metadata": json.dumps({"aliases": []}),
            },
            # 3) re-fetch source (after merge, marked as merged)
            {
                "id": str(source_id),
                "name": "Jon Smith",
                "type": "PERSON",
                "merged_into": str(target_id),
                "metadata": None,
            },
            # 4) re-fetch target (after merge, with alias added)
            {
                "id": str(target_id),
                "name": "John Smith",
                "type": "PERSON",
                "metadata": json.dumps({"aliases": ["Jon Smith"]}),
            },
        ]

        # traverse for source relationships (returns empty for simplicity)
        mock_client.traverse.return_value = []

        result = await memory.merge_duplicate_entities(source_id, target_id)

        assert result is not None
        source, target = result
        assert source.name == "Jon Smith"
        assert target.name == "John Smith"

    @pytest.mark.asyncio
    async def test_review_duplicate_confirm(self, memory, mock_client):
        """Test confirming a duplicate pair."""
        source_id = uuid4()
        target_id = uuid4()

        # merge_duplicate_entities calls get_node multiple times
        mock_client.get_node.side_effect = [
            # 1) source
            {
                "id": str(source_id),
                "name": "Jon Smith",
                "type": "PERSON",
                "metadata": None,
            },
            # 2) target
            {
                "id": str(target_id),
                "name": "John Smith",
                "type": "PERSON",
                "metadata": None,
            },
            # 3) re-fetch source
            {
                "id": str(source_id),
                "name": "Jon Smith",
                "type": "PERSON",
                "merged_into": str(target_id),
                "metadata": None,
            },
            # 4) re-fetch target
            {
                "id": str(target_id),
                "name": "John Smith",
                "type": "PERSON",
                "metadata": None,
            },
        ]

        # traverse for source relationships
        mock_client.traverse.return_value = []

        result = await memory.review_duplicate(source_id, target_id, confirm=True)

        assert result is True
        # Should have called merge (update_node x2) and link_nodes for SAME_AS status
        assert mock_client.link_nodes.call_count >= 1

    @pytest.mark.asyncio
    async def test_review_duplicate_reject(self, memory, mock_client):
        """Test rejecting a duplicate pair."""
        source_id = uuid4()
        target_id = uuid4()

        result = await memory.review_duplicate(source_id, target_id, confirm=False)

        assert result is True
        # Should have updated the SAME_AS status to rejected via link_nodes
        mock_client.link_nodes.assert_called()

    @pytest.mark.asyncio
    async def test_get_same_as_cluster(self, memory, mock_client):
        """Test getting entities in a SAME_AS cluster."""
        entity_id = uuid4()
        related_id1 = str(uuid4())
        related_id2 = str(uuid4())

        # get_node is called by _get_entity_by_id for the original entity
        mock_client.get_node.return_value = {
            "id": str(entity_id),
            "name": "John Smith",
            "type": "PERSON",
            "metadata": None,
        }

        # traverse returns SAME_AS neighbors (flat dicts)
        mock_client.traverse.return_value = [
            {
                "id": related_id1,
                "name": "Jon Smith",
                "type": "PERSON",
                "metadata": None,
            },
            {
                "id": related_id2,
                "name": "J. Smith",
                "type": "PERSON",
                "metadata": None,
            },
        ]

        entities = await memory.get_same_as_cluster(entity_id)

        assert len(entities) == 3
        names = [e.name for e in entities]
        assert "John Smith" in names
        assert "Jon Smith" in names
        assert "J. Smith" in names

    @pytest.mark.asyncio
    async def test_get_deduplication_stats(self, memory, mock_client):
        """Test getting deduplication statistics."""
        entity1_id = str(uuid4())
        entity2_id = str(uuid4())
        entity3_id = str(uuid4())

        # count_nodes returns total
        mock_client.count_nodes.return_value = 100

        # query_nodes returns all entities (some merged)
        entities_list = []
        for i in range(85):
            entities_list.append({
                "id": str(uuid4()),
                "name": f"Entity {i}",
                "type": "PERSON",
                "merged_into": None,
                "metadata": None,
            })
        for i in range(15):
            entities_list.append({
                "id": str(uuid4()),
                "name": f"Merged Entity {i}",
                "type": "PERSON",
                "merged_into": str(uuid4()),
                "metadata": None,
            })
        mock_client.query_nodes.return_value = entities_list

        # traverse is called for each entity to find SAME_AS edges.
        # For simplicity, return empty for most, but set up some SAME_AS pairs.
        # We need 20 SAME_AS relationships total, 5 pending.
        # Since traverse is called for each of 100 entities, set up side effects.
        traverse_results = []
        same_as_pairs_made = 0
        pending_made = 0
        for i, ent in enumerate(entities_list):
            if same_as_pairs_made < 20 and i % 5 == 0 and i + 1 < len(entities_list):
                status = "pending" if pending_made < 5 else "confirmed"
                traverse_results.append([
                    {
                        "id": entities_list[i + 1]["id"],
                        "name": entities_list[i + 1]["name"],
                        "type": "PERSON",
                        "metadata": None,
                        "_edge": {"status": status, "confidence": 0.9},
                    }
                ])
                same_as_pairs_made += 1
                if status == "pending":
                    pending_made += 1
            else:
                traverse_results.append([])
        mock_client.traverse.side_effect = traverse_results

        stats = await memory.get_deduplication_stats()

        assert stats.total_entities == 100
        assert stats.merged_entities == 15
        assert stats.same_as_relationships == 20
        assert stats.pending_reviews == 5

    @pytest.mark.asyncio
    async def test_get_deduplication_stats_empty(self, memory, mock_client):
        """Test getting stats when no data exists."""
        mock_client.count_nodes.return_value = 0
        mock_client.query_nodes.return_value = []

        stats = await memory.get_deduplication_stats()

        assert stats.total_entities == 0
        assert stats.merged_entities == 0


class TestDeduplicationWithFuzzyMatching:
    """Tests for deduplication with fuzzy string matching."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphBackend client."""
        return _make_graph_backend_mock()

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 384)
        return embedder

    @pytest.fixture
    def memory_with_fuzzy(self, mock_client, mock_embedder):
        """Create memory with fuzzy matching enabled."""
        return LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(
                use_fuzzy_matching=True,
                fuzzy_threshold=0.85,
            ),
        )

    @pytest.fixture
    def memory_without_fuzzy(self, mock_client, mock_embedder):
        """Create memory with fuzzy matching disabled."""
        return LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(
                use_fuzzy_matching=False,
            ),
        )

    @pytest.mark.asyncio
    async def test_fuzzy_matching_boosts_score(self, memory_with_fuzzy, mock_client):
        """Test that fuzzy matching can boost the combined score."""
        existing_entity_id = str(uuid4())

        # Entity with medium embedding similarity but high fuzzy similarity
        mock_client.vector_search.return_value = [
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "canonical_name": "John Smith",
                "type": "PERSON",
                "metadata": None,
                "merged_into": None,
                "_score": 0.88,
            }
        ]

        # "Jon Smith" has high fuzzy similarity to "John Smith"
        entity, dedup_result = await memory_with_fuzzy.add_entity(
            name="Jon Smith",  # Very similar string
            entity_type="PERSON",
        )

        # With fuzzy matching, the combined score should be boosted
        assert dedup_result.is_duplicate is True
        # The match type should indicate both methods matched
        assert dedup_result.match_type in ["embedding", "both"]

    @pytest.mark.asyncio
    async def test_no_fuzzy_matching_when_disabled(self, memory_without_fuzzy, mock_client):
        """Test that fuzzy matching is not used when disabled."""
        existing_entity_id = str(uuid4())

        mock_client.vector_search.return_value = [
            {
                "id": existing_entity_id,
                "name": "John Smith",
                "type": "PERSON",
                "metadata": None,
                "merged_into": None,
                "_score": 0.88,
            }
        ]

        entity, dedup_result = await memory_without_fuzzy.add_entity(
            name="Jon Smith",
            entity_type="PERSON",
        )

        # Should only use embedding matching
        assert dedup_result.match_type == "embedding"


class TestDeduplicationConfigurationOptions:
    """Tests for various deduplication configuration options."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphBackend client."""
        return _make_graph_backend_mock()

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 384)
        return embedder

    @pytest.mark.asyncio
    async def test_match_same_type_only_true(self, mock_client, mock_embedder):
        """Test matching only same entity types."""
        memory = LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(match_same_type_only=True),
        )

        await memory.add_entity(name="Apple", entity_type="ORGANIZATION")

        # Check that vector_search was called with filters including type
        mock_client.vector_search.assert_called_once()
        call_kwargs = mock_client.vector_search.call_args
        # vector_search(label, property, embedding, limit=..., threshold=..., filters=...)
        filters = call_kwargs.kwargs.get("filters") or (call_kwargs[1].get("filters") if len(call_kwargs) > 1 else None)
        assert filters is not None
        assert filters.get("type") == "ORGANIZATION"

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_client, mock_embedder):
        """Test custom threshold values."""
        memory = LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(
                auto_merge_threshold=0.99,
                flag_threshold=0.90,
            ),
        )

        existing_entity_id = str(uuid4())
        mock_client.vector_search.return_value = [
            {
                "id": existing_entity_id,
                "name": "Test Entity",
                "type": "OBJECT",
                "metadata": None,
                "merged_into": None,
                "_score": 0.95,  # Above default but below custom auto_merge
            }
        ]

        entity, dedup_result = await memory.add_entity(
            name="Test Entity 2",
            entity_type="OBJECT",
        )

        # With custom thresholds, 0.95 should only flag, not merge
        assert dedup_result.action == "flagged"

    @pytest.mark.asyncio
    async def test_max_candidates_limit(self, mock_client, mock_embedder):
        """Test that max_candidates limits the search."""
        memory = LongTermMemory(
            client=mock_client,
            embedder=mock_embedder,
            deduplication=DeduplicationConfig(max_candidates=5),
        )

        await memory.add_entity(name="Test", entity_type="OBJECT")

        # Check the limit parameter passed to vector_search
        mock_client.vector_search.assert_called_once()
        call_kwargs = mock_client.vector_search.call_args
        limit = call_kwargs.kwargs.get("limit") or (call_kwargs[1].get("limit") if len(call_kwargs) > 1 else None)
        assert limit == 5
