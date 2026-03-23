"""Unit tests for relation storage in short-term memory."""

from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import pytest

from neo4j_agent_memory.extraction.base import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from neo4j_agent_memory.memory.short_term import Message, MessageRole, ShortTermMemory


class MockExtractorWithRelations:
    """Mock extractor that returns both entities and relations."""

    def __init__(self, entities: list[ExtractedEntity], relations: list[ExtractedRelation]):
        self._entities = entities
        self._relations = relations

    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> ExtractionResult:
        return ExtractionResult(
            entities=self._entities,
            relations=self._relations if extract_relations else [],
            source_text=text,
        )


def _make_mock_client():
    """Create a mock GraphBackend client with all async methods."""
    client = MagicMock()
    client.upsert_node = AsyncMock(return_value={})
    client.get_node = AsyncMock(return_value=None)
    client.link_nodes = AsyncMock(return_value={})
    client.traverse = AsyncMock(return_value=[])
    client.query_nodes = AsyncMock(return_value=[])
    client.count_nodes = AsyncMock(return_value=0)
    client.create_node_with_links = AsyncMock(return_value={})
    client.vector_search = AsyncMock(return_value=[])
    client.update_node = AsyncMock(return_value={})
    client.unlink_nodes = AsyncMock(return_value=True)
    client.delete_node = AsyncMock(return_value=True)
    return client


class TestRelationStorage:
    """Tests for storing extracted relations."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphBackend client."""
        return _make_mock_client()

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            ExtractedEntity(
                name="Brian Chesky",
                type="PERSON",
                confidence=0.95,
            ),
            ExtractedEntity(
                name="Airbnb",
                type="ORGANIZATION",
                confidence=0.92,
            ),
        ]

    @pytest.fixture
    def sample_relations(self):
        """Sample relations for testing."""
        return [
            ExtractedRelation(
                source="Brian Chesky",
                target="Airbnb",
                relation_type="FOUNDED",
                confidence=0.9,
            ),
        ]

    @pytest.mark.asyncio
    async def test_extract_and_link_entities_stores_relations(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test that _extract_and_link_entities stores relations."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        message = Message(
            id=uuid4(),
            role=MessageRole.USER,
            content="Brian Chesky founded Airbnb",
        )

        await memory._extract_and_link_entities(message, extract_relations=True)

        # Should have 2 upsert_node calls (one per entity)
        assert mock_client.upsert_node.call_count == 2

        # Should have 2 MENTIONS link_nodes + 1 RELATED_TO link_nodes = 3
        assert mock_client.link_nodes.call_count == 3

        # Find the RELATED_TO call
        relation_calls = [
            c for c in mock_client.link_nodes.call_args_list
            if c[0][4] == "RELATED_TO"  # 5th positional arg is relationship_type
        ]
        assert len(relation_calls) == 1

        # Verify the RELATED_TO call has correct properties
        rel_call = relation_calls[0]
        assert rel_call[0][0] == "Entity"  # from_label
        assert rel_call[0][2] == "Entity"  # to_label
        assert rel_call[1]["properties"]["relation_type"] == "FOUNDED"
        assert rel_call[1]["properties"]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_extract_and_link_entities_skips_relations_when_disabled(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test that relations are not stored when extract_relations=False."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        message = Message(
            id=uuid4(),
            role=MessageRole.USER,
            content="Brian Chesky founded Airbnb",
        )

        await memory._extract_and_link_entities(message, extract_relations=False)

        # Should have 2 upsert_node calls (one per entity)
        assert mock_client.upsert_node.call_count == 2

        # Should have only 2 MENTIONS link_nodes, NO RELATED_TO
        assert mock_client.link_nodes.call_count == 2

        # Verify all link_nodes calls are MENTIONS (none are RELATED_TO)
        for c in mock_client.link_nodes.call_args_list:
            assert c[0][4] == "MENTIONS"

    @pytest.mark.asyncio
    async def test_store_relations_uses_id_based_query_for_local_entities(
        self, mock_client, sample_relations
    ):
        """Test that _store_relations uses ID-based query when both entities are local."""
        memory = ShortTermMemory(mock_client)

        entity_name_to_id = {
            "brian chesky": "entity-1",
            "airbnb": "entity-2",
        }

        stored = await memory._store_relations(sample_relations, entity_name_to_id)

        assert stored == 1

        # Check that link_nodes was called with the right args
        mock_client.link_nodes.assert_called_once()
        c = mock_client.link_nodes.call_args

        # Positional args: from_label, from_id, to_label, to_id, relationship_type
        assert c[0][0] == "Entity"
        assert c[0][1] == "entity-1"  # source_id
        assert c[0][2] == "Entity"
        assert c[0][3] == "entity-2"  # target_id
        assert c[0][4] == "RELATED_TO"

        # Keyword arg: properties
        assert c[1]["properties"]["relation_type"] == "FOUNDED"
        assert c[1]["properties"]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_store_relations_uses_name_based_query_for_cross_message_entities(
        self, mock_client, sample_relations
    ):
        """Test that _store_relations uses name-based query for cross-message relations."""
        memory = ShortTermMemory(mock_client)

        # Only one entity is in local mapping
        entity_name_to_id = {
            "brian chesky": "entity-1",
            # "airbnb" is not in local mapping - simulating cross-message relation
        }

        # Mock get_node to return entity for the name-based lookup of "Airbnb"
        async def get_node_side_effect(label, *, id=None, filters=None):
            if label == "Entity" and filters and filters.get("name") == "Airbnb":
                return {"id": "entity-remote-2", "name": "Airbnb"}
            return None

        mock_client.get_node = AsyncMock(side_effect=get_node_side_effect)

        stored = await memory._store_relations(sample_relations, entity_name_to_id)

        assert stored == 1

        # get_node should have been called for the missing entity "Airbnb"
        mock_client.get_node.assert_called()
        # Find the call for "Airbnb"
        name_lookup_calls = [
            c for c in mock_client.get_node.call_args_list
            if c[1].get("filters", {}).get("name") == "Airbnb"
        ]
        assert len(name_lookup_calls) >= 1

        # link_nodes should have been called with the resolved IDs
        mock_client.link_nodes.assert_called_once()
        c = mock_client.link_nodes.call_args
        assert c[0][0] == "Entity"
        assert c[0][1] == "entity-1"  # source from local mapping
        assert c[0][2] == "Entity"
        assert c[0][3] == "entity-remote-2"  # target from get_node lookup
        assert c[0][4] == "RELATED_TO"

    @pytest.mark.asyncio
    async def test_store_relations_returns_zero_for_empty_relations(self, mock_client):
        """Test that _store_relations returns 0 for empty relations list."""
        memory = ShortTermMemory(mock_client)

        stored = await memory._store_relations([], {})

        assert stored == 0
        mock_client.link_nodes.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_message_with_extract_relations_true(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test add_message with extract_relations=True."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        # Mock conversation exists via get_node
        conv_id = str(uuid4())
        mock_client.get_node = AsyncMock(
            return_value={"id": conv_id, "session_id": "test"}
        )

        await memory.add_message(
            "test-session",
            MessageRole.USER,
            "Brian Chesky founded Airbnb",
            extract_entities=True,
            extract_relations=True,
        )

        # Verify that link_nodes was called with RELATED_TO for relation storage
        relation_calls = [
            c for c in mock_client.link_nodes.call_args_list
            if c[0][4] == "RELATED_TO"
        ]
        assert len(relation_calls) >= 1

    @pytest.mark.asyncio
    async def test_add_message_with_extract_relations_false(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test add_message with extract_relations=False."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        # Mock conversation exists via get_node
        conv_id = str(uuid4())
        mock_client.get_node = AsyncMock(
            return_value={"id": conv_id, "session_id": "test"}
        )

        await memory.add_message(
            "test-session",
            MessageRole.USER,
            "Brian Chesky founded Airbnb",
            extract_entities=True,
            extract_relations=False,
        )

        # Should NOT have any RELATED_TO link_nodes calls
        relation_calls = [
            c for c in mock_client.link_nodes.call_args_list
            if c[0][4] == "RELATED_TO"
        ]
        assert len(relation_calls) == 0


class TestExtractEntitiesFromSessionWithRelations:
    """Tests for extract_entities_from_session with relation support."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphBackend client."""
        return _make_mock_client()

    @pytest.fixture
    def sample_entities(self):
        return [
            ExtractedEntity(name="Brian Chesky", type="PERSON", confidence=0.95),
            ExtractedEntity(name="Airbnb", type="ORGANIZATION", confidence=0.92),
        ]

    @pytest.fixture
    def sample_relations(self):
        return [
            ExtractedRelation(
                source="Brian Chesky",
                target="Airbnb",
                relation_type="FOUNDED",
                confidence=0.9,
            ),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_from_session_returns_relation_count(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test that extract_entities_from_session returns relations_extracted count."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        conv_id = str(uuid4())

        # Mock get_node to return conversation
        mock_client.get_node = AsyncMock(
            return_value={"id": conv_id, "session_id": "test-session"}
        )

        # Mock traverse: first call returns messages, subsequent calls return []
        # (for skip_existing checks)
        msg_list = [
            {"id": "msg-1", "content": "Brian Chesky founded Airbnb"},
        ]

        async def traverse_side_effect(label, node_id, *, relationship_types=None, target_labels=None, direction=None, limit=None):
            if label == "Conversation" and relationship_types == ["HAS_MESSAGE"]:
                return msg_list
            # For MENTIONS check (skip_existing) and any other traverse, return empty
            return []

        mock_client.traverse = AsyncMock(side_effect=traverse_side_effect)

        result = await memory.extract_entities_from_session(
            "test-session",
            extract_relations=True,
        )

        assert "relations_extracted" in result
        assert result["relations_extracted"] >= 1
        assert result["messages_processed"] == 1
        assert result["entities_extracted"] == 2

    @pytest.mark.asyncio
    async def test_extract_entities_from_session_without_relations(
        self, mock_client, sample_entities, sample_relations
    ):
        """Test extract_entities_from_session with extract_relations=False."""
        extractor = MockExtractorWithRelations(sample_entities, sample_relations)
        memory = ShortTermMemory(mock_client, extractor=extractor)

        conv_id = str(uuid4())

        # Mock get_node to return conversation
        mock_client.get_node = AsyncMock(
            return_value={"id": conv_id, "session_id": "test-session"}
        )

        # Mock traverse
        msg_list = [
            {"id": "msg-1", "content": "Brian Chesky founded Airbnb"},
        ]

        async def traverse_side_effect(label, node_id, *, relationship_types=None, target_labels=None, direction=None, limit=None):
            if label == "Conversation" and relationship_types == ["HAS_MESSAGE"]:
                return msg_list
            return []

        mock_client.traverse = AsyncMock(side_effect=traverse_side_effect)

        result = await memory.extract_entities_from_session(
            "test-session",
            extract_relations=False,
        )

        assert result["relations_extracted"] == 0
        assert result["messages_processed"] == 1
        assert result["entities_extracted"] == 2

    @pytest.mark.asyncio
    async def test_extract_entities_from_session_no_extractor(self, mock_client):
        """Test extract_entities_from_session returns zeros when no extractor."""
        memory = ShortTermMemory(mock_client)  # No extractor

        result = await memory.extract_entities_from_session("test-session")

        assert result == {
            "messages_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
        }


class TestCypherQueries:
    """Tests for the new Cypher queries."""

    def test_create_entity_relation_by_id_query_exists(self):
        """Test that CREATE_ENTITY_RELATION_BY_ID query is defined."""
        from neo4j_agent_memory.graph import queries

        assert hasattr(queries, "CREATE_ENTITY_RELATION_BY_ID")
        query = queries.CREATE_ENTITY_RELATION_BY_ID

        # Check query contains expected elements
        assert "MATCH (source:Entity {id: $source_id})" in query
        assert "MATCH (target:Entity {id: $target_id})" in query
        assert "MERGE (source)-[r:RELATED_TO]->(target)" in query
        assert "relation_type" in query
        assert "confidence" in query

    def test_create_entity_relation_by_name_query_exists(self):
        """Test that CREATE_ENTITY_RELATION_BY_NAME query is defined."""
        from neo4j_agent_memory.graph import queries

        assert hasattr(queries, "CREATE_ENTITY_RELATION_BY_NAME")
        query = queries.CREATE_ENTITY_RELATION_BY_NAME

        # Check query contains expected elements
        assert "toLower(source.name)" in query or "source.name" in query
        assert "toLower(target.name)" in query or "target.name" in query
        assert "MERGE (source)-[r:RELATED_TO]->(target)" in query
        assert "relation_type" in query
        assert "confidence" in query
