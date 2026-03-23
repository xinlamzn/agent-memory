"""Unit tests for provenance tracking."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from neo4j_agent_memory.memory.long_term import Entity, LongTermMemory


@pytest.fixture
def mock_client():
    """Create a mock GraphBackend client."""
    client = MagicMock()
    client.upsert_node = AsyncMock(return_value={})
    client.get_node = AsyncMock(return_value=None)
    client.link_nodes = AsyncMock(return_value={})
    client.traverse = AsyncMock(return_value=[])
    client.query_nodes = AsyncMock(return_value=[])
    client.count_nodes = AsyncMock(return_value=0)
    client.unlink_nodes = AsyncMock(return_value=True)
    client.vector_search = AsyncMock(return_value=[])
    client.update_node = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def long_term_memory(mock_client, mock_embedder):
    """Create a LongTermMemory instance with mocks."""
    return LongTermMemory(
        client=mock_client,
        embedder=mock_embedder,
    )


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id=uuid4(),
        name="John Smith",
        type="PERSON",
        subtype="INDIVIDUAL",
        description="A test person",
        confidence=0.95,
    )


class TestRegisterExtractor:
    """Tests for register_extractor method."""

    @pytest.mark.asyncio
    async def test_register_new_extractor(self, long_term_memory, mock_client):
        """Test registering a new extractor."""
        extractor_id = str(uuid4())
        mock_client.upsert_node.return_value = {
            "id": extractor_id,
            "name": "GLiNEREntityExtractor",
            "version": "1.0.0",
        }

        result = await long_term_memory.register_extractor(
            "GLiNEREntityExtractor",
            version="1.0.0",
            config={"threshold": 0.5},
        )

        assert result["name"] == "GLiNEREntityExtractor"
        assert result["version"] == "1.0.0"
        mock_client.upsert_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_extractor_minimal(self, long_term_memory, mock_client):
        """Test registering extractor with minimal info."""
        mock_client.upsert_node.return_value = {
            "id": str(uuid4()),
            "name": "SpacyNER",
            "version": None,
        }

        result = await long_term_memory.register_extractor("SpacyNER")

        assert result["name"] == "SpacyNER"
        assert result["version"] is None


class TestLinkEntityToMessage:
    """Tests for link_entity_to_message method."""

    @pytest.mark.asyncio
    async def test_link_entity_to_message(self, long_term_memory, mock_client, sample_entity):
        """Test linking entity to source message."""
        mock_client.link_nodes.return_value = {"confidence": 0.9}
        message_id = uuid4()

        result = await long_term_memory.link_entity_to_message(
            sample_entity,
            message_id,
            confidence=0.9,
            start_pos=10,
            end_pos=20,
            context="... John Smith works at ...",
        )

        assert result is True
        mock_client.link_nodes.assert_called_once()
        # Verify the arguments passed to link_nodes
        call_kwargs = mock_client.link_nodes.call_args
        assert call_kwargs[0][0] == "Entity"  # source_label
        assert call_kwargs[0][1] == str(sample_entity.id)  # source_id
        assert call_kwargs[0][2] == "Message"  # target_label
        assert call_kwargs[0][3] == str(message_id)  # target_id
        assert call_kwargs[0][4] == "EXTRACTED_FROM"  # relationship_type
        props = call_kwargs[1]["properties"]
        assert props["confidence"] == 0.9
        assert props["start_pos"] == 10
        assert props["end_pos"] == 20

    @pytest.mark.asyncio
    async def test_link_entity_by_uuid(self, long_term_memory, mock_client):
        """Test linking entity using UUID directly."""
        mock_client.link_nodes.return_value = {}
        entity_id = uuid4()
        message_id = uuid4()

        result = await long_term_memory.link_entity_to_message(
            entity_id,
            message_id,
        )

        assert result is True
        call_kwargs = mock_client.link_nodes.call_args
        assert call_kwargs[0][1] == str(entity_id)


class TestLinkEntityToExtractor:
    """Tests for link_entity_to_extractor method."""

    @pytest.mark.asyncio
    async def test_link_entity_to_extractor(self, long_term_memory, mock_client, sample_entity):
        """Test linking entity to extractor."""
        extractor_id = str(uuid4())
        # upsert_node is called by register_extractor
        mock_client.upsert_node.return_value = {
            "id": extractor_id,
            "name": "GLiNER",
            "version": None,
        }
        # get_node is called to find the extractor by name
        mock_client.get_node.return_value = {"id": extractor_id, "name": "GLiNER"}
        # link_nodes creates the EXTRACTED_BY link
        mock_client.link_nodes.return_value = {}

        result = await long_term_memory.link_entity_to_extractor(
            sample_entity,
            "GLiNER",
            confidence=0.85,
            extraction_time_ms=150.5,
        )

        assert result is True
        # upsert_node called once (register_extractor)
        mock_client.upsert_node.assert_called_once()
        # link_nodes called once (EXTRACTED_BY)
        mock_client.link_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_entity_by_uuid_to_extractor(self, long_term_memory, mock_client):
        """Test linking entity using UUID."""
        extractor_id = str(uuid4())
        mock_client.upsert_node.return_value = {
            "id": extractor_id,
            "name": "SpacyNER",
            "version": None,
        }
        mock_client.get_node.return_value = {"id": extractor_id, "name": "SpacyNER"}
        mock_client.link_nodes.return_value = {}
        entity_id = uuid4()

        result = await long_term_memory.link_entity_to_extractor(
            entity_id,
            "SpacyNER",
        )

        assert result is True


class TestGetEntityProvenance:
    """Tests for get_entity_provenance method."""

    @pytest.mark.asyncio
    async def test_get_provenance_with_sources(self, long_term_memory, mock_client, sample_entity):
        """Test getting provenance with message sources."""
        msg_id = str(uuid4())
        # traverse is called twice: once for EXTRACTED_FROM, once for EXTRACTED_BY
        mock_client.traverse.side_effect = [
            # First call: EXTRACTED_FROM → Messages
            [
                {
                    "id": msg_id,
                    "content": "John works at Acme",
                    "_edge": {"confidence": 0.9, "start_pos": 0, "end_pos": 4},
                }
            ],
            # Second call: EXTRACTED_BY → Extractors
            [],
        ]

        result = await long_term_memory.get_entity_provenance(sample_entity)

        assert len(result["sources"]) == 1
        assert result["sources"][0]["confidence"] == 0.9
        assert result["sources"][0]["start_pos"] == 0
        assert len(result["extractors"]) == 0

    @pytest.mark.asyncio
    async def test_get_provenance_with_extractors(
        self, long_term_memory, mock_client, sample_entity
    ):
        """Test getting provenance with extractor info."""
        mock_client.traverse.side_effect = [
            # First call: EXTRACTED_FROM → Messages
            [],
            # Second call: EXTRACTED_BY → Extractors
            [
                {
                    "id": str(uuid4()),
                    "name": "GLiNER",
                    "version": "1.0",
                    "_edge": {"confidence": 0.85, "extraction_time_ms": 100},
                }
            ],
        ]

        result = await long_term_memory.get_entity_provenance(sample_entity)

        assert len(result["extractors"]) == 1
        assert result["extractors"][0]["name"] == "GLiNER"
        assert result["extractors"][0]["version"] == "1.0"
        assert result["extractors"][0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_provenance_no_results(self, long_term_memory, mock_client, sample_entity):
        """Test getting provenance when none exists."""
        mock_client.traverse.side_effect = [
            [],  # EXTRACTED_FROM
            [],  # EXTRACTED_BY
        ]

        result = await long_term_memory.get_entity_provenance(sample_entity)

        assert result == {"sources": [], "extractors": []}

    @pytest.mark.asyncio
    async def test_get_provenance_by_uuid(self, long_term_memory, mock_client):
        """Test getting provenance using UUID directly."""
        entity_id = uuid4()
        mock_client.traverse.side_effect = [
            [],  # EXTRACTED_FROM
            [],  # EXTRACTED_BY
        ]

        result = await long_term_memory.get_entity_provenance(entity_id)

        assert result == {"sources": [], "extractors": []}


class TestGetEntitiesFromMessage:
    """Tests for get_entities_from_message method."""

    @pytest.mark.asyncio
    async def test_get_entities_from_message(self, long_term_memory, mock_client):
        """Test getting entities extracted from a message."""
        message_id = uuid4()
        entity1_id = str(uuid4())
        entity2_id = str(uuid4())
        mock_client.traverse.return_value = [
            {
                "id": entity1_id,
                "name": "John",
                "type": "PERSON",
                "confidence": 0.9,
                "_edge": {"confidence": 0.9, "start_pos": 0, "end_pos": 4},
            },
            {
                "id": entity2_id,
                "name": "Acme",
                "type": "ORGANIZATION",
                "confidence": 0.85,
                "_edge": {"confidence": 0.85, "start_pos": 15, "end_pos": 19},
            },
        ]

        result = await long_term_memory.get_entities_from_message(message_id)

        assert len(result) == 2
        assert result[0][0].name == "John"
        assert result[0][1]["start_pos"] == 0
        assert result[1][0].name == "Acme"
        assert result[1][1]["start_pos"] == 15

    @pytest.mark.asyncio
    async def test_get_entities_from_message_empty(self, long_term_memory, mock_client):
        """Test getting entities when none exist."""
        mock_client.traverse.return_value = []

        result = await long_term_memory.get_entities_from_message(uuid4())

        assert result == []


class TestGetEntitiesByExtractor:
    """Tests for get_entities_by_extractor method."""

    @pytest.mark.asyncio
    async def test_get_entities_by_extractor(self, long_term_memory, mock_client):
        """Test getting entities by extractor name."""
        extractor_id = str(uuid4())
        mock_client.get_node.return_value = {"id": extractor_id, "name": "GLiNER"}
        mock_client.traverse.return_value = [
            {
                "id": str(uuid4()),
                "name": "John",
                "type": "PERSON",
                "confidence": 0.9,
                "_edge": {"confidence": 0.9, "extraction_time_ms": 50.0},
            },
        ]

        result = await long_term_memory.get_entities_by_extractor("GLiNER", limit=50)

        assert len(result) == 1
        assert result[0][0].name == "John"
        assert result[0][1]["extraction_time_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_get_entities_by_extractor_with_limit(self, long_term_memory, mock_client):
        """Test limit parameter is passed correctly."""
        extractor_id = str(uuid4())
        mock_client.get_node.return_value = {"id": extractor_id, "name": "SpacyNER"}
        mock_client.traverse.return_value = []

        await long_term_memory.get_entities_by_extractor("SpacyNER", limit=25)

        # Verify traverse was called with the correct limit
        mock_client.traverse.assert_called_once()
        call_kwargs = mock_client.traverse.call_args[1]
        assert call_kwargs["limit"] == 25


class TestListExtractors:
    """Tests for list_extractors method."""

    @pytest.mark.asyncio
    async def test_list_extractors(self, long_term_memory, mock_client):
        """Test listing all extractors."""
        ex1_id = str(uuid4())
        ex2_id = str(uuid4())
        mock_client.query_nodes.return_value = [
            {"id": ex1_id, "name": "GLiNER", "version": "1.0"},
            {"id": ex2_id, "name": "SpacyNER", "version": "3.7"},
        ]
        # traverse is called for each extractor to count entities
        mock_client.traverse.side_effect = [
            [{"id": str(uuid4())} for _ in range(100)],  # 100 entities for GLiNER
            [{"id": str(uuid4())} for _ in range(50)],   # 50 entities for SpacyNER
        ]

        result = await long_term_memory.list_extractors()

        assert len(result) == 2
        assert result[0]["name"] == "GLiNER"
        assert result[0]["entity_count"] == 100
        assert result[1]["name"] == "SpacyNER"

    @pytest.mark.asyncio
    async def test_list_extractors_empty(self, long_term_memory, mock_client):
        """Test listing when no extractors exist."""
        mock_client.query_nodes.return_value = []

        result = await long_term_memory.list_extractors()

        assert result == []


class TestGetExtractionStats:
    """Tests for get_extraction_stats method."""

    @pytest.mark.asyncio
    async def test_get_extraction_stats(self, long_term_memory, mock_client):
        """Test getting overall extraction stats."""
        mock_client.count_nodes.return_value = 500

        mock_client.query_nodes.side_effect = [
            # First call: query_nodes("Extractor")
            [{"name": "GLiNER"}, {"name": "SpacyNER"}],
            # Second call: query_nodes("Entity", limit=10000)
            [
                {"id": str(uuid4()), "name": "E1", "type": "PERSON"},
                {"id": str(uuid4()), "name": "E2", "type": "PERSON"},
            ],
        ]

        # For each entity, traverse to find source messages
        msg_id_1 = str(uuid4())
        msg_id_2 = str(uuid4())
        mock_client.traverse.side_effect = [
            [{"id": msg_id_1}],  # entity 1 sources
            [{"id": msg_id_2}],  # entity 2 sources
        ]

        result = await long_term_memory.get_extraction_stats()

        assert result["total_entities"] == 500
        assert result["source_messages"] == 2
        assert "GLiNER" in result["extractors"]

    @pytest.mark.asyncio
    async def test_get_extraction_stats_empty(self, long_term_memory, mock_client):
        """Test stats when no data exists."""
        mock_client.count_nodes.return_value = 0
        mock_client.query_nodes.side_effect = [
            [],  # Extractor query
            [],  # Entity query
        ]

        result = await long_term_memory.get_extraction_stats()

        assert result["total_entities"] == 0
        assert result["source_messages"] == 0
        assert result["extractors"] == []


class TestGetExtractorStats:
    """Tests for get_extractor_stats method."""

    @pytest.mark.asyncio
    async def test_get_extractor_stats(self, long_term_memory, mock_client):
        """Test getting per-extractor stats."""
        ex1_id = str(uuid4())
        ex2_id = str(uuid4())
        mock_client.query_nodes.return_value = [
            {"id": ex1_id, "name": "GLiNER", "version": "1.0"},
            {"id": ex2_id, "name": "SpacyNER", "version": "3.7"},
        ]
        # traverse for each extractor to get entities with edge data
        mock_client.traverse.side_effect = [
            # GLiNER: 100 entities with avg confidence 0.85
            [
                {"id": str(uuid4()), "name": f"E{i}", "type": "PERSON", "_edge": {"confidence": 0.85}}
                for i in range(100)
            ],
            # SpacyNER: 50 entities with avg confidence 0.75
            [
                {"id": str(uuid4()), "name": f"E{i}", "type": "PERSON", "_edge": {"confidence": 0.75}}
                for i in range(50)
            ],
        ]

        result = await long_term_memory.get_extractor_stats()

        assert len(result) == 2
        assert result[0]["name"] == "GLiNER"
        assert result[0]["entity_count"] == 100
        assert abs(result[0]["avg_confidence"] - 0.85) < 1e-10


class TestDeleteEntityProvenance:
    """Tests for delete_entity_provenance method."""

    @pytest.mark.asyncio
    async def test_delete_provenance(self, long_term_memory, mock_client, sample_entity):
        """Test deleting provenance for an entity."""
        msg_id_1 = str(uuid4())
        msg_id_2 = str(uuid4())
        ext_id = str(uuid4())
        # traverse EXTRACTED_FROM, then traverse EXTRACTED_BY
        mock_client.traverse.side_effect = [
            [{"id": msg_id_1}, {"id": msg_id_2}],  # 2 source messages
            [{"id": ext_id}],                        # 1 extractor
        ]
        mock_client.unlink_nodes.return_value = True

        result = await long_term_memory.delete_entity_provenance(sample_entity)

        assert result == 3

    @pytest.mark.asyncio
    async def test_delete_provenance_none(self, long_term_memory, mock_client, sample_entity):
        """Test deleting when no provenance exists."""
        mock_client.traverse.side_effect = [
            [],  # no source messages
            [],  # no extractors
        ]

        result = await long_term_memory.delete_entity_provenance(sample_entity)

        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_provenance_by_uuid(self, long_term_memory, mock_client):
        """Test deleting provenance using UUID."""
        entity_id = uuid4()
        msg_id = str(uuid4())
        ext_id = str(uuid4())
        mock_client.traverse.side_effect = [
            [{"id": msg_id}],   # 1 source message
            [{"id": ext_id}],   # 1 extractor
        ]
        mock_client.unlink_nodes.return_value = True

        result = await long_term_memory.delete_entity_provenance(entity_id)

        assert result == 2
        # Verify unlink_nodes was called with correct entity_id
        calls = mock_client.unlink_nodes.call_args_list
        assert calls[0][0][1] == str(entity_id)
        assert calls[1][0][1] == str(entity_id)


class TestProvenanceEdgeCases:
    """Edge case tests for provenance tracking."""

    @pytest.mark.asyncio
    async def test_provenance_with_dict_relationship(
        self, long_term_memory, mock_client, sample_entity
    ):
        """Test provenance parsing when relationship is a dict."""
        msg_id = str(uuid4())
        mock_client.traverse.side_effect = [
            # EXTRACTED_FROM with edge data
            [
                {
                    "id": msg_id,
                    "content": "Test",
                    "_edge": {"confidence": 0.8},
                }
            ],
            # EXTRACTED_BY
            [],
        ]

        result = await long_term_memory.get_entity_provenance(sample_entity)

        assert len(result["sources"]) == 1
        assert result["sources"][0]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_empty_result_handling(self, long_term_memory, mock_client):
        """Test handling of empty results."""
        mock_client.traverse.side_effect = [
            [],  # no source messages
            [],  # no extractors
        ]

        result = await long_term_memory.delete_entity_provenance(uuid4())

        assert result == 0

    @pytest.mark.asyncio
    async def test_link_with_string_message_id(self, long_term_memory, mock_client, sample_entity):
        """Test linking with string message ID."""
        mock_client.link_nodes.return_value = {}
        message_id = str(uuid4())

        result = await long_term_memory.link_entity_to_message(
            sample_entity,
            message_id,  # String instead of UUID
        )

        assert result is True
