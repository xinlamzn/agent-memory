"""Reasoning memory for reasoning traces and tool usage."""

import json
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from neo4j_agent_memory.core.memory import BaseMemory, MemoryEntry
from neo4j_agent_memory.graph.result_adapter import (
    deserialize_metadata,
    serialize_metadata,
    to_python_datetime,
)


def _serialize_json(data: dict[str, Any] | list | None) -> str | None:
    """Serialize dict/list to JSON string for storage.

    ``serialize_metadata`` only handles dicts; this helper also covers
    plain lists and the empty-collection → ``None`` mapping needed by
    several call-sites.
    """
    if data is None or data == {} or data == []:
        return None
    if isinstance(data, dict):
        return serialize_metadata(data)
    return json.dumps(data)


def _deserialize_json(data_str: str | None) -> dict[str, Any] | list | None:
    """Deserialize JSON string (dicts and lists)."""
    if data_str is None:
        return None
    try:
        return json.loads(data_str)
    except (json.JSONDecodeError, TypeError):
        return None


if TYPE_CHECKING:
    from neo4j_agent_memory.embeddings.base import Embedder
    from neo4j_agent_memory.graph.backend_protocol import GraphBackend


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolCall(MemoryEntry):
    """A tool call made during reasoning."""

    tool_name: str = Field(description="Name of the tool")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Any | None = Field(default=None, description="Tool result")
    status: ToolCallStatus = Field(default=ToolCallStatus.PENDING, description="Call status")
    duration_ms: int | None = Field(default=None, description="Duration in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")
    step_id: UUID | None = Field(default=None, description="Parent reasoning step ID")


class ReasoningStep(MemoryEntry):
    """A step in the agent's reasoning process."""

    trace_id: UUID = Field(description="Parent trace ID")
    step_number: int = Field(description="Step number in sequence")
    thought: str | None = Field(default=None, description="Agent's thought/reasoning")
    action: str | None = Field(default=None, description="Action taken")
    observation: str | None = Field(default=None, description="Observation from action")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls in this step")


class ReasoningTrace(MemoryEntry):
    """A complete reasoning trace for a task."""

    session_id: str = Field(description="Session identifier")
    task: str = Field(description="Task description")
    task_embedding: list[float] | None = Field(default=None, description="Task embedding")
    steps: list[ReasoningStep] = Field(default_factory=list, description="Reasoning steps")
    outcome: str | None = Field(default=None, description="Final outcome")
    success: bool | None = Field(default=None, description="Whether task succeeded")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")


class Tool(BaseModel):
    """A registered tool that can be used by the agent."""

    name: str = Field(description="Unique tool name")


class ToolStats(BaseModel):
    """Pre-aggregated statistics for a tool.

    These stats are maintained incrementally on Tool nodes for fast retrieval
    without needing to aggregate across all ToolCall nodes.
    """

    name: str = Field(description="Tool name")
    description: str | None = Field(default=None, description="Tool description")
    total_calls: int = Field(default=0, description="Total number of calls")
    successful_calls: int = Field(default=0, description="Number of successful calls")
    failed_calls: int = Field(default=0, description="Number of failed calls (error/timeout)")
    success_rate: float = Field(default=0.0, description="Success rate (0.0 to 1.0)")
    avg_duration_ms: float | None = Field(default=None, description="Average duration in ms")
    last_used_at: datetime | None = Field(default=None, description="Last time tool was used")


def _trace_from_dict(data: dict[str, Any]) -> ReasoningTrace:
    """Build a ReasoningTrace from a flat node property dict."""
    return ReasoningTrace(
        id=UUID(data["id"]),
        session_id=data["session_id"],
        task=data["task"],
        task_embedding=data.get("task_embedding"),
        outcome=data.get("outcome"),
        success=data.get("success"),
        started_at=to_python_datetime(data.get("started_at")),
        completed_at=to_python_datetime(data.get("completed_at"))
        if data.get("completed_at")
        else None,
        metadata=deserialize_metadata(data.get("metadata")),
    )


class StreamingTraceRecorder:
    """
    Context manager for recording traces during streaming agent execution.

    Handles timing automatically and provides convenient methods for recording
    tool calls and observations during streaming responses.

    Example:
        async with StreamingTraceRecorder(memory.reasoning, session_id, "Find restaurants") as recorder:
            step = await recorder.start_step(thought="Searching for restaurants")

            start = time.time()
            result = await search_tool(query="Italian restaurants")
            duration = int((time.time() - start) * 1000)

            await recorder.record_tool_call(
                "search",
                {"query": "Italian"},
                result=result,
                duration_ms=duration,
            )

            await recorder.add_observation(f"Found {len(result)} restaurants")

        # Trace is automatically completed when exiting the context
    """

    def __init__(
        self,
        reasoning_memory: "ReasoningMemory",
        session_id: str,
        task: str,
        *,
        generate_task_embedding: bool = True,
        generate_step_embeddings: bool = False,
    ):
        """
        Initialize the streaming trace recorder.

        Args:
            reasoning_memory: The ReasoningMemory instance to use
            session_id: Session identifier
            task: Task description
            generate_task_embedding: Whether to generate embedding for the task
            generate_step_embeddings: Whether to generate embeddings for steps during
                                     recording. If False, can batch generate at completion
                                     using complete_trace(generate_step_embeddings=True)
        """
        self.memory = reasoning_memory
        self.session_id = session_id
        self.task = task
        self.generate_task_embedding = generate_task_embedding
        self.generate_step_embeddings = generate_step_embeddings

        self.trace: ReasoningTrace | None = None
        self.current_step: ReasoningStep | None = None
        self._step_start_time: datetime | None = None
        self._tool_call_times: dict[str, datetime] = {}
        self._outcome: str | None = None
        self._success: bool | None = None
        self._error: Exception | None = None

    async def __aenter__(self) -> "StreamingTraceRecorder":
        """Start the trace when entering the context."""
        self.trace = await self.memory.start_trace(
            self.session_id,
            self.task,
            generate_embedding=self.generate_task_embedding,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Complete the trace when exiting the context."""
        if self.trace is None:
            return

        # Determine success based on exception
        if exc_type is not None:
            self._success = False
            self._outcome = str(exc_val) if exc_val else f"Error: {exc_type.__name__}"
            self._error = exc_val

        await self.memory.complete_trace(
            self.trace.id,
            outcome=self._outcome,
            success=self._success if self._success is not None else True,
            generate_step_embeddings=self.generate_step_embeddings,
        )

    async def start_step(
        self,
        *,
        thought: str | None = None,
        action: str | None = None,
    ) -> ReasoningStep:
        """
        Start a new reasoning step.

        Args:
            thought: The agent's reasoning/thought
            action: The action being taken

        Returns:
            The created ReasoningStep
        """
        if self.trace is None:
            raise RuntimeError("Trace not started. Use within 'async with' context.")

        self._step_start_time = datetime.utcnow()
        self.current_step = await self.memory.add_step(
            self.trace.id,
            thought=thought,
            action=action,
            generate_embedding=self.generate_step_embeddings,
        )
        return self.current_step

    async def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        result: Any = None,
        status: ToolCallStatus = ToolCallStatus.SUCCESS,
        duration_ms: int | None = None,
        error: str | None = None,
        auto_observation: bool = False,
    ) -> ToolCall:
        """
        Record a tool call with optional automatic timing.

        If no step is active, automatically creates one with action set to the tool name.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            status: Call status
            duration_ms: Duration in milliseconds (if not provided, calculated from step start)
            error: Error message if failed
            auto_observation: If True, set the step's observation from the result

        Returns:
            The created ToolCall
        """
        if self.trace is None:
            raise RuntimeError("Trace not started. Use within 'async with' context.")

        # Auto-create step if none exists
        if self.current_step is None:
            await self.start_step(action=f"call:{tool_name}")

        # Calculate duration if not provided
        if duration_ms is None and self._step_start_time is not None:
            duration_ms = int((datetime.utcnow() - self._step_start_time).total_seconds() * 1000)

        return await self.memory.record_tool_call(
            self.current_step.id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            duration_ms=duration_ms,
            error=error,
            auto_observation=auto_observation,
        )

    async def add_observation(self, observation: str) -> None:
        """
        Add an observation to the current step.

        Args:
            observation: The observation text
        """
        if self.current_step is None:
            raise RuntimeError("No active step. Call start_step() first.")

        await self.memory._client.update_node(
            "ReasoningStep",
            str(self.current_step.id),
            {"observation": observation},
        )

    def set_outcome(self, outcome: str, success: bool = True) -> None:
        """
        Set the outcome for the trace (will be applied on context exit).

        Args:
            outcome: The outcome description
            success: Whether the task succeeded
        """
        self._outcome = outcome
        self._success = success

    @property
    def trace_id(self) -> UUID | None:
        """Get the current trace ID."""
        return self.trace.id if self.trace else None

    @property
    def step_id(self) -> UUID | None:
        """Get the current step ID."""
        return self.current_step.id if self.current_step else None


class ReasoningMemory(BaseMemory[ReasoningStep]):
    """
    Reasoning memory stores reasoning traces and tool usage patterns.

    Provides:
    - Reasoning trace recording
    - Tool call tracking with statistics
    - Similar task retrieval for learning from past experiences
    """

    def __init__(
        self,
        client: "GraphBackend",
        embedder: "Embedder | None" = None,
    ):
        """Initialize reasoning memory."""
        super().__init__(client, embedder, None)

    async def add(self, content: str, **kwargs: Any) -> ReasoningStep:
        """Add content as a reasoning step."""
        trace_id = kwargs.get("trace_id")
        if not trace_id:
            raise ValueError("trace_id is required")
        return await self.add_step(
            trace_id,
            thought=content,
            action=kwargs.get("action"),
            observation=kwargs.get("observation"),
        )

    async def start_trace(
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool = True,
        metadata: dict[str, Any] | None = None,
        triggered_by_message_id: UUID | str | None = None,
    ) -> ReasoningTrace:
        """
        Start a new reasoning trace.

        Args:
            session_id: Session identifier
            task: Task description
            generate_embedding: Whether to generate task embedding
            metadata: Optional metadata
            triggered_by_message_id: Optional message ID that initiated this trace.
                Creates an INITIATED_BY relationship from ReasoningTrace to Message.

        Returns:
            The created reasoning trace
        """
        # Generate task embedding
        task_embedding = None
        if generate_embedding and self._embedder is not None:
            task_embedding = await self._embedder.embed(task)

        trace = ReasoningTrace(
            id=uuid4(),
            session_id=session_id,
            task=task,
            task_embedding=task_embedding,
            metadata=metadata or {},
        )

        await self._client.upsert_node(
            "ReasoningTrace",
            id=str(trace.id),
            properties={
                "session_id": trace.session_id,
                "task": trace.task,
                "task_embedding": trace.task_embedding,
                "outcome": None,
                "success": None,
                "completed_at": None,
                "started_at": datetime.utcnow().isoformat(),
                "metadata": _serialize_json(trace.metadata),
            },
        )

        # Link to message if provided
        if triggered_by_message_id is not None:
            msg_id_str = (
                str(triggered_by_message_id)
                if isinstance(triggered_by_message_id, UUID)
                else triggered_by_message_id
            )
            await self._client.link_nodes(
                "ReasoningTrace",
                str(trace.id),
                "Message",
                msg_id_str,
                "INITIATED_BY",
            )

        return trace

    async def add_step(
        self,
        trace_id: UUID,
        *,
        thought: str | None = None,
        action: str | None = None,
        observation: str | None = None,
        generate_embedding: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningStep:
        """
        Add a reasoning step to a trace.

        Args:
            trace_id: Parent trace ID
            thought: Agent's thought/reasoning
            action: Action taken
            observation: Observation from action
            generate_embedding: Whether to generate step embedding
            metadata: Optional metadata

        Returns:
            The created reasoning step
        """
        # Get current step count by traversing existing steps
        existing_steps = await self._client.traverse(
            "ReasoningTrace",
            str(trace_id),
            relationship_types=["HAS_STEP"],
            target_labels=["ReasoningStep"],
            direction="outgoing",
        )
        step_number = len(existing_steps) + 1

        # Generate embedding
        embedding = None
        if generate_embedding and self._embedder is not None:
            text_parts = []
            if thought:
                text_parts.append(f"Thought: {thought}")
            if action:
                text_parts.append(f"Action: {action}")
            if observation:
                text_parts.append(f"Observation: {observation}")
            if text_parts:
                embedding = await self._embedder.embed(" ".join(text_parts))

        step = ReasoningStep(
            id=uuid4(),
            trace_id=trace_id,
            step_number=step_number,
            thought=thought,
            action=action,
            observation=observation,
            embedding=embedding,
            metadata=metadata or {},
        )

        await self._client.create_node_with_links(
            "ReasoningStep",
            id=str(step.id),
            properties={
                "step_number": step.step_number,
                "thought": step.thought,
                "action": step.action,
                "observation": step.observation,
                "embedding": step.embedding,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": _serialize_json(step.metadata),
            },
            links=[
                {
                    "target_label": "ReasoningTrace",
                    "target_id": str(trace_id),
                    "relationship_type": "HAS_STEP",
                    "direction": "incoming",
                    "properties": {"order": step.step_number},
                },
            ],
        )

        return step

    async def record_tool_call(
        self,
        step_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        result: Any | None = None,
        status: ToolCallStatus = ToolCallStatus.SUCCESS,
        duration_ms: int | None = None,
        error: str | None = None,
        auto_observation: bool = False,
        message_id: UUID | str | None = None,
    ) -> ToolCall:
        """
        Record a tool call within a reasoning step.

        Args:
            step_id: Parent reasoning step ID
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            status: Call status
            duration_ms: Duration in milliseconds
            error: Error message if failed
            auto_observation: If True, automatically set the step's observation field
                from the tool result. Useful for ReAct-style agents where the observation
                is the tool's output.
            message_id: Optional message ID that triggered this tool call.
                Creates a TRIGGERED_BY relationship from ToolCall to Message.

        Returns:
            The created tool call
        """
        tool_call = ToolCall(
            id=uuid4(),
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

        status_value = status.value if hasattr(status, "value") else str(status)

        # 1. Create the ToolCall node linked to its parent ReasoningStep
        await self._client.create_node_with_links(
            "ToolCall",
            id=str(tool_call.id),
            properties={
                "tool_name": tool_name,
                "arguments": _serialize_json(arguments),
                "result": _serialize_json(result) if result is not None else None,
                "status": status_value,
                "duration_ms": duration_ms,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
            },
            links=[
                {
                    "target_label": "ReasoningStep",
                    "target_id": str(step_id),
                    "relationship_type": "USES_TOOL",
                    "direction": "incoming",
                },
            ],
        )

        # 2. Upsert the Tool node (create if it doesn't exist)
        await self._client.upsert_node(
            "Tool",
            id=tool_name,
            properties={
                "name": tool_name,
                "created_at": datetime.utcnow().isoformat(),
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0,
            },
            on_match_update={},  # Don't overwrite existing properties on match
        )

        # 3. Link ToolCall to Tool
        await self._client.link_nodes(
            "ToolCall",
            str(tool_call.id),
            "Tool",
            tool_name,
            "INSTANCE_OF",
        )

        # 4. Update Tool stats (atomic increments)
        is_success = status_value == "success"
        is_failure = status_value in ("error", "timeout")
        await self._client.update_node(
            "Tool",
            tool_name,
            properties={"last_used_at": datetime.utcnow().isoformat()},
            increment={
                "total_calls": 1,
                "successful_calls": 1 if is_success else 0,
                "failed_calls": 1 if is_failure else 0,
                "total_duration_ms": duration_ms or 0,
            },
        )

        # Link to message if provided
        if message_id is not None:
            msg_id_str = str(message_id) if isinstance(message_id, UUID) else message_id
            await self._client.link_nodes(
                "ToolCall",
                str(tool_call.id),
                "Message",
                msg_id_str,
                "TRIGGERED_BY",
            )

        # Auto-populate the step's observation from the tool result
        if auto_observation and result is not None:
            observation = self._format_observation(result)
            await self._client.update_node(
                "ReasoningStep",
                str(step_id),
                {"observation": observation},
            )

        return tool_call

    def _format_observation(self, result: Any, max_length: int = 1000) -> str:
        """
        Format a tool result as an observation string.

        Args:
            result: The tool result to format
            max_length: Maximum length of the observation string

        Returns:
            Formatted observation string
        """
        if isinstance(result, str):
            return result[:max_length] if len(result) > max_length else result
        if isinstance(result, (dict, list)):
            formatted = json.dumps(result, indent=2, default=str)
            return formatted[:max_length] if len(formatted) > max_length else formatted
        return str(result)[:max_length]

    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None = None,
        success: bool | None = None,
        generate_step_embeddings: bool = False,
    ) -> ReasoningTrace:
        """
        Complete a reasoning trace.

        Args:
            trace_id: Trace ID to complete
            outcome: Final outcome description
            success: Whether the task succeeded
            generate_step_embeddings: If True, batch generate embeddings for all steps
                that don't have them yet. Useful when steps were recorded with
                generate_embedding=False during streaming.

        Returns:
            The updated reasoning trace
        """
        # Batch generate step embeddings if requested
        if generate_step_embeddings and self._embedder is not None:
            await self._generate_step_embeddings_batch(trace_id)

        trace_data = await self._client.update_node(
            "ReasoningTrace",
            str(trace_id),
            {
                "outcome": outcome,
                "success": success,
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

        if not trace_data:
            raise ValueError(f"Trace not found: {trace_id}")

        return _trace_from_dict(trace_data)

    async def link_trace_to_message(
        self,
        trace_id: UUID | str,
        message_id: UUID | str,
    ) -> None:
        """
        Link a reasoning trace to the message that initiated it.

        Creates an INITIATED_BY relationship from ReasoningTrace to Message.
        This is useful for linking traces after they've been created, for example
        when the message context wasn't available at trace creation time.

        Args:
            trace_id: The reasoning trace ID
            message_id: The message ID that initiated this trace

        Example:
            # Link a trace to its triggering message
            await memory.reasoning.link_trace_to_message(trace.id, user_message.id)
        """
        trace_id_str = str(trace_id) if isinstance(trace_id, UUID) else trace_id
        msg_id_str = str(message_id) if isinstance(message_id, UUID) else message_id

        await self._client.link_nodes(
            "ReasoningTrace",
            trace_id_str,
            "Message",
            msg_id_str,
            "INITIATED_BY",
        )

    async def _generate_step_embeddings_batch(self, trace_id: UUID) -> int:
        """
        Batch generate embeddings for all steps in a trace that don't have them.

        Args:
            trace_id: Trace ID to process

        Returns:
            Number of steps that had embeddings generated
        """
        if self._embedder is None:
            return 0

        # Get all steps via traversal, then filter for missing embeddings
        all_steps = await self._client.traverse(
            "ReasoningTrace",
            str(trace_id),
            relationship_types=["HAS_STEP"],
            target_labels=["ReasoningStep"],
            direction="outgoing",
        )

        steps_without_embedding = [s for s in all_steps if s.get("embedding") is None]

        if not steps_without_embedding:
            return 0

        # Build texts for embedding
        texts = []
        step_ids = []
        for step in steps_without_embedding:
            text_parts = []
            if step.get("thought"):
                text_parts.append(f"Thought: {step['thought']}")
            if step.get("action"):
                text_parts.append(f"Action: {step['action']}")
            if step.get("observation"):
                text_parts.append(f"Observation: {step['observation']}")
            if text_parts:
                texts.append(" ".join(text_parts))
                step_ids.append(step["id"])

        if not texts:
            return 0

        # Batch generate embeddings
        embeddings = await self._embedder.embed_batch(texts)

        # Batch update steps with embeddings
        for step_id, embedding in zip(step_ids, embeddings):
            await self._client.update_node(
                "ReasoningStep",
                step_id,
                {"embedding": embedding},
            )

        return len(step_ids)

    async def search(self, query: str, **kwargs: Any) -> list[ReasoningStep]:
        """Search is not directly supported for reasoning memory."""
        return []

    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[ReasoningTrace]:
        """
        Find similar past reasoning traces.

        Args:
            task: Task description to match
            limit: Maximum number of results
            success_only: Only return successful traces
            threshold: Minimum similarity threshold

        Returns:
            List of similar reasoning traces
        """
        if self._embedder is None:
            return []

        task_embedding = await self._embedder.embed(task)

        # Fetch more results than needed to account for client-side filtering
        fetch_limit = limit * 3 if success_only else limit

        results = await self._client.vector_search(
            "ReasoningTrace",
            "task_embedding",
            task_embedding,
            limit=fetch_limit,
            threshold=threshold,
            query_text=task,
        )

        traces = []
        for row in results:
            # Apply success_only filter client-side
            if success_only and row.get("success") is not True:
                continue

            trace = ReasoningTrace(
                id=UUID(row["id"]),
                session_id=row["session_id"],
                task=row["task"],
                task_embedding=row.get("task_embedding"),
                outcome=row.get("outcome"),
                success=row.get("success"),
                started_at=to_python_datetime(row.get("started_at")),
                completed_at=to_python_datetime(row.get("completed_at"))
                if row.get("completed_at")
                else None,
                metadata={"similarity": row["_score"]},
            )
            traces.append(trace)

            if len(traces) >= limit:
                break

        return traces

    async def get_tool_usage_stats(
        self,
        tool_name: str | None = None,
    ) -> dict[str, Tool]:
        """
        Get tool usage statistics.

        .. deprecated:: Use get_tool_stats() instead for pre-aggregated stats.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            Dictionary of tool name to Tool statistics
        """
        filters = {"name": tool_name} if tool_name else None
        results = await self._client.query_nodes("Tool", filters=filters)

        tools = {}
        for row in results:
            name = row.get("name") or row.get("id")
            if not name:
                continue

            tools[name] = Tool(
                name=name,
            )

        return tools

    async def get_tool_stats(
        self,
        tool_name: str | None = None,
    ) -> list[ToolStats]:
        """
        Get pre-aggregated tool statistics.

        Returns tool usage statistics that are maintained incrementally on Tool nodes.
        This is much faster than computing stats from ToolCall nodes, especially
        with many tool calls.

        Args:
            tool_name: Optional filter by specific tool name

        Returns:
            List of ToolStats objects ordered by total_calls descending

        Example:
            # Get stats for all tools
            stats = await memory.reasoning.get_tool_stats()
            for tool in stats:
                print(f"{tool.name}: {tool.total_calls} calls, {tool.success_rate:.1%} success")

            # Get stats for a specific tool
            stats = await memory.reasoning.get_tool_stats("search_memory")
        """
        filters = {"name": tool_name} if tool_name else None
        results = await self._client.query_nodes(
            "Tool",
            filters=filters,
            order_by="total_calls",
            order_dir="desc",
        )

        tool_stats = []
        for row in results:
            name = row.get("name") or row.get("id")
            if not name:
                continue

            total_calls = row.get("total_calls", 0) or 0
            successful_calls = row.get("successful_calls", 0) or 0
            failed_calls = row.get("failed_calls", 0) or 0
            total_duration_ms = row.get("total_duration_ms", 0) or 0

            success_rate = (
                float(successful_calls) / total_calls if total_calls > 0 else 0.0
            )
            avg_duration = (
                float(total_duration_ms) / total_calls if total_calls > 0 else None
            )

            tool_stats.append(
                ToolStats(
                    name=name,
                    description=row.get("description"),
                    total_calls=total_calls,
                    successful_calls=successful_calls,
                    failed_calls=failed_calls,
                    success_rate=success_rate,
                    avg_duration_ms=avg_duration,
                )
            )

        return tool_stats

    async def migrate_tool_stats(self) -> dict[str, int]:
        """
        Migrate tool statistics from ToolCall nodes to pre-aggregated Tool node properties.

        This is a one-time migration for existing data. New tool calls automatically
        update the pre-aggregated stats. Run this if you have existing ToolCall data
        that was created before the pre-aggregation optimization was added.

        Returns:
            Dictionary mapping tool names to number of calls migrated

        Example:
            # Run migration for existing data
            migrated = await memory.reasoning.migrate_tool_stats()
            print(f"Migrated stats for {len(migrated)} tools")
            for tool, count in migrated.items():
                print(f"  {tool}: {count} calls")
        """
        # Get all Tool nodes
        tools = await self._client.query_nodes("Tool")

        migrated: dict[str, int] = {}
        for tool_node in tools:
            tool_name = tool_node.get("name") or tool_node.get("id")
            if not tool_name:
                continue

            # Traverse from Tool to its ToolCall nodes via INSTANCE_OF
            tool_calls = await self._client.traverse(
                "Tool",
                tool_name,
                relationship_types=["INSTANCE_OF"],
                target_labels=["ToolCall"],
                direction="incoming",
            )

            total = len(tool_calls)
            success = sum(
                1 for tc in tool_calls if tc.get("status") == "success"
            )
            failed = sum(
                1 for tc in tool_calls
                if tc.get("status") in ("error", "timeout")
            )
            duration = sum(
                tc.get("duration_ms", 0) or 0 for tc in tool_calls
            )

            await self._client.update_node(
                "Tool",
                tool_name,
                {
                    "total_calls": total,
                    "successful_calls": success,
                    "failed_calls": failed,
                    "total_duration_ms": duration,
                },
            )

            migrated[tool_name] = total

        return migrated

    async def get_context(self, query: str, **kwargs: Any) -> str:
        """
        Get reasoning context for similar tasks.

        Args:
            query: Task description to find similar traces
            max_traces: Maximum traces to include
            include_successful_only: Only include successful traces

        Returns:
            Formatted context string
        """
        max_traces = kwargs.get("max_traces", 3)
        success_only = kwargs.get("include_successful_only", True)

        traces = await self.get_similar_traces(query, limit=max_traces, success_only=success_only)

        if not traces:
            return ""

        parts = ["### Similar Past Tasks"]
        for trace in traces:
            similarity = trace.metadata.get("similarity", 0)
            parts.append(f"\n**Task**: {trace.task}")
            parts.append(f"- Similarity: {similarity:.2f}")
            if trace.outcome:
                parts.append(f"- Outcome: {trace.outcome}")
            if trace.success is not None:
                parts.append(f"- Success: {'Yes' if trace.success else 'No'}")

        return "\n".join(parts)

    async def get_trace_with_steps(self, trace_id: UUID) -> ReasoningTrace | None:
        """
        Get a complete trace with all steps and tool calls.

        Args:
            trace_id: Trace ID to retrieve

        Returns:
            Complete reasoning trace or None
        """
        # 1. Get the trace node
        trace_data = await self._client.get_node(
            "ReasoningTrace",
            id=str(trace_id),
        )

        if not trace_data:
            return None

        # 2. Get all steps for this trace
        steps_data = await self._client.traverse(
            "ReasoningTrace",
            str(trace_id),
            relationship_types=["HAS_STEP"],
            target_labels=["ReasoningStep"],
            direction="outgoing",
        )

        # 3. For each step, get its tool calls
        steps = []
        for sd in steps_data:
            step_id = sd["id"]
            tc_data = await self._client.traverse(
                "ReasoningStep",
                step_id,
                relationship_types=["USES_TOOL"],
                target_labels=["ToolCall"],
                direction="outgoing",
            )

            step_tool_calls = []
            for tc in tc_data:
                step_tool_calls.append(
                    ToolCall(
                        id=UUID(tc["id"]),
                        tool_name=tc["tool_name"],
                        arguments=_deserialize_json(tc.get("arguments")) or {},
                        result=_deserialize_json(tc.get("result")),
                        status=ToolCallStatus(tc.get("status", "success")),
                        duration_ms=tc.get("duration_ms"),
                        error=tc.get("error"),
                    )
                )

            step = ReasoningStep(
                id=UUID(sd["id"]),
                trace_id=trace_id,
                step_number=sd["step_number"],
                thought=sd.get("thought"),
                action=sd.get("action"),
                observation=sd.get("observation"),
                tool_calls=step_tool_calls,
            )
            steps.append(step)

        # Sort steps by step number
        steps.sort(key=lambda s: s.step_number)

        return ReasoningTrace(
            id=UUID(trace_data["id"]),
            session_id=trace_data["session_id"],
            task=trace_data["task"],
            task_embedding=trace_data.get("task_embedding"),
            steps=steps,
            outcome=trace_data.get("outcome"),
            success=trace_data.get("success"),
            started_at=to_python_datetime(trace_data.get("started_at")),
            completed_at=to_python_datetime(trace_data.get("completed_at"))
            if trace_data.get("completed_at")
            else None,
        )

    async def get_trace(self, trace_id: UUID | str) -> ReasoningTrace | None:
        """
        Get a trace by ID (alias for get_trace_with_steps).

        Args:
            trace_id: Trace ID to retrieve (UUID or string)

        Returns:
            Complete reasoning trace or None
        """
        if isinstance(trace_id, str):
            try:
                trace_id = UUID(trace_id)
            except ValueError:
                return None
        return await self.get_trace_with_steps(trace_id)

    async def get_session_traces(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> list[ReasoningTrace]:
        """
        Get all traces for a session.

        Args:
            session_id: Session identifier
            limit: Maximum traces to return

        Returns:
            List of reasoning traces for the session
        """
        return await self.list_traces(session_id=session_id, limit=limit)

    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        success_only: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Literal["started_at", "completed_at"] = "started_at",
        order_dir: Literal["asc", "desc"] = "desc",
    ) -> list[ReasoningTrace]:
        """
        List reasoning traces with filtering and pagination.

        Args:
            session_id: Filter by session (optional)
            success_only: Filter by success status (None for all, True for successful,
                         False for failed)
            since: Only traces started after this time
            until: Only traces started before this time
            limit: Maximum traces to return
            offset: Number of traces to skip (for pagination)
            order_by: Field to order by ('started_at' or 'completed_at')
            order_dir: Sort direction ('asc' or 'desc')

        Returns:
            List of ReasoningTrace objects (without steps - use get_trace for full details)
        """
        # Build equality filters for query_nodes
        filters: dict[str, Any] = {}
        if session_id is not None:
            filters["session_id"] = session_id
        if success_only is not None:
            filters["success"] = success_only

        # Fetch more to allow for client-side time filtering
        fetch_limit = limit
        if since or until:
            fetch_limit = (limit + offset) * 3  # Over-fetch to compensate

        results = await self._client.query_nodes(
            "ReasoningTrace",
            filters=filters if filters else None,
            order_by=order_by,
            order_dir=order_dir,
            limit=fetch_limit,
            offset=0 if (since or until) else offset,
        )

        traces = []
        skipped = 0
        for row in results:
            # Client-side time filtering for since/until
            started_at_val = row.get("started_at")
            if started_at_val is not None and (since or until):
                started_at_dt = to_python_datetime(started_at_val)
                if since and started_at_dt < since:
                    continue
                if until and started_at_dt > until:
                    continue

            # Handle offset for time-filtered results
            if since or until:
                if skipped < offset:
                    skipped += 1
                    continue

            trace = _trace_from_dict(row)
            traces.append(trace)

            if len(traces) >= limit:
                break

        return traces


# Backward compatibility alias (deprecated)
ProceduralMemory = ReasoningMemory
