"""Tool call tracking for streaming LangGraph responses.

Tracks tool calls as they stream in via chunk fragments, accumulating
args across multiple chunks and matching tool results to their calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ActiveToolCall:
    """Represents a single tool call observed during a LangGraph stream.

    Args in LangGraph streams arrive as JSON string fragments spread across
    multiple AIMessageChunk chunks. This class accumulates those fragments
    so the full args can be reconstructed later.

    Attributes:
        call_id: Unique tool call ID (e.g. "call_abc123" or "call_0" for index-based)
        name: Tool function name (may be "..." until the name chunk arrives)
        args_chunks: List of JSON string fragments to accumulate into full args
        status: Current status: "in_progress", "complete", or "error"
        result: Tool result content (set when a matching ToolMessage arrives)
    """

    call_id: str
    name: str
    args_chunks: List[str] = field(default_factory=list)
    status: str = "in_progress"
    result: Optional[str] = None

    def args_json(self) -> str:
        """Return the accumulated args as a single JSON string."""
        return "".join(self.args_chunks)


class ToolCallTracker:
    """Tracks tool calls across streaming LangGraph message chunks.

    LangGraph streams tool calls as fragments via AIMessageChunk.tool_call_chunks.
    Each fragment may include a call ID, tool name, and partial JSON args.
    This class assembles the full picture and matches tool results to calls.

    Usage:
        tracker = ToolCallTracker()

        # In the streaming loop:
        for tc_chunk in message_data.get("tool_call_chunks", []):
            is_new = tracker.handle_chunk(tc_chunk)
            if is_new:
                # A new tool call started

        # When a ToolMessage arrives:
        tracker.handle_result(tool_call_id, result_content)

        # Get all tracked calls:
        calls = tracker.get_calls()
    """

    def __init__(self) -> None:
        # Maps call_id -> ActiveToolCall
        self.calls: dict[str, ActiveToolCall] = {}
        # Maps chunk index (int) -> call_id for chunks without explicit IDs
        self._index_to_id: dict[int, str] = {}

    def handle_chunk(self, tc_chunk: dict) -> bool:
        """Process a single tool_call_chunk from an AIMessageChunk.

        Args:
            tc_chunk: A dict from message_data["tool_call_chunks"] containing:
                - id: Optional unique tool call ID
                - name: Optional tool function name
                - args: Optional JSON string fragment
                - index: Integer chunk index (used as fallback key when id is absent)

        Returns:
            True if this chunk started a NEW tool call, False otherwise.
        """
        call_id: Optional[str] = tc_chunk.get("id")
        name: str = tc_chunk.get("name", "")
        args: str = tc_chunk.get("args", "")
        index: int = tc_chunk.get("index", 0)

        if call_id:
            if call_id not in self.calls:
                # New tool call with an explicit ID
                self.calls[call_id] = ActiveToolCall(
                    call_id=call_id,
                    name=name or "...",
                )
                if args:
                    self.calls[call_id].args_chunks.append(args)
                self._index_to_id[index] = call_id
                return True
            else:
                # Existing call - accumulate name and args
                call = self.calls[call_id]
                if name and call.name == "...":
                    call.name = name
                if args:
                    call.args_chunks.append(args)
                return False
        else:
            # No explicit ID - use index as fallback
            if index not in self._index_to_id:
                # New call identified only by index
                synthetic_id = f"call_{index}"
                self.calls[synthetic_id] = ActiveToolCall(
                    call_id=synthetic_id,
                    name=name or "...",
                )
                if args:
                    self.calls[synthetic_id].args_chunks.append(args)
                self._index_to_id[index] = synthetic_id
                return True
            else:
                # Existing call identified by index
                existing_id = self._index_to_id[index]
                call = self.calls[existing_id]
                if name and call.name == "...":
                    call.name = name
                if args:
                    call.args_chunks.append(args)
                return False

    def handle_result(self, tool_call_id: str, result: str) -> None:
        """Mark a tool call as complete and store its result.

        Matches by tool_call_id. If not found (e.g. index-based ID mismatch),
        falls back to marking the first in-progress call as complete.

        Args:
            tool_call_id: The tool_call_id from the ToolMessage
            result: The tool's return value as a string
        """
        if tool_call_id in self.calls:
            self.calls[tool_call_id].status = "complete"
            self.calls[tool_call_id].result = result
        else:
            # Fallback: mark first in-progress call without a result
            for call in self.calls.values():
                if call.status == "in_progress":
                    call.status = "complete"
                    call.result = result
                    break

    def get_calls(self) -> List[ActiveToolCall]:
        """Return all tracked tool calls in insertion order."""
        return list(self.calls.values())

    def has_calls(self) -> bool:
        """Return True if any tool calls have been tracked."""
        return len(self.calls) > 0
