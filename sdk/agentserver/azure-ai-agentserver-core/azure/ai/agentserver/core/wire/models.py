# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wire format dataclasses for the agent backend protocol.

These models are the *only* types that cross the core↔adapter process boundary.
They are deliberately simpler than the full OpenAI Responses API models so that
adapter authors only need to produce a minimal representation. The core server
handles expanding these into the rich OpenAI event lifecycle.

Design principles:
  - All fields are plain Python types (str, int, bool, list, dict) — no
    framework-specific or Azure SDK types leak across the boundary.
  - Every model is a dataclass for easy construction and serialization.
  - Enums are plain strings to allow forward-compatible extensibility.
  - Optional fields default to ``None`` so adapters only set what they need.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations (string-based for extensibility)
# ---------------------------------------------------------------------------


class StreamEventType(str, Enum):
    """Well-known stream event types an adapter can emit."""

    TEXT_DELTA = "text_delta"
    FUNCTION_CALL_ARGUMENTS_DELTA = "function_call_arguments_delta"
    OUTPUT_ITEM_ADDED = "output_item_added"
    OUTPUT_ITEM_DONE = "output_item_done"
    COMPLETED = "completed"
    ERROR = "error"


class OutputItemType(str, Enum):
    """Discriminator for OutputItem."""

    TEXT_MESSAGE = "text_message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"


# ---------------------------------------------------------------------------
# Shared / Leaf types
# ---------------------------------------------------------------------------


@dataclass
class ErrorInfo:
    """Error information returned by an adapter."""

    code: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorInfo":
        return cls(code=data.get("code", ""), message=data.get("message", ""))


@dataclass
class ContentPart:
    """A typed content part within a message.

    Maps loosely to OpenAI's ``ItemContent`` hierarchy but flattened for
    the wire.  Only ``type`` and the relevant value field need to be set.
    """

    type: str = "text"  # "text", "image", "audio", "file"
    text: Optional[str] = None
    # Future: image_url, audio_data, file_id, …
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.extra is not None:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentPart":
        return cls(
            type=data.get("type", "text"),
            text=data.get("text"),
            extra=data.get("extra"),
        )


@dataclass
class ToolCallInfo:
    """A tool/function call inside an assistant message."""

    id: str = ""
    name: str = ""
    arguments: str = ""  # JSON-encoded arguments

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallInfo":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", ""),
        )


@dataclass
class AgentInfo:
    """Identity metadata about the agent, echoed in responses."""

    name: Optional[str] = None
    version: Optional[str] = None
    type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.name is not None:
            d["name"] = self.name
        if self.version is not None:
            d["version"] = self.version
        if self.type is not None:
            d["type"] = self.type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        return cls(
            name=data.get("name"),
            version=data.get("version"),
            type=data.get("type"),
        )


# ---------------------------------------------------------------------------
# Request models  (core → adapter)
# ---------------------------------------------------------------------------


@dataclass
class InputMessage:
    """A single message in the conversation history.

    This is a simplified, flattened representation of OpenAI's
    ``ResponseInputItemParam`` hierarchy.  The ``role`` field indicates
    the kind of message; ``tool_call_id`` and ``tool_calls`` are
    populated only for the relevant roles.
    """

    role: str = "user"  # "user", "assistant", "system", "tool"
    content: str = ""
    content_parts: Optional[List[ContentPart]] = None
    tool_call_id: Optional[str] = None  # set when role == "tool"
    tool_calls: Optional[List[ToolCallInfo]] = None  # set when role == "assistant" with calls

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.content_parts is not None:
            d["content_parts"] = [p.to_dict() for p in self.content_parts]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputMessage":
        content_parts = None
        if "content_parts" in data:
            content_parts = [ContentPart.from_dict(p) for p in data["content_parts"]]
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCallInfo.from_dict(tc) for tc in data["tool_calls"]]
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            content_parts=content_parts,
            tool_call_id=data.get("tool_call_id"),
            tool_calls=tool_calls,
        )


@dataclass
class AgentRequest:
    """The wire-format request sent from core server to an adapter backend.

    This is a simplified, framework-neutral projection of the OpenAI
    Responses API ``CreateResponse`` payload.  The adapter reads these
    fields and maps them into its framework's native types.
    """

    response_id: str = ""
    conversation_id: str = ""
    stream: bool = False
    instructions: Optional[str] = None
    messages: List[InputMessage] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    agent: Optional[AgentInfo] = None
    # Reserved for future extensions — adapters should ignore unknown keys.
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "response_id": self.response_id,
            "conversation_id": self.conversation_id,
            "stream": self.stream,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }
        if self.instructions is not None:
            d["instructions"] = self.instructions
        if self.agent is not None:
            d["agent"] = self.agent.to_dict()
        if self.extensions:
            d["extensions"] = self.extensions
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRequest":
        messages = [InputMessage.from_dict(m) for m in data.get("messages", [])]
        agent = AgentInfo.from_dict(data["agent"]) if data.get("agent") else None
        return cls(
            response_id=data.get("response_id", ""),
            conversation_id=data.get("conversation_id", ""),
            stream=data.get("stream", False),
            instructions=data.get("instructions"),
            messages=messages,
            metadata=data.get("metadata", {}),
            agent=agent,
            extensions=data.get("extensions", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentRequest":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Response models  (adapter → core)
# ---------------------------------------------------------------------------


@dataclass
class OutputItem:
    """A single output item produced by the adapter.

    Use the ``type`` discriminator to indicate which fields are populated:
      - ``text_message`` → ``role``, ``content``
      - ``function_call`` → ``call_id``, ``name``, ``arguments``
      - ``function_call_output`` → ``call_id``, ``output``
    """

    type: str = OutputItemType.TEXT_MESSAGE  # OutputItemType values

    # text_message fields
    role: Optional[str] = None  # "assistant", "user", "system"
    content: Optional[str] = None
    content_parts: Optional[List[ContentPart]] = None

    # function_call fields
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None  # JSON-encoded

    # function_call_output fields
    output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.role is not None:
            d["role"] = self.role
        if self.content is not None:
            d["content"] = self.content
        if self.content_parts is not None:
            d["content_parts"] = [p.to_dict() for p in self.content_parts]
        if self.call_id is not None:
            d["call_id"] = self.call_id
        if self.name is not None:
            d["name"] = self.name
        if self.arguments is not None:
            d["arguments"] = self.arguments
        if self.output is not None:
            d["output"] = self.output
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputItem":
        content_parts = None
        if "content_parts" in data:
            content_parts = [ContentPart.from_dict(p) for p in data["content_parts"]]
        return cls(
            type=data.get("type", OutputItemType.TEXT_MESSAGE),
            role=data.get("role"),
            content=data.get("content"),
            content_parts=content_parts,
            call_id=data.get("call_id"),
            name=data.get("name"),
            arguments=data.get("arguments"),
            output=data.get("output"),
        )


@dataclass
class AgentResponse:
    """The wire-format non-streaming response from an adapter backend.

    The core server translates this into the full OpenAI ``Response``
    object, populating IDs, timestamps, agent identity, etc.
    """

    status: str = "completed"  # "completed", "failed", "incomplete"
    output: List[OutputItem] = field(default_factory=list)
    error: Optional[ErrorInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "output": [item.to_dict() for item in self.output],
        }
        if self.error is not None:
            d["error"] = self.error.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResponse":
        output = [OutputItem.from_dict(item) for item in data.get("output", [])]
        error = ErrorInfo.from_dict(data["error"]) if data.get("error") else None
        return cls(
            status=data.get("status", "completed"),
            output=output,
            error=error,
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentResponse":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Streaming models  (adapter → core, one event at a time)
# ---------------------------------------------------------------------------


@dataclass
class AgentStreamEvent:
    """A single streaming event from the adapter.

    The core server expands these into the full OpenAI ``ResponseStreamEvent``
    lifecycle (created, in_progress, content_part_added, deltas, done, completed).
    Adapters only need to emit simplified events.

    Event types and their required fields:

      ``text_delta``
          Incremental text chunk.  Set ``output_index`` and ``delta``.

      ``function_call_arguments_delta``
          Incremental function-call argument chunk.  Set ``output_index``
          and ``delta``.

      ``output_item_added``
          A new output item started.  Set ``output_index`` and ``item``.

      ``output_item_done``
          An output item finished.  Set ``output_index`` and ``item``
          (with final content populated).

      ``completed``
          The full response is finished.  Set ``response`` with the
          final ``AgentResponse``.

      ``error``
          An error occurred.  Set ``error``.
    """

    type: str = StreamEventType.TEXT_DELTA  # StreamEventType values
    output_index: int = 0
    delta: Optional[str] = None
    item: Optional[OutputItem] = None
    response: Optional[AgentResponse] = None
    error: Optional[ErrorInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type, "output_index": self.output_index}
        if self.delta is not None:
            d["delta"] = self.delta
        if self.item is not None:
            d["item"] = self.item.to_dict()
        if self.response is not None:
            d["response"] = self.response.to_dict()
        if self.error is not None:
            d["error"] = self.error.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStreamEvent":
        item = OutputItem.from_dict(data["item"]) if data.get("item") else None
        response = AgentResponse.from_dict(data["response"]) if data.get("response") else None
        error = ErrorInfo.from_dict(data["error"]) if data.get("error") else None
        return cls(
            type=data.get("type", StreamEventType.TEXT_DELTA),
            output_index=data.get("output_index", 0),
            delta=data.get("delta"),
            item=item,
            response=response,
            error=error,
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentStreamEvent":
        return cls.from_dict(json.loads(text))
