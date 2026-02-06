# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wire format models for the agent service protocol.

These dataclasses define the simplified, framework-neutral wire format
that crosses the boundary between the FoundryCBAgent server and
out-of-process adapter backends. The wire format is intentionally simpler
than the OpenAI Responses API — the core server owns the translation to/from
the full OpenAI event lifecycle.

Serialization:
  - JSON for HTTP transport
  - Protobuf for gRPC transport (see protos/agent_service.proto)
"""

from .models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ContentPart,
    ErrorInfo,
    InputMessage,
    OutputItem,
    StreamEventType,
    ToolCallInfo,
)
from .state_converter import WireStateConverter

__all__ = [
    "AgentInfo",
    "AgentRequest",
    "AgentResponse",
    "AgentStreamEvent",
    "ContentPart",
    "ErrorInfo",
    "InputMessage",
    "OutputItem",
    "StreamEventType",
    "ToolCallInfo",
    "WireStateConverter",
]
