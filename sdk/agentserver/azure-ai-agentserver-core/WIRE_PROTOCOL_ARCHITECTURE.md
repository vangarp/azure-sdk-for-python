# Wire Protocol Architecture

## Overview

The FoundryCBAgent wire protocol decouples framework-specific adapters
(LangGraph, Semantic Kernel, CrewAI, AutoGen, ...) from the core HTTP
server.  Instead of inheriting from a base class and importing heavy Azure
SDK / OpenAI model trees, adapters communicate through a small set of
plain dataclasses serialized over gRPC.

```
┌──────────────────────────────┐          wire format          ┌─────────────────────────────┐
│     FoundryCBAgent           │   ←── gRPC (loopback) ──→    │     GrpcAdapterServer        │
│     (core HTTP server)       │                               │     (adapter process)        │
│                              │                               │                              │
│  • OpenAI Responses API      │                               │  • LangGraphWireAdapter      │
│  • SSE streaming             │   AgentRequest ──────→        │  • CrewAIAdapter             │
│  • Tracing / OTel            │   AgentResponse ←─────        │  • SemanticKernelAdapter     │
│  • Protocol translation      │   AgentStreamEvent ←──        │  • (your framework here)     │
│  • Health endpoints          │                               │                              │
└──────────────────────────────┘                               └─────────────────────────────┘
```

In **LocalBackend** mode (the default and recommended path) both sides
live in the same Python process.  A gRPC loopback channel on an ephemeral
port carries the traffic, preserving the serialization boundary without
requiring two separate deployments.

---

## Why?

### Before: ABC inheritance

```python
# Old world — adapter IS-A FoundryCBAgent
class LangGraphAdapter(FoundryCBAgent):           # tight coupling
    async def agent_run(self, context):            # must import all core models
        ...                                        # every core version bump = adapter rebuild
```

Problems:

| Problem | Impact |
|---------|--------|
| **Dependency hell** | Adapter packages transitively pull in the entire core model tree (`openai`, `azure-ai-agents`, `azure-ai-projects`, ...). A single breaking change in any of those forces every adapter to re-release. |
| **Version lockstep** | Core and adapter must be released in sync; SemVer ranges become impossible to enforce. |
| **Framework isolation** | A bug in LangChain message serialization can crash the core HTTP server. |
| **Testing** | Integration tests require the full Azure environment; you can't test the adapter in isolation. |
| **Polyglot** | Only Python adapters are possible. |

### After: wire protocol

```python
# New world — adapter runs behind a protocol boundary
class CrewAIAdapter(GrpcAdapterServer):
    async def on_run(self, request: AgentRequest) -> AgentResponse:
        ...   # only depends on wire.models (5 dataclasses)
    async def on_run_stream(self, request: AgentRequest):
        ...   # yields AgentStreamEvent
```

Benefits:

| Benefit | Detail |
|---------|--------|
| **Minimal surface** | Adapters depend only on `wire.models` — 5 dataclasses, zero Azure SDK types. |
| **Independent versioning** | Core and adapters evolve on their own release cadence. |
| **Fault isolation** | Adapter exceptions can't crash the core server; errors are marshalled as `ErrorInfo`. |
| **Testable in isolation** | Adapters can be unit-tested with plain `AgentRequest` / `AgentResponse` objects — no OpenAI/Azure SDK required. |
| **Polyglot-ready** | The `.proto` definition allows adapters in Go, Rust, Java, C#, etc. |
| **Streaming simplified** | Adapters emit 6 event types; the core expands them into the full 15+ OpenAI event lifecycle. |

---

## Wire Format at a Glance

### Types an adapter sees

```
AgentRequest
  ├── response_id: str
  ├── conversation_id: str
  ├── stream: bool
  ├── instructions: str | None
  ├── messages: list[InputMessage]
  │     ├── role: "user" | "assistant" | "system" | "tool"
  │     ├── content: str
  │     ├── tool_call_id: str | None
  │     └── tool_calls: list[ToolCallInfo] | None
  ├── metadata: dict[str, str]
  └── extensions: dict[str, Any]      ← reserved for future APIs

AgentResponse
  ├── status: "completed" | "failed" | "incomplete"
  ├── output: list[OutputItem]
  └── error: ErrorInfo | None

OutputItem
  ├── type: "text_message" | "function_call" | "function_call_output"
  ├── role / content          (text_message)
  ├── call_id / name / arguments   (function_call)
  └── call_id / output        (function_call_output)

AgentStreamEvent
  ├── type: text_delta | function_call_arguments_delta
  │         | output_item_added | output_item_done
  │         | completed | error
  ├── output_index: int
  ├── delta: str | None
  ├── item: OutputItem | None
  └── error: ErrorInfo | None
```

That's the **entire API surface** an adapter author needs to learn.

---

## LocalBackend — how it works

`LocalBackend` starts the adapter's gRPC server on `127.0.0.1` with an
auto-selected ephemeral port, then connects a `GrpcBackendClient` to it.
All traffic still passes through protobuf serialization/deserialization,
so the contract is identical to a true out-of-process deployment.

```
                     same Python process
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  FoundryCBAgent ──→ LocalBackend ──→ GrpcBackendClient   │
│       ▲                                    │             │
│       │                                    ▼             │
│       │                         gRPC loopback :0         │
│       │                                    │             │
│       │                                    ▼             │
│       │                          GrpcAdapterServer       │
│       │                          (your adapter here)     │
│       │                                    │             │
│  HTTP ←──── SSE ←──── OpenAI events ←── wire events     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Lifecycle:

1. `FoundryCBAgent(backend=LocalBackend(adapter))` — nothing starts yet.
2. `server.run(port=8088)` — starts the HTTP server.
3. First request arrives → `LocalBackend.start()` launches the adapter
   gRPC server on an ephemeral port and creates a `GrpcBackendClient`.
4. `request_to_wire()` converts the OpenAI Responses API payload into an
   `AgentRequest`.
5. The `GrpcBackendClient` serializes the request into protobuf, sends it
   over the loopback channel.
6. The adapter's `on_run()` / `on_run_stream()` implementation executes,
   returning wire-format responses/events.
7. The core's **protocol translator** expands the simplified wire events
   into the full OpenAI event lifecycle (ResponseCreated, InProgress,
   ContentPartAdded, TextDelta, ..., Completed).
8. SSE chunks are streamed back to the HTTP client.

---

## Writing a New Adapter — CrewAI Example

Below is a complete example of integrating [CrewAI](https://docs.crewai.com/)
with the wire protocol.  The adapter is a **single self-contained file** that
only uses the wire dataclasses — no OpenAI SDK types, no Azure SDK types.

### File layout

Adapters live under `samples/adapters/` in the core package:

```
samples/adapters/crewai/
├── crewai_adapter.py        # the adapter (~300–500 lines)
├── requirements.txt         # crewai + core deps
├── README.md
└── examples/
    └── simple_crew/
        ├── main.py
        └── requirements.txt
```

### requirements.txt

```
azure-ai-agentserver-core
crewai>=0.80.0
```

> At runtime the only things imported from `azure-ai-agentserver-core` are
> `GrpcAdapterServer` and the `wire.models` dataclasses.  The adapter never
> touches OpenAI SDK types, Azure SDK types, or Starlette.

### adapter.py

```python
"""CrewAI adapter for FoundryCBAgent."""
from __future__ import annotations

from typing import AsyncGenerator

from crewai import Agent, Crew, Process, Task

from azure.ai.agentserver.core.server.adapter_server import GrpcAdapterServer
from azure.ai.agentserver.core.wire.models import (
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ErrorInfo,
    OutputItem,
    OutputItemType,
    StreamEventType,
)


class CrewAIAdapter(GrpcAdapterServer):
    """Wire-protocol adapter that runs a CrewAI Crew.

    :param crew: A fully configured CrewAI Crew instance.
    """

    def __init__(self, crew: Crew):
        self.crew = crew

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        """Execute the crew and return the final result."""
        try:
            # Extract the user's latest message as the crew input
            user_input = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_input = msg.content
                    break

            # CrewAI's kickoff is synchronous — run in executor
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.crew.kickoff(inputs={"query": user_input})
            )

            return AgentResponse(
                status="completed",
                output=[
                    OutputItem(
                        type=OutputItemType.TEXT_MESSAGE,
                        role="assistant",
                        content=str(result),
                    )
                ],
            )
        except Exception as e:
            return AgentResponse(
                status="failed",
                error=ErrorInfo(code="crewai_error", message=str(e)),
            )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Stream crew execution with per-task updates."""
        try:
            user_input = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_input = msg.content
                    break

            # CrewAI supports step callbacks — use them to emit events
            import asyncio

            task_results = []

            def on_task_complete(task_output):
                task_results.append(str(task_output))

            # Run kickoff in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.crew.kickoff(inputs={"query": user_input}),
            )

            # Emit intermediate task results as separate output items
            for i, task_text in enumerate(task_results):
                item = OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content=task_text,
                )
                yield AgentStreamEvent(
                    type=StreamEventType.OUTPUT_ITEM_ADDED,
                    output_index=i,
                    item=item,
                )
                yield AgentStreamEvent(
                    type=StreamEventType.OUTPUT_ITEM_DONE,
                    output_index=i,
                    item=item,
                )

            # Emit final result
            final_idx = len(task_results)
            final_item = OutputItem(
                type=OutputItemType.TEXT_MESSAGE,
                role="assistant",
                content=str(result),
            )
            yield AgentStreamEvent(
                type=StreamEventType.OUTPUT_ITEM_ADDED,
                output_index=final_idx,
                item=final_item,
            )
            yield AgentStreamEvent(
                type=StreamEventType.TEXT_DELTA,
                output_index=final_idx,
                delta=str(result),
            )
            yield AgentStreamEvent(
                type=StreamEventType.OUTPUT_ITEM_DONE,
                output_index=final_idx,
                item=final_item,
            )

            # Signal completion
            yield AgentStreamEvent(
                type=StreamEventType.COMPLETED,
                output_index=0,
            )
        except Exception as e:
            yield AgentStreamEvent(
                type=StreamEventType.ERROR,
                error=ErrorInfo(code="crewai_error", message=str(e)),
            )
```

### __init__.py

```python
from azure.ai.agentserver.core import FoundryCBAgent, LocalBackend
from .adapter import CrewAIAdapter


def from_crewai(crew) -> FoundryCBAgent:
    """Create a FoundryCBAgent backed by a CrewAI Crew.

    :param crew: A configured CrewAI Crew.
    :return: A server ready to .run().
    """
    adapter = CrewAIAdapter(crew)
    return FoundryCBAgent(backend=LocalBackend(adapter))
```

### Running it

```python
from crewai import Agent, Crew, Process, Task

researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="You are a skilled researcher.",
)
task = Task(
    description="Research {query}",
    expected_output="A detailed summary",
    agent=researcher,
)
crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)

# One line to serve it as an OpenAI-compatible endpoint:
from azure.ai.agentserver.crewai import from_crewai
server = from_crewai(crew)
server.run(port=8088)
```

```bash
# Test it
curl -X POST http://localhost:8088/responses \
  -H "Content-Type: application/json" \
  -d '{"input": "What is quantum computing?", "stream": false}'
```

---

## Event Translation — What the Core Does for You

Adapters emit a minimal set of events.  The core's **protocol translator**
automatically wraps them into the full OpenAI Responses API event stream:

| Adapter emits | Core generates upstream |
|---------------|----------------------|
| *(first event arrives)* | `response.created` → `response.in_progress` |
| `output_item_added` | `response.output_item.added` → `response.content_part.added` |
| `text_delta` | `response.output_text.delta` |
| `function_call_arguments_delta` | `response.function_call_arguments.delta` |
| `output_item_done` | `response.output_text.done` → `response.content_part.done` → `response.output_item.done` |
| `completed` | `response.completed` (with full Response body) |
| `error` | `error` event + `data: [DONE]` |

Adapter authors never need to think about `response.created`,
`content_part.added`, sequence numbers, or the Response envelope.

---

## Adding Support for a New API

The wire format (`AgentRequest` / `AgentResponse` / `AgentStreamEvent`)
is deliberately **API-agnostic**.  It captures the common patterns —
messages, tool calls, streaming deltas — without being tied to any
particular HTTP API surface.  This means you can expose entirely new
client-facing APIs without touching a single adapter.

```
                         ┌────────────────────────┐
  POST /responses ──────→│  Responses API         │
  (OpenAI Responses)     │  protocol_translator   │──┐
                         └────────────────────────┘  │
                                                     │   AgentRequest
                         ┌────────────────────────┐  ├─────────────────→  Adapter
  POST /v1/chat/   ────→│  Chat Completions      │──┘   AgentResponse     (unchanged)
  completions            │  protocol_translator   │  ←─────────────────
  (new!)                 └────────────────────────┘  │   AgentStreamEvent
                                                     │
                         ┌────────────────────────┐  │
  POST /my-api ────────→│  Custom API            │──┘
  (new!)                 │  protocol_translator   │
                         └────────────────────────┘
```

Every API surface follows the same pattern:

1. **Parse** the incoming HTTP request into the API's native types.
2. **Translate** those types into a wire-format `AgentRequest` (one
   function: `request_to_wire`).
3. **Send** the `AgentRequest` to the backend (unchanged — same
   `BackendClient.run()` / `run_stream()` call).
4. **Translate** the wire-format `AgentResponse` / `AgentStreamEvent`
   back into the API's native response types (two functions:
   `wire_response_to_*` and `wire_stream_to_*`).
5. **Serialize** the response to HTTP (JSON, SSE, etc.).

Steps 1–2 and 4–5 live in a **protocol translator** — a pure-function
module that has no I/O, no gRPC, no adapter awareness.  Step 3 is the
existing backend call that all API surfaces share.

### Step-by-step: adding OpenAI Chat Completions

Below is a concrete walkthrough of adding the Chat Completions API
(`POST /v1/chat/completions`) alongside the existing Responses API.

#### 1. Write the protocol translator

Create `server/chat_completions_translator.py`.  It converts between
Chat Completions types and the wire format — same role as the existing
`protocol_translator.py` but for a different HTTP schema:

```python
"""Chat Completions API ↔ Wire format translator."""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List

from ..wire.models import (
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    InputMessage,
    OutputItem,
    OutputItemType,
    StreamEventType,
    ToolCallInfo,
)


# ── Inbound: Chat Completions request → wire ──────────────────────────

def chat_request_to_wire(payload: dict) -> AgentRequest:
    """Convert a Chat Completions ``/v1/chat/completions`` payload into
    a wire-format ``AgentRequest``.

    The caller is responsible for parsing the raw HTTP body; this
    function is a pure dict→dataclass transform.
    """
    messages: List[InputMessage] = []
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        tool_call_id = msg.get("tool_call_id")
        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                ToolCallInfo(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                )
                for tc in msg["tool_calls"]
            ]

        messages.append(InputMessage(
            role=role,
            content=content if isinstance(content, str) else str(content),
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        ))

    return AgentRequest(
        response_id=str(uuid.uuid4()),
        conversation_id=payload.get("user", ""),
        stream=payload.get("stream", False),
        instructions=_extract_system_prompt(payload),
        messages=messages,
        metadata={},
    )


def _extract_system_prompt(payload: dict) -> str | None:
    """Pull the system message out as instructions, if present."""
    for msg in payload.get("messages", []):
        if msg.get("role") == "system":
            return msg.get("content", "")
    return None


# ── Outbound: wire → Chat Completions response ────────────────────────

def wire_response_to_chat(
    wire: AgentResponse,
    model: str = "foundry-agent",
) -> dict:
    """Convert a wire ``AgentResponse`` into a Chat Completions JSON body."""
    choices = []
    for i, item in enumerate(wire.output):
        if item.type == OutputItemType.TEXT_MESSAGE:
            choices.append({
                "index": i,
                "message": {"role": item.role or "assistant", "content": item.content},
                "finish_reason": "stop",
            })
        elif item.type == OutputItemType.FUNCTION_CALL:
            choices.append({
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }],
                },
                "finish_reason": "tool_calls",
            })

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices or [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Outbound: wire stream → Chat Completions SSE chunks ───────────────

async def wire_stream_to_chat(
    events: AsyncIterator[AgentStreamEvent],
    model: str = "foundry-agent",
) -> AsyncGenerator[dict, None]:
    """Convert wire stream events into Chat Completions SSE chunk dicts."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async for event in events:
        if event.type == StreamEventType.TEXT_DELTA:
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": event.output_index,
                    "delta": {"content": event.delta},
                    "finish_reason": None,
                }],
            }
        elif event.type == StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA:
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": event.output_index,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": event.delta},
                        }],
                    },
                    "finish_reason": None,
                }],
            }
        elif event.type == StreamEventType.COMPLETED:
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
```

Note: this file imports **only** from `wire.models` — no Starlette, no
gRPC, no framework types.  It is pure transformation logic, trivially
unit-testable.

#### 2. Register the HTTP route

In `base.py` (or in a new server subclass), add a route that uses the
new translator:

```python
from .chat_completions_translator import (
    chat_request_to_wire,
    wire_response_to_chat,
    wire_stream_to_chat,
)

async def chat_completions_endpoint(request: Request):
    payload = await request.json()
    wire_request = chat_request_to_wire(payload)
    model = payload.get("model", "foundry-agent")

    if wire_request.stream:
        wire_events = self._backend.run_stream(wire_request)
        chat_chunks = wire_stream_to_chat(wire_events, model=model)

        async def sse_gen():
            async for chunk in chat_chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    wire_response = await self._backend.run(wire_request)
    return JSONResponse(wire_response_to_chat(wire_response, model=model))

# Add to routes list:
Route("/v1/chat/completions", chat_completions_endpoint, methods=["POST"]),
```

That's it.  **No adapter changes required.**  Every existing adapter —
LangGraph, CrewAI, Semantic Kernel — automatically works with the Chat
Completions endpoint because the wire format is identical.

#### 3. (Optional) Add API-specific middleware

If the new API has authentication, rate-limiting, or payload validation
requirements, add them as Starlette middleware scoped to the new routes.
The middleware sits above the translator and never touches the wire
format.

### What changes vs. what stays the same

| Layer | Changes? | What to do |
|-------|----------|------------|
| Wire format (`wire/models.py`) | **No** | Nothing — it's the stable contract |
| Protobuf / gRPC transport | **No** | Same `AgentRequest` / `AgentResponse` messages |
| `BackendClient` / `LocalBackend` | **No** | Same `run()` / `run_stream()` calls |
| All adapters | **No** | They only speak wire format |
| Protocol translator | **New file** | One module: inbound + outbound conversion |
| HTTP routes (`base.py`) | **Small addition** | Register the new route, call the new translator |
| Tests | **New file** | Unit-test the new translator in isolation |

### Design guidelines for new translators

1. **Keep translators pure.**  They should be stateless functions that
   convert dicts/dataclasses in → dicts/dataclasses out.  No I/O, no
   side effects, no `await` in the request path (streaming generators
   are the exception).

2. **Don't stretch the wire format.**  If the new API has a concept that
   genuinely can't be expressed in the current wire types, propose an
   extension to `wire/models.py` (e.g. a new `OutputItemType` or a new
   field on `AgentRequest.extensions`).  All translators and adapters
   benefit from shared vocabulary.

3. **Prefer `extensions` for API-specific pass-through.**  The
   `AgentRequest.extensions` dict exists for fields that only one API
   surface cares about.  For example, Chat Completions' `temperature`,
   `top_p`, and `max_tokens` can be forwarded as:
   ```python
   AgentRequest(
       ...,
       extensions={
           "chat_completions": {
               "temperature": 0.7,
               "top_p": 1.0,
               "max_tokens": 4096,
           }
       },
   )
   ```
   Adapters that understand these extensions can use them; others safely
   ignore them.

4. **One translator per API surface.**  Don't mix Chat Completions logic
   into the Responses API translator.  Each `*_translator.py` is a
   self-contained, independently testable module.

5. **Test the translator, not the plumbing.**  Unit-test the translator
   by calling its functions directly with dict/dataclass inputs.  Save
   integration tests (HTTP → backend round-trip) for a separate suite.

---

## Custom State Converters

For frameworks that don't use a standard chat-message interface (e.g. a
LangGraph using a custom state schema), adapters can accept a
**WireStateConverter** — a simple protocol that maps between the wire
format and the framework's native state representation:

```python
class WireStateConverter(ABC):
    """Converts wire format ↔ framework-native state."""

    @abstractmethod
    def request_to_state(self, request: AgentRequest) -> Any:
        """Convert a wire AgentRequest into framework-native input."""
        ...

    @abstractmethod
    def state_to_response(self, state: Any) -> AgentResponse:
        """Convert framework-native output into a wire AgentResponse."""
        ...

    @abstractmethod
    async def state_to_stream(
        self, stream: AsyncIterator, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Convert framework-native streaming output into wire events."""
        ...
```

The converter only touches wire types — no OpenAI or Azure SDK imports
needed.  See the LangGraph adapter's `LangGraphWireAdapter` for a
concrete example.

---

## Project Structure

```
azure-ai-agentserver-core/
├── protos/
│   └── agent_service.proto          # gRPC service definition
├── azure/ai/agentserver/core/
│   ├── wire/
│   │   ├── __init__.py
│   │   └── models.py                # AgentRequest, AgentResponse, etc.
│   ├── proto/
│   │   ├── __init__.py
│   │   ├── agent_service_pb2.py     # generated protobuf stubs
│   │   └── agent_service_pb2_grpc.py
│   └── server/
│       ├── backend.py               # BackendClient ABC
│       ├── local_backend.py         # LocalBackend (in-process gRPC loopback)
│       ├── grpc_backend.py          # GrpcBackendClient
│       ├── adapter_server.py        # GrpcAdapterServer base class
│       ├── protocol_translator.py   # OpenAI ↔ wire conversion
│       └── base.py                  # FoundryCBAgent
│
samples/adapters/                    # reference adapter implementations
├── README.md
├── langgraph/
│   ├── langgraph_adapter.py         # LangGraphWireAdapter (self-contained)
│   ├── requirements.txt
│   └── examples/
├── agentframework/
│   ├── agentframework_adapter.py    # AgentFrameworkWireAdapter
│   ├── requirements.txt
│   └── examples/
```

---

## FAQ

**Q: Is the gRPC loopback in LocalBackend expensive?**

No.  On `localhost`, gRPC uses Unix-domain or loopback TCP sockets with
zero network hops.  The serialization overhead (protobuf encode/decode)
is negligible compared to the LLM latency inside the adapter.  The
benefit — enforcing the protocol boundary — catches contract violations
early and ensures the adapter works identically whether it's in-process
or out-of-process.

**Q: Can I test my adapter without the core server?**

Yes.  Instantiate your adapter directly and call `on_run()` /
`on_run_stream()` with hand-built `AgentRequest` objects:

```python
adapter = CrewAIAdapter(crew=my_crew)
response = await adapter.on_run(AgentRequest(
    response_id="test",
    conversation_id="test",
    messages=[InputMessage(role="user", content="Hello")],
))
assert response.status == "completed"
```

No HTTP server, no OpenAI SDK, no Azure credentials.

**Q: What about tool calling / function calling?**

Fully supported.  Adapters emit `OutputItem(type="function_call", ...)`
and accept `InputMessage(role="tool", ...)` for results.  The core
translates these to/from the OpenAI `function_call` / `function_call_output`
item types automatically.

**Q: Can adapters be written in languages other than Python?**

Yes.  The `protos/agent_service.proto` file defines the full gRPC service.
Generate stubs for Go, Rust, Java, C#, etc. with `protoc` and implement
`AgentAdapterService`.  Connect the core server to the adapter via
`GrpcBackendClient("host:port")`.

**Q: If I add a new API surface, do adapters need to change?**

No.  Adapters only speak wire format.  A new API surface means a new
protocol translator and new HTTP routes in the core server.  All existing
adapters automatically work with the new API because the wire contract is
unchanged.  See [Adding Support for a New API](#adding-support-for-a-new-api).
