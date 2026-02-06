# Writing Wire-Protocol Adapters

This directory contains **reference adapter implementations** for connecting
third-party agent frameworks to the Foundry Agent Server via the wire protocol.

Each adapter is a single, self-contained Python file (~400-500 lines) that you
can copy into your own project and modify as needed.

## Available Adapters

| Framework | Adapter file | Description |
|-----------|-------------|-------------|
| **LangGraph** | [`langgraph/langgraph_adapter.py`](langgraph/langgraph_adapter.py) | Wraps a compiled `StateGraph` |
| **Agent Framework** | [`agentframework/agentframework_adapter.py`](agentframework/agentframework_adapter.py) | Wraps an `AgentProtocol` |

## How It Works

Every adapter follows the same pattern:

1. **Subclass `GrpcAdapterServer`** from `azure-ai-agentserver-core`
2. **Implement `on_run()`** — convert `AgentRequest` → framework input, run the
   agent, convert output → `AgentResponse`
3. **Implement `on_run_stream()`** — same, but yield `AgentStreamEvent` objects
4. **Provide a factory function** (e.g. `from_langgraph()`) that wires up a
   `LocalBackend` for in-process usage

The wire protocol uses three simple dataclasses as the contract:

- **`AgentRequest`** — messages, instructions, conversation ID, metadata
- **`AgentResponse`** — status + list of `OutputItem` (text, function call, etc.)
- **`AgentStreamEvent`** — typed stream events (text delta, item added/done, etc.)

The core server handles all OpenAI Responses API complexity (event lifecycle,
SSE formatting, response assembly). Adapters only deal with these simple types.

## Adding a New Framework (e.g. CrewAI)

1. Copy one of the existing adapter files as a starting point
2. Replace the framework-specific imports and conversion logic
3. Implement the `on_run()` and `on_run_stream()` methods
4. Add a factory function for convenience
5. Create an `examples/` directory with runnable samples

See [WIRE_PROTOCOL_ARCHITECTURE.md](../../WIRE_PROTOCOL_ARCHITECTURE.md) for
the full architecture documentation.

## Running an Example

```bash
cd langgraph/
pip install -r requirements.txt
cd examples/simple_react_agent/
pip install -r requirements.txt
python main.py
```
