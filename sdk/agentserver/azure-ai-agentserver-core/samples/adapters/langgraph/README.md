# LangGraph Wire-Protocol Adapter

Self-contained adapter for serving LangGraph agents via the Foundry Agent Server.

## Files

- **`langgraph_adapter.py`** — The adapter (~350 lines). Copy this into your project.
- **`requirements.txt`** — Python dependencies (install alongside `azure-ai-agentserver-core`).
- **`examples/`** — Runnable sample applications.

## Quick Start

```python
from langgraph_adapter import from_langgraph

# Build your LangGraph graph...
graph = builder.compile()

# Wrap it and run the server
agent = from_langgraph(graph)
agent.run(port=8088)
```

## Custom State Schemas

For graphs that don't use `MessagesState`, provide a `WireStateConverter`:

```python
from azure.ai.agentserver.core import WireStateConverter
from langgraph_adapter import from_langgraph

class MyConverter(WireStateConverter):
    def request_to_state(self, request): ...
    def state_to_response(self, state): ...
    def state_to_stream(self, stream, request): ...

agent = from_langgraph(graph, state_converter=MyConverter())
agent.run(port=8088)
```

See `examples/custom_state/` for a complete example.

## Examples

| Example | Description |
|---------|-------------|
| `simple_react_agent/` | Minimal ReAct agent |
| `agent_calculator/` | Calculator tool agent |
| `custom_state/` | Custom state schema with `WireStateConverter` |
| `mcp_simple/` | MCP server integration |
| `mcp_apikey/` | MCP with API key auth |
| `simple_agent_with_redis_checkpointer/` | Redis-backed checkpointer |
| `agentic_rag/` | Multi-node RAG workflow |
