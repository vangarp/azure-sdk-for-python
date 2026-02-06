# Agent Framework Wire-Protocol Adapter

Self-contained adapter for serving Agent Framework agents via the Foundry Agent Server.

## Files

- **`agentframework_adapter.py`** — The adapter (~350 lines). Copy this into your project.
- **`requirements.txt`** — Python dependencies (install alongside `azure-ai-agentserver-core`).
- **`examples/`** — Runnable sample applications.

## Quick Start

```python
from agentframework_adapter import from_agent_framework

# Build your Agent Framework agent...
agent = from_agent_framework(my_agent)
agent.run(port=8088)
```

## Examples

| Example | Description |
|---------|-------------|
| `basic_simple/` | Minimal sync agent |
| `simple_async/` | Minimal async agent |
| `mcp_simple/` | MCP server integration |
| `mcp_apikey/` | MCP with API key auth |
| `workflow_agent_simple/` | Multi-step workflow agent |
