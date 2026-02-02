# Claude Agent Sample

This sample demonstrates how to use the agents hosting adapter with Claude SDK.

## Prerequisites

### Environment Variables

Copy `.envtemplate` to `.env` and supply:

```
ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

You can get your Anthropic API key from [Anthropic Console](https://console.anthropic.com/).

## Running the Sample

Follow these steps from this folder:

1) Start the agent server (defaults to 0.0.0.0:8088):

```bash
python main.py
```

2) Send a non-streaming request (returns a single JSON response):

```bash
curl -sS \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d "{\"input\":\"What is the capital of France?\",\"stream\":false}"
```

3) Send a streaming request (server-sent events). Use -N to disable curl buffering:

```bash
curl -N \
  -H "Content-Type: application/json" \
  -X POST http://localhost:8088/responses \
  -d "{\"input\":\"Tell me a short story about a robot.\",\"stream\":true}"
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Make sure you've set the `ANTHROPIC_API_KEY` in your `.env` file
2. **Port Already in Use**: If port 8088 is already in use, you can change it in the code by passing `port=<new_port>` to the `run()` method
3. **Connection Errors**: Ensure you have internet connectivity to reach the Anthropic API

## Next Steps

- Check out other samples in the `samples` directory
- Read the [Claude SDK documentation](https://platform.claude.com/docs/en/agent-sdk/python)
- Explore the [Azure AI Agent Server documentation](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/agentserver)
