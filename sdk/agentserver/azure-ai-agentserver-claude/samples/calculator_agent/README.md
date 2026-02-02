# Claude Calculator Agent Sample

This sample demonstrates how to create a calculator agent using the Claude Agent SDK with tool calling capabilities and hosting it with the Azure AI Agent Server adapter. The agent can perform basic arithmetic operations (addition, multiplication, and division) by utilizing tools and making decisions about when to use them.

## Overview

The sample consists of several key components:

- **Claude Agent SDK**: A calculator agent that uses tools to perform arithmetic operations
- **Azure AI Agent Server Adapter**: Wraps the Claude agent and hosts it as a service on your local machine

## Files Description

- `main.py` - The main implementation with calculator tools and agent setup
- `.envtemplate` - A template for environment variables (Anthropic API configuration)
- `requirements.txt` - Python dependencies for the sample

## Setup

### Prerequisites

You need an Anthropic API key to use Claude. Get one from [Anthropic Console](https://console.anthropic.com/).

### Environment Configuration

1. Create a `.env` file in this directory based on `.envtemplate`:
   ```bash
   cp .envtemplate .env
   ```

2. Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or if developing locally with the packages:

```bash
# Install core and adapter from local source
pip install -e ../../azure-ai-agentserver-core
pip install -e ../..
pip install -r requirements.txt
```

## Usage

### Running as HTTP Server

1. Start the agent server:
   ```bash
   python main.py
   ```
   The server will start on `http://localhost:8088`

2. Test the agent with arithmetic operations:

   **Addition example:**
   ```bash
   curl -X POST http://localhost:8088/responses \
     -H "Content-Type: application/json" \
     -d '{
       "input": "What is 15 plus 27?",
       "stream": false
     }'
   ```

   **Multiplication example:**
   ```bash
   curl -X POST http://localhost:8088/responses \
     -H "Content-Type: application/json" \
     -d '{
       "input": "What is 8 times 7?",
       "stream": false
     }'
   ```

   **Division example:**
   ```bash
   curl -X POST http://localhost:8088/responses \
     -H "Content-Type: application/json" \
     -d '{
       "input": "What is 100 divided by 4?",
       "stream": false
     }'
   ```

   **Streaming example:**
   ```bash
   curl -N \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8088/responses \
     -d '{
       "input": "Calculate 12 times 5, then add 10",
       "stream": true
     }'
   ```

## How It Works

1. **Tool Definition**: Three calculator tools are defined using the `@tool` decorator:
   - `add`: Adds two numbers
   - `multiply`: Multiplies two numbers
   - `divide`: Divides two numbers (with zero-check)

2. **MCP Server**: The tools are registered in an MCP (Model Context Protocol) server that makes them available to Claude.

3. **Claude Agent**: A `ClaudeSDKClient` is configured with:
   - System prompt defining the agent as a calculator assistant
   - The MCP server containing the calculator tools
   - Allowed tools list specifying which tools Claude can use

4. **Azure AI Agent Server Adapter**: The `from_claude()` function wraps the Claude agent, making it compatible with the Azure AI Agent Server framework and exposing it via an HTTP API.

## Architecture

```
User Request (HTTP)
    ↓
Azure AI Agent Server (localhost:8088)
    ↓
Claude Agent (with tools)
    ↓
Calculator Tools (add, multiply, divide)
    ↓
Response (HTTP)
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Make sure you've set the `ANTHROPIC_API_KEY` in your `.env` file
2. **Port Already in Use**: If port 8088 is already in use, you can change it in the code by passing `port=<new_port>` to the `run()` method
3. **Connection Errors**: Ensure you have internet connectivity to reach the Anthropic API
4. **Tool Not Found**: If Claude doesn't use the tools, verify that the `allowed_tools` list matches the MCP server name and tool names

### Debug Mode

To see more detailed logs, you can set the logging level in the code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

- Explore adding more complex calculator operations
- Try combining multiple operations in a single request
- Experiment with natural language queries (e.g., "What's 5% of 200?")
- Check out the [Claude Agent SDK documentation](https://github.com/anthropics/claude-agent-sdk-python) for more advanced features
- Explore other samples in the `samples` directory
