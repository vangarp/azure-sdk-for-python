# Copyright (c) Microsoft. All rights reserved.

import anyio
import os
from dotenv import load_dotenv

from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient
from azure.ai.agentserver.claude import from_claude

load_dotenv()


# Define calculator tools
@tool("add", "Adds two numbers together", {"a": int, "b": int})
async def add(args):
    """Adds a and b.
    
    Args:
        a: first int
        b: second int
    """
    result = args["a"] + args["b"]
    return {
        "content": [{
            "type": "text",
            "text": str(result)
        }]
    }


@tool("multiply", "Multiply two numbers together", {"a": int, "b": int})
async def multiply(args):
    """Multiply a and b.
    
    Args:
        a: first int
        b: second int
    """
    result = args["a"] * args["b"]
    return {
        "content": [{
            "type": "text",
            "text": str(result)
        }]
    }


@tool("divide", "Divide two numbers", {"a": int, "b": int})
async def divide(args):
    """Divide a by b.
    
    Args:
        a: first int (numerator)
        b: second int (denominator)
    """
    if args["b"] == 0:
        return {
            "content": [{
                "type": "text",
                "text": "Error: Cannot divide by zero"
            }]
        }
    result = args["a"] / args["b"]
    return {
        "content": [{
            "type": "text",
            "text": str(result)
        }]
    }


async def main() -> None:
    # Set up Claude API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Create MCP server with calculator tools
    tools_server = create_sdk_mcp_server(
        name="calculator-tools",
        version="1.0.0",
        tools=[add, multiply, divide]
    )

    # Create Claude agent with tools
    client = ClaudeSDKClient()
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful calculator assistant. Use the available tools to perform arithmetic operations.",
        mcp_servers={"calculator": tools_server},
        allowed_tools=[
            "mcp__calculator__add",
            "mcp__calculator__multiply",
            "mcp__calculator__divide"
        ]
    )

    # Initialize the agent with an empty prompt
    await client.start(prompt="", options=options)

    # Wrap with Azure AI Agent Server adapter and run
    from_claude(client).run()


if __name__ == "__main__":
    anyio.run(main)
