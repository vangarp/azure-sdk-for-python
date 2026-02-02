# Copyright (c) Microsoft. All rights reserved.

import anyio
import os
from dotenv import load_dotenv

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from azure.ai.agentserver.claude import from_claude

load_dotenv()


async def main() -> None:
    # Set up Claude API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Create Claude agent
    client = ClaudeSDKClient()
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant that provides concise and accurate information."
    )

    # Initialize the agent with an empty prompt
    await client.start(prompt="", options=options)

    # Wrap with Azure AI Agent Server adapter and run
    from_claude(client).run()


if __name__ == "__main__":
    anyio.run(main)
