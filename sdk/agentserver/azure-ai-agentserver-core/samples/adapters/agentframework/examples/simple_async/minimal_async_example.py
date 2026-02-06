# Copyright (c) Microsoft. All rights reserved.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from agentframework_adapter import from_agent_framework  # noqa: E402

import asyncio
from random import randint
from typing import Annotated

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    agent = AzureOpenAIChatClient(credential=DefaultAzureCredential()).create_agent(
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    async with agent:
        await from_agent_framework(agent).run_async()


if __name__ == "__main__":
    asyncio.run(main())
