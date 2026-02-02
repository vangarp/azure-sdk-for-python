# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, Union

from opentelemetry import trace

from azure.ai.agentserver.core import AgentRunContext, FoundryCBAgent
from azure.ai.agentserver.core.constants import Constants as AdapterConstants
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import (
    Response as OpenAIResponse,
    ResponseStreamEvent,
)

from .models.claude_input_converter import ClaudeInputConverter
from .models.claude_output_non_streaming_converter import ClaudeOutputNonStreamingConverter
from .models.claude_output_streaming_converter import ClaudeOutputStreamingConverter

logger = get_logger()


class ClaudeAdapter(FoundryCBAgent):
    """
    Adapter class for integrating Claude SDK agents with the FoundryCB agent interface.

    This class wraps a Claude SDK client and provides a unified interface
    for running agents in both streaming and non-streaming modes. It handles input and output
    conversion between the Claude SDK and the expected formats for FoundryCB agents.

    Parameters:
        agent: A Claude SDK client instance to be adapted.

    Usage:
        - Instantiate with a Claude SDK client.
        - Call `agent_run` with an `AgentRunContext` to execute the agent.
        - Supports both streaming and non-streaming responses based on the `stream` flag.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        logger.info(f"Initialized ClaudeAdapter with agent: {type(agent).__name__}")

    def init_tracing(self):
        exporter = os.environ.get(AdapterConstants.OTEL_EXPORTER_ENDPOINT)
        app_insights_conn_str = os.environ.get(AdapterConstants.APPLICATION_INSIGHTS_CONNECTION_STRING)

        if exporter or app_insights_conn_str:
            # Set up OpenTelemetry tracing for Claude
            os.environ["OTEL_ENABLED"] = "true"
            logger.info("OpenTelemetry tracing enabled for Claude adapter")

        self.tracer = trace.get_tracer(__name__)

    async def agent_run(
        self, context: AgentRunContext
    ) -> Union[
        OpenAIResponse,
        AsyncGenerator[ResponseStreamEvent, Any],
    ]:
        logger.info(f"Starting agent_run with stream={context.stream}")
        request_input = context.request.get("input")

        input_converter = ClaudeInputConverter()
        message = input_converter.transform_input(request_input)
        logger.debug(f"Transformed input message: {message}")

        # Use split converters
        if context.stream:
            logger.info("Running agent in streaming mode")
            streaming_converter = ClaudeOutputStreamingConverter(context)

            async def stream_updates():
                update_count = 0
                logger.info("Starting streaming")
                for ev in streaming_converter.initial_events():
                    yield ev

                # Send the message to the Claude agent
                await self.agent.send(message)

                # Stream responses
                async for update in self.agent.receive_response():
                    update_count += 1
                    transformed = streaming_converter.transform_output_for_streaming(update)
                    for event in transformed:
                        yield event

                for ev in streaming_converter.completion_events():
                    yield ev
                logger.info("Streaming completed with %d updates", update_count)

            return stream_updates()

        # Non-streaming path
        logger.info("Running agent in non-streaming mode")
        non_streaming_converter = ClaudeOutputNonStreamingConverter(context)

        # Send the message and collect the response
        await self.agent.send(message)
        full_response = []
        async for response_chunk in self.agent.receive_response():
            full_response.append(response_chunk)

        logger.debug(f"Agent run completed, response chunks: {len(full_response)}")
        transformed_result = non_streaming_converter.transform_output_for_response(full_response)
        logger.info("Agent run and transformation completed successfully")
        return transformed_result
