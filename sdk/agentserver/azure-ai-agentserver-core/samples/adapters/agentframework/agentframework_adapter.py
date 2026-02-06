# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
"""Self-contained Agent Framework wire-protocol adapter.

Copy this file into your project to serve an Agent Framework agent via
the Foundry Agent Server wire protocol.  The only install requirement
beyond ``agent-framework`` itself is ``azure-ai-agentserver-core``.

Quick start (in-process)::

    from agentframework_adapter import from_agent_framework

    agent = from_agent_framework(my_agent)
    agent.run(port=8088)

See ``examples/`` for complete runnable samples.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

from agent_framework import AgentProtocol, ChatMessage, Role as ChatRole
from agent_framework._types import (
    ErrorContent,
    FunctionCallContent,
    FunctionResultContent,
    TextContent,
)

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

# ---------------------------------------------------------------------------
# Lazy logger
# ---------------------------------------------------------------------------

_logger = None


def _get_logger():
    global _logger  # pylint: disable=global-statement
    if _logger is None:
        import logging

        _logger = logging.getLogger("agentframework_adapter")
    return _logger


_DEFAULT_STREAM_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AgentFrameworkWireAdapter(GrpcAdapterServer):
    """Wire-protocol adapter for Agent Framework agents.

    :param agent: An Agent Framework ``AgentProtocol`` instance.
    """

    def __init__(self, agent: AgentProtocol):
        self.agent = agent

    async def on_startup(self) -> None:
        self._init_tracing()

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        logger = _get_logger()
        try:
            message = _wire_to_agent_framework_input(request)
            result = await self.agent.run(message)
            output_items = _agent_response_to_wire(result)
            return AgentResponse(status="completed", output=output_items)
        except Exception as e:
            logger.error(f"Error in on_run: {e}")
            return AgentResponse(
                status="failed",
                error=ErrorInfo(code="agent_framework_error", message=str(e)),
            )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        logger = _get_logger()
        try:
            message = _wire_to_agent_framework_input(request)
            timeout_s = _resolve_stream_timeout(request)
            output_index = 0

            aiter = self.agent.run_stream(message).__aiter__()
            while True:
                try:
                    update = await asyncio.wait_for(aiter.__anext__(), timeout=timeout_s)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("Streaming idle timeout (%.1fs); terminating.", timeout_s)
                    break

                contents = getattr(update, "contents", None)
                if not contents:
                    continue

                for content in contents:
                    events = _content_to_wire_events(content, output_index)
                    for event in events:
                        yield event
                        if event.type == StreamEventType.OUTPUT_ITEM_DONE:
                            output_index += 1

            yield AgentStreamEvent(type=StreamEventType.COMPLETED, output_index=0)

        except Exception as e:
            logger.error(f"Error in on_run_stream: {e}")
            yield AgentStreamEvent(
                type=StreamEventType.ERROR,
                error=ErrorInfo(code="agent_framework_error", message=str(e)),
            )

    def _init_tracing(self):
        logger = _get_logger()
        app_insights_conn_str = os.environ.get(
            "_AGENT_RUNTIME_APP_INSIGHTS_CONNECTION_STRING"
        )
        project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

        if project_endpoint:
            try:
                from agent_framework.azure import AzureAIAgentClient
                from azure.ai.projects import AIProjectClient
                from azure.identity import DefaultAzureCredential

                project_client = AIProjectClient(
                    endpoint=project_endpoint,
                    credential=DefaultAzureCredential(),
                )
                agent_client = AzureAIAgentClient(project_client=project_client)
                agent_client.setup_azure_ai_observability()
                logger.info("AzureAI observability initialized via project endpoint.")
            except Exception as e:
                logger.error(f"Failed to setup AzureAI observability: {e}")
        elif app_insights_conn_str:
            try:
                os.environ["WORKFLOW_ENABLE_OTEL"] = "true"
                from agent_framework.observability import setup_observability

                otel_endpoint = os.environ.get("OTEL_EXPORTER_ENDPOINT")
                setup_observability(
                    enable_sensitive_data=True,
                    otlp_endpoint=otel_endpoint,
                    applicationinsights_connection_string=app_insights_conn_str,
                )
                logger.info("Agent Framework observability initialized.")
            except Exception as e:
                logger.error(f"Failed to setup observability: {e}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def from_agent_framework(agent):
    """Create a ``FoundryCBAgent`` backed by an Agent Framework agent.

    :param agent: An Agent Framework ``AgentProtocol`` instance.
    :return: A ``FoundryCBAgent`` instance ready to ``.run()``.
    """
    from azure.ai.agentserver.core import FoundryCBAgent, LocalBackend

    adapter = AgentFrameworkWireAdapter(agent=agent)
    return FoundryCBAgent(backend=LocalBackend(adapter))


# ---------------------------------------------------------------------------
# Wire ↔ Agent Framework conversion helpers
# ---------------------------------------------------------------------------


def _wire_to_agent_framework_input(
    request: AgentRequest,
) -> str | ChatMessage | list[str] | list[ChatMessage] | None:
    if not request.messages:
        return None

    messages: list[ChatMessage] = []
    role_map = {
        "user": ChatRole.USER,
        "assistant": ChatRole.ASSISTANT,
        "system": ChatRole.SYSTEM,
    }

    if request.instructions:
        messages.append(ChatMessage(role=ChatRole.SYSTEM, text=request.instructions))

    for msg in request.messages:
        role = role_map.get(msg.role, ChatRole.USER)
        messages.append(ChatMessage(role=role, text=msg.content))

    if len(messages) == 1 and messages[0].role == ChatRole.USER:
        return messages[0].text

    return messages


def _agent_response_to_wire(result) -> list[OutputItem]:
    items: list[OutputItem] = []
    for message in result.messages:
        contents = getattr(message, "contents", None)
        if not contents:
            continue
        for content in contents:
            item = _content_to_wire_item(content)
            if item is not None:
                items.append(item)
    return items


def _content_to_wire_item(content) -> OutputItem | None:
    if isinstance(content, TextContent):
        text = getattr(content, "text", None)
        if not text:
            return None
        return OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=text)

    if isinstance(content, FunctionCallContent):
        name = getattr(content, "name", "") or ""
        arguments = getattr(content, "arguments", "")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except Exception:
                arguments = str(arguments)
        call_id = getattr(content, "call_id", None) or ""
        return OutputItem(
            type=OutputItemType.FUNCTION_CALL,
            call_id=call_id, name=name, arguments=arguments or "",
        )

    if isinstance(content, FunctionResultContent):
        raw = getattr(content, "result", None)
        output_str = ""
        if isinstance(raw, str):
            output_str = raw
        elif isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, TextContent):
                    parts.append(getattr(item, "text", ""))
                else:
                    parts.append(str(item))
            output_str = json.dumps(parts) if parts else ""
        call_id = getattr(content, "call_id", None) or ""
        return OutputItem(
            type=OutputItemType.FUNCTION_CALL_OUTPUT,
            call_id=call_id, output=output_str,
        )

    return None


def _content_to_wire_events(content, output_index: int) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []

    if isinstance(content, TextContent):
        text = getattr(content, "text", None) or ""
        if not text:
            return events
        item = OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=text)
        events.append(
            AgentStreamEvent(
                type=StreamEventType.OUTPUT_ITEM_ADDED, output_index=output_index,
                item=OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""),
            )
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.TEXT_DELTA, output_index=output_index, delta=text)
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
        )

    elif isinstance(content, FunctionCallContent):
        name = getattr(content, "name", "") or ""
        arguments = getattr(content, "arguments", "")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except Exception:
                arguments = str(arguments)
        call_id = getattr(content, "call_id", None) or ""
        item = OutputItem(
            type=OutputItemType.FUNCTION_CALL,
            call_id=call_id, name=name, arguments=arguments or "",
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_ADDED, output_index=output_index, item=item)
        )
        if arguments:
            events.append(
                AgentStreamEvent(
                    type=StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA,
                    output_index=output_index, delta=arguments,
                )
            )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
        )

    elif isinstance(content, FunctionResultContent):
        raw = getattr(content, "result", None)
        output_str = ""
        if isinstance(raw, str):
            output_str = raw
        elif isinstance(raw, list):
            parts = []
            for item_val in raw:
                if isinstance(item_val, str):
                    parts.append(item_val)
                elif isinstance(item_val, TextContent):
                    parts.append(getattr(item_val, "text", ""))
                else:
                    parts.append(str(item_val))
            output_str = json.dumps(parts) if parts else ""
        call_id = getattr(content, "call_id", None) or ""
        item = OutputItem(
            type=OutputItemType.FUNCTION_CALL_OUTPUT,
            call_id=call_id, output=output_str,
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_ADDED, output_index=output_index, item=item)
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
        )

    elif isinstance(content, ErrorContent):
        events.append(
            AgentStreamEvent(
                type=StreamEventType.ERROR,
                error=ErrorInfo(
                    code=getattr(content, "error_code", None) or "server_error",
                    message=getattr(content, "message", None) or "An error occurred",
                ),
            )
        )

    return events


def _resolve_stream_timeout(request: AgentRequest) -> float:
    env_val = os.getenv("AGENTS_ADAPTER_STREAM_TIMEOUT_S")
    return float(env_val) if env_val is not None else _DEFAULT_STREAM_TIMEOUT_S
