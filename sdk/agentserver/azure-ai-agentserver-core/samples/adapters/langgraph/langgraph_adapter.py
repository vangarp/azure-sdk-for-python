# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation,broad-exception-caught
# mypy: disable-error-code="assignment,arg-type"
"""Self-contained LangGraph wire-protocol adapter.

Copy this file into your project to serve a LangGraph agent via the
Foundry Agent Server wire protocol.  The only install requirement beyond
LangGraph itself is ``azure-ai-agentserver-core``.

Quick start (in-process)::

    from langgraph_adapter import from_langgraph

    agent = from_langgraph(my_compiled_graph)
    agent.run(port=8088)

Out-of-process::

    from langgraph_adapter import LangGraphWireAdapter

    adapter = LangGraphWireAdapter(graph=my_graph)
    await adapter.serve(port=50051)      # gRPC endpoint

See ``examples/`` for complete runnable samples.
"""
from __future__ import annotations

import json
import os
import re
from typing import AsyncGenerator, Optional, get_type_hints

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

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
from azure.ai.agentserver.core.wire.state_converter import WireStateConverter

# ---------------------------------------------------------------------------
# Lazy logger
# ---------------------------------------------------------------------------

_logger = None


def _get_logger():
    global _logger  # pylint: disable=global-statement
    if _logger is None:
        import logging

        _logger = logging.getLogger("langgraph_adapter")
    return _logger


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def extract_function_call(tool_call: dict):
    """Extract (name, call_id, arguments_json) from a LangChain tool_call dict."""
    name = tool_call.get("name")
    call_id = tool_call.get("id")
    argument = None
    arguments_raw = tool_call.get("args")
    if isinstance(arguments_raw, str):
        argument = arguments_raw
    elif isinstance(arguments_raw, dict):
        argument = json.dumps(arguments_raw)
    return name, call_id, argument


def is_state_schema_valid(state_schema) -> bool:
    """Return ``True`` if the schema has a ``messages`` field (i.e.
    it is a ``MessagesState`` compatible schema)."""
    fields = _get_typeddict_fields(state_schema)
    return "messages" in fields


def _get_typeddict_fields(schema_class) -> dict:
    try:
        return get_type_hints(schema_class)
    except (TypeError, AttributeError):
        if hasattr(schema_class, "__annotations__"):
            return schema_class.__annotations__
    return {}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LangGraphWireAdapter(GrpcAdapterServer):
    """Wire-protocol adapter for LangGraph agents.

    :param graph: A compiled LangGraph state graph.
    :param state_converter: Optional ``WireStateConverter`` for non-MessagesState
        graphs.  When ``None``, the adapter uses built-in MessagesState handling.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        state_converter: Optional[WireStateConverter] = None,
    ):
        self.graph = graph
        self.azure_ai_tracer = None
        self._state_converter = state_converter

        if not state_converter and not is_state_schema_valid(self.graph.builder.state_schema):
            raise ValueError(
                "state_converter is required for non-MessagesState graphs."
            )

    async def on_startup(self) -> None:
        self._init_tracing()

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        logger = _get_logger()
        try:
            if self._state_converter:
                input_data = self._state_converter.request_to_state(request)
                config = self._create_config(request)
                stream_mode = self._state_converter.get_stream_mode(request)
                result = await self.graph.ainvoke(input_data, config=config, stream_mode=stream_mode)
                return self._state_converter.state_to_response(result)
            else:
                input_data = _wire_to_langgraph_input(request)
                config = self._create_config(request)
                result = await self.graph.ainvoke(input_data, config=config, stream_mode="updates")
                output_items = _langgraph_result_to_wire(result)
                return AgentResponse(status="completed", output=output_items)
        except Exception as e:
            logger.error(f"Error in on_run: {e}")
            return AgentResponse(
                status="failed",
                error=ErrorInfo(code="internal", message=str(e)),
            )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        logger = _get_logger()
        try:
            if self._state_converter:
                input_data = self._state_converter.request_to_state(request)
                config = self._create_config(request)
                stream_mode = self._state_converter.get_stream_mode(request)
                stream = self.graph.astream(
                    input=input_data, config=config, stream_mode=stream_mode
                )
                async for event in self._state_converter.state_to_stream(stream, request):
                    yield event
            else:
                input_data = _wire_to_langgraph_input(request)
                config = self._create_config(request)
                output_index = 0
                stream = self.graph.astream(
                    input=input_data, config=config, stream_mode="messages"
                )
                async for message, metadata in stream:
                    events = _langgraph_message_to_wire_events(
                        message, metadata, output_index
                    )
                    for event in events:
                        yield event
                        if event.type == StreamEventType.OUTPUT_ITEM_DONE:
                            output_index += 1

                yield AgentStreamEvent(type=StreamEventType.COMPLETED, output_index=0)

        except Exception as e:
            logger.error(f"Error in on_run_stream: {e}")
            yield AgentStreamEvent(
                type=StreamEventType.ERROR,
                error=ErrorInfo(code="internal", message=str(e)),
            )

    def _create_config(self, request: AgentRequest) -> RunnableConfig:
        config = RunnableConfig(
            configurable={"thread_id": request.conversation_id},
            callbacks=[self.azure_ai_tracer] if self.azure_ai_tracer else None,
        )
        return config

    def _init_tracing(self):
        logger = _get_logger()
        app_insights_conn_str = os.environ.get(
            "_AGENT_RUNTIME_APP_INSIGHTS_CONNECTION_STRING"
        )
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_OTEL_ONLY"] = "true"

        if app_insights_conn_str:
            try:
                from langchain_azure_ai.callbacks.tracers import (
                    AzureAIOpenTelemetryTracer,
                )

                self.azure_ai_tracer = AzureAIOpenTelemetryTracer(
                    connection_string=app_insights_conn_str,
                    enable_content_recording=True,
                    name=self._get_agent_identifier(),
                )
                logger.info("AzureAIOpenTelemetryTracer initialized.")
            except Exception as e:
                logger.error(f"Failed to init AzureAIOpenTelemetryTracer: {e}")

    def _get_agent_identifier(self) -> str:
        agent_name = os.getenv("AGENT_NAME")
        if agent_name:
            return agent_name
        agent_id = os.getenv("AGENT_ID")
        if agent_id:
            return agent_id
        return "HostedAgent-LangGraph"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def from_langgraph(agent, state_converter: Optional[WireStateConverter] = None):
    """Create a ``FoundryCBAgent`` backed by a LangGraph graph.

    :param agent: A compiled LangGraph state graph.
    :param state_converter: Optional ``WireStateConverter`` for non-MessagesState graphs.
    :return: A ``FoundryCBAgent`` instance ready to ``.run()``.
    """
    from azure.ai.agentserver.core import FoundryCBAgent, LocalBackend

    adapter = LangGraphWireAdapter(graph=agent, state_converter=state_converter)
    return FoundryCBAgent(backend=LocalBackend(adapter))


# ---------------------------------------------------------------------------
# Wire ↔ LangChain conversion helpers
# ---------------------------------------------------------------------------


def _wire_to_langgraph_input(request: AgentRequest) -> dict:
    """Convert a wire ``AgentRequest`` into LangGraph ``{"messages": [...]}``."""
    messages: list[AnyMessage] = []
    if request.instructions:
        messages.append(SystemMessage(content=request.instructions))
    for msg in request.messages:
        lc_msg = _wire_message_to_langchain(msg)
        if lc_msg is not None:
            messages.append(lc_msg)
    return {"messages": messages}


def _wire_message_to_langchain(msg) -> Optional[AnyMessage]:
    role = msg.role
    if role == "user":
        return HumanMessage(content=msg.content)
    if role == "system":
        return SystemMessage(content=msg.content)
    if role == "assistant":
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    args=json.loads(tc.arguments) if tc.arguments else {},
                )
                for tc in msg.tool_calls
            ]
            return AIMessage(tool_calls=tool_calls, content=msg.content or "")
        return AIMessage(content=msg.content)
    if role == "tool":
        return ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id or "")
    _get_logger().warning(f"Unknown message role: {role}")
    return HumanMessage(content=msg.content)


def _langgraph_result_to_wire(result) -> list[OutputItem]:
    """Convert non-streaming ``ainvoke(stream_mode="updates")`` result to wire."""
    items: list[OutputItem] = []
    for step in result:
        for _node_name, node_output in step.items():
            message_arr = node_output.get("messages")
            if not message_arr:
                continue
            for message in message_arr:
                item = _langchain_message_to_wire(message)
                if item is not None:
                    items.append(item)
    return items


def _langchain_message_to_wire(message: AnyMessage) -> Optional[OutputItem]:
    if isinstance(message, HumanMessage):
        return OutputItem(
            type=OutputItemType.TEXT_MESSAGE, role="user",
            content=_extract_text_content(message.content),
        )
    if isinstance(message, SystemMessage):
        return OutputItem(
            type=OutputItemType.TEXT_MESSAGE, role="system",
            content=_extract_text_content(message.content),
        )
    if isinstance(message, AIMessage):
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name, call_id, arguments = extract_function_call(tool_call)
            return OutputItem(
                type=OutputItemType.FUNCTION_CALL,
                call_id=call_id or "", name=name or "", arguments=arguments or "",
            )
        return OutputItem(
            type=OutputItemType.TEXT_MESSAGE, role="assistant",
            content=_extract_text_content(message.content),
        )
    if isinstance(message, ToolMessage):
        return OutputItem(
            type=OutputItemType.FUNCTION_CALL_OUTPUT,
            call_id=message.tool_call_id or "",
            output=_extract_text_content(message.content),
        )
    return None


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return " ".join(texts)
    return str(content)


def _langgraph_message_to_wire_events(
    message: AnyMessage, metadata: dict, output_index: int
) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []
    item = _langchain_message_to_wire(message)
    if item is None:
        return events

    if item.type == OutputItemType.TEXT_MESSAGE and item.role == "assistant":
        text = item.content or ""
        if text:
            events.append(
                AgentStreamEvent(
                    type=StreamEventType.OUTPUT_ITEM_ADDED,
                    output_index=output_index,
                    item=OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""),
                )
            )
            events.append(
                AgentStreamEvent(type=StreamEventType.TEXT_DELTA, output_index=output_index, delta=text)
            )
            events.append(
                AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
            )

    elif item.type == OutputItemType.FUNCTION_CALL:
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_ADDED, output_index=output_index, item=item)
        )
        if item.arguments:
            events.append(
                AgentStreamEvent(
                    type=StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA,
                    output_index=output_index, delta=item.arguments,
                )
            )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
        )

    elif item.type == OutputItemType.FUNCTION_CALL_OUTPUT:
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_ADDED, output_index=output_index, item=item)
        )
        events.append(
            AgentStreamEvent(type=StreamEventType.OUTPUT_ITEM_DONE, output_index=output_index, item=item)
        )

    return events
