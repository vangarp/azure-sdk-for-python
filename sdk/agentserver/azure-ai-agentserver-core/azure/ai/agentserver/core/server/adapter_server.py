# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
"""Adapter server helpers for building out-of-process adapter backends.

This module provides base classes that adapter authors use to expose their
framework-specific logic as a gRPC service.  The adapter implements
the abstract ``on_run`` / ``on_run_stream`` methods; the base classes handle
all transport boilerplate.

Example::

    class MyAdapter(GrpcAdapterServer):
        async def on_run(self, request):
            ...
            return AgentResponse(...)

        async def on_run_stream(self, request):
            ...
            yield AgentStreamEvent(...)

    server = MyAdapter()
    await server.serve(port=50051)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterator, Optional

from ..logger import get_logger
from ..wire.models import AgentRequest, AgentResponse, AgentStreamEvent

logger = get_logger()


class AdapterServer(ABC):
    """Abstract interface for adapter backends.

    Adapter authors implement ``on_run`` and ``on_run_stream`` with
    their framework-specific logic.  The transport layer (gRPC) is
    handled by the concrete ``GrpcAdapterServer`` subclass.
    """

    @abstractmethod
    async def on_run(self, request: AgentRequest) -> AgentResponse:
        """Handle a non-streaming agent invocation.

        :param request: The wire-format request from the core server.
        :type request: AgentRequest
        :return: The wire-format response.
        :rtype: AgentResponse
        """
        ...

    @abstractmethod
    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Handle a streaming agent invocation.

        :param request: The wire-format request from the core server.
        :type request: AgentRequest
        :return: An async generator of wire-format stream events.
        :rtype: AsyncGenerator[AgentStreamEvent, None]
        """
        ...

    async def on_startup(self) -> None:
        """Optional hook called when the adapter server starts."""

    async def on_shutdown(self) -> None:
        """Optional hook called when the adapter server stops."""


class GrpcAdapterServer(AdapterServer):
    """gRPC-based adapter server.

    Implements the ``AgentAdapterService`` gRPC interface, delegating to
    the ``on_run`` / ``on_run_stream`` methods.  Start with ``serve()``.

    Usage::

        class MyAdapter(GrpcAdapterServer):
            async def on_run(self, request): ...
            async def on_run_stream(self, request): ...

        await MyAdapter().serve(port=50051)
    """

    async def serve(self, port: int = 50051, credentials=None) -> None:
        """Start the gRPC server.

        :param port: Port to listen on.
        :type port: int
        :param credentials: Optional gRPC server credentials for TLS.
        """
        import grpc
        import grpc.aio

        from ..proto import agent_service_pb2_grpc

        server = grpc.aio.server()
        servicer = _GrpcServicerBridge(self)
        agent_service_pb2_grpc.add_AgentAdapterServiceServicer_to_server(
            servicer, server
        )

        if credentials:
            server.add_secure_port(f"[::]:{port}", credentials)
        else:
            server.add_insecure_port(f"[::]:{port}")

        await self.on_startup()
        logger.info(f"gRPC adapter server starting on port {port}")
        await server.start()

        try:
            await server.wait_for_termination()
        finally:
            await self.on_shutdown()

    async def serve_background(self, port: int = 50051, credentials=None):
        """Start the gRPC server and return the server object for lifecycle management.

        Useful for ``LocalBackend`` or testing scenarios where you need
        programmatic control over the server.

        :param port: Port to listen on.
        :type port: int
        :param credentials: Optional gRPC server credentials for TLS.
        :return: The gRPC server object.
        """
        import grpc
        import grpc.aio

        from ..proto import agent_service_pb2_grpc

        server = grpc.aio.server()
        servicer = _GrpcServicerBridge(self)
        agent_service_pb2_grpc.add_AgentAdapterServiceServicer_to_server(
            servicer, server
        )

        if credentials:
            server.add_secure_port(f"[::]:{port}", credentials)
        else:
            server.add_insecure_port(f"[::]:{port}")

        await self.on_startup()
        logger.info(f"gRPC adapter server starting in background on port {port}")
        await server.start()
        return server


class _GrpcServicerBridge:
    """Bridges the gRPC generated servicer interface to our ``AdapterServer``.

    This is an internal implementation detail — adapters don't interact
    with this class.
    """

    def __init__(self, adapter: AdapterServer):
        self._adapter = adapter

    async def Run(self, request, context):
        """Handle unary Run RPC."""
        from ..proto import agent_service_pb2

        wire_request = _proto_request_to_wire(request)

        try:
            wire_response = await self._adapter.on_run(wire_request)
            return _wire_response_to_proto(wire_response, agent_service_pb2)
        except Exception as e:
            logger.error(f"Error in adapter on_run: {e}")
            import grpc

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return agent_service_pb2.AgentResponse(
                status="failed",
                error=agent_service_pb2.ErrorInfo(code="internal", message=str(e)),
            )

    async def RunStream(self, request, context):
        """Handle server-streaming RunStream RPC."""
        from ..proto import agent_service_pb2

        wire_request = _proto_request_to_wire(request)

        try:
            async for wire_event in self._adapter.on_run_stream(wire_request):
                pb_event = _wire_event_to_proto(wire_event, agent_service_pb2)
                yield pb_event
        except Exception as e:
            logger.error(f"Error in adapter on_run_stream: {e}")
            import grpc

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# ---------------------------------------------------------------------------
# Protobuf ↔ wire dataclass helpers (adapter-server side)
# ---------------------------------------------------------------------------


def _proto_request_to_wire(pb_request) -> AgentRequest:
    """Convert a protobuf ``AgentRequest`` to the wire dataclass."""
    from ..wire.models import AgentInfo, ContentPart, InputMessage, ToolCallInfo

    messages = []
    for pb_msg in pb_request.messages:
        content_parts = None
        if pb_msg.content_parts:
            content_parts = [
                ContentPart(
                    type=p.type,
                    text=p.text if p.HasField("text") else None,
                )
                for p in pb_msg.content_parts
            ]
        tool_calls = None
        if pb_msg.tool_calls:
            tool_calls = [
                ToolCallInfo(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in pb_msg.tool_calls
            ]
        messages.append(
            InputMessage(
                role=pb_msg.role,
                content=pb_msg.content,
                content_parts=content_parts,
                tool_call_id=pb_msg.tool_call_id if pb_msg.HasField("tool_call_id") else None,
                tool_calls=tool_calls,
            )
        )

    agent = None
    if pb_request.HasField("agent"):
        agent = AgentInfo(
            name=pb_request.agent.name if pb_request.agent.HasField("name") else None,
            version=pb_request.agent.version if pb_request.agent.HasField("version") else None,
            type=pb_request.agent.type if pb_request.agent.HasField("type") else None,
        )

    return AgentRequest(
        response_id=pb_request.response_id,
        conversation_id=pb_request.conversation_id,
        stream=pb_request.stream,
        instructions=pb_request.instructions if pb_request.HasField("instructions") else None,
        messages=messages,
        metadata=dict(pb_request.metadata),
        agent=agent,
    )


def _wire_response_to_proto(wire_resp: AgentResponse, pb2):
    """Convert a wire ``AgentResponse`` to its protobuf representation."""
    pb_output = [_wire_item_to_proto(item, pb2) for item in wire_resp.output]
    pb_resp = pb2.AgentResponse(
        status=wire_resp.status,
        output=pb_output,
    )
    if wire_resp.error:
        pb_resp.error.CopyFrom(
            pb2.ErrorInfo(code=wire_resp.error.code, message=wire_resp.error.message)
        )
    return pb_resp


def _wire_item_to_proto(item, pb2):
    """Convert a wire ``OutputItem`` to its protobuf representation."""
    pb_item = pb2.OutputItem(type=item.type)
    if item.role is not None:
        pb_item.role = item.role
    if item.content is not None:
        pb_item.content = item.content
    if item.call_id is not None:
        pb_item.call_id = item.call_id
    if item.name is not None:
        pb_item.name = item.name
    if item.arguments is not None:
        pb_item.arguments = item.arguments
    if item.output is not None:
        pb_item.output = item.output
    if item.content_parts:
        for part in item.content_parts:
            pb_part = pb2.ContentPart(type=part.type)
            if part.text is not None:
                pb_part.text = part.text
            pb_item.content_parts.append(pb_part)
    return pb_item


def _wire_event_to_proto(event: AgentStreamEvent, pb2):
    """Convert a wire ``AgentStreamEvent`` to its protobuf representation."""
    pb_event = pb2.AgentStreamEvent(
        type=event.type,
        output_index=event.output_index,
    )
    if event.delta is not None:
        pb_event.delta = event.delta
    if event.item is not None:
        pb_event.item.CopyFrom(_wire_item_to_proto(event.item, pb2))
    if event.response is not None:
        pb_event.response.CopyFrom(_wire_response_to_proto(event.response, pb2))
    if event.error is not None:
        pb_event.error.CopyFrom(
            pb2.ErrorInfo(code=event.error.code, message=event.error.message)
        )
    return pb_event
