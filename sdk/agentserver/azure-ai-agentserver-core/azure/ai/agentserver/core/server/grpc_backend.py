# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
"""gRPC backend client for out-of-process adapter communication.

This module implements ``BackendClient`` over gRPC, connecting to an adapter
that implements the ``AgentAdapterService`` defined in
``protos/agent_service.proto``.

The gRPC client:
  - Sends ``AgentRequest`` as a unary message for non-streaming calls.
  - Receives a server-stream of ``AgentStreamEvent`` for streaming calls.
  - Propagates OpenTelemetry trace context via gRPC metadata.
  - Manages the gRPC channel lifecycle.

Usage::

    backend = GrpcBackendClient("localhost:50051")
    response = await backend.run(request)
    # or
    async for event in await backend.run_stream(request):
        ...
    await backend.close()

Dependencies:
    - grpcio
    - grpcio-tools (for proto compilation, build-time only)
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

from ..logger import get_logger
from ..wire.models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ContentPart,
    ErrorInfo,
    InputMessage,
    OutputItem,
    ToolCallInfo,
)
from .backend import BackendClient

logger = get_logger()


class GrpcBackendClient(BackendClient):
    """gRPC-based backend client that communicates with an adapter service.

    :param target: gRPC server address (e.g. ``"localhost:50051"``).
    :type target: str
    :param credentials: Optional gRPC channel credentials for TLS.
    :param options: Optional gRPC channel options.
    :param timeout: Default deadline (seconds) for RPCs. ``None`` for no deadline.
    """

    def __init__(
        self,
        target: str,
        credentials=None,
        options=None,
        timeout: Optional[float] = None,
    ):
        self._target = target
        self._timeout = timeout
        self._channel = None
        self._stub = None
        self._credentials = credentials
        self._options = options

    def _ensure_channel(self):
        """Lazily create the gRPC channel and stub."""
        if self._channel is not None:
            return

        import grpc
        import grpc.aio

        if self._credentials:
            self._channel = grpc.aio.secure_channel(
                self._target, self._credentials, options=self._options
            )
        else:
            self._channel = grpc.aio.insecure_channel(
                self._target, options=self._options
            )

        # Import generated stubs — these live alongside the proto definition
        # or are generated as part of the build process.
        from ..proto import agent_service_pb2_grpc

        self._stub = agent_service_pb2_grpc.AgentAdapterServiceStub(self._channel)
        logger.info(f"gRPC channel opened to {self._target}")

    def _inject_trace_metadata(self) -> list:
        """Inject OpenTelemetry trace context into gRPC metadata."""
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            carrier = {}
            TraceContextTextMapPropagator().inject(carrier)
            return [(k, v) for k, v in carrier.items()]
        except Exception:  # noqa: BLE001
            return []

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a non-streaming agent invocation over gRPC.

        :param request: The wire-format request.
        :type request: AgentRequest
        :return: The wire-format response.
        :rtype: AgentResponse
        """
        self._ensure_channel()
        from ..proto import agent_service_pb2

        pb_request = _request_to_proto(request, agent_service_pb2)
        metadata = self._inject_trace_metadata()

        pb_response = await self._stub.Run(
            pb_request,
            timeout=self._timeout,
            metadata=metadata,
        )
        return _proto_to_response(pb_response)

    async def run_stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
        """Execute a streaming agent invocation over gRPC.

        :param request: The wire-format request.
        :type request: AgentRequest
        :return: An async iterator of wire-format stream events.
        :rtype: AsyncIterator[AgentStreamEvent]
        """
        self._ensure_channel()
        from ..proto import agent_service_pb2

        pb_request = _request_to_proto(request, agent_service_pb2)
        metadata = self._inject_trace_metadata()

        stream = self._stub.RunStream(
            pb_request,
            timeout=self._timeout,
            metadata=metadata,
        )

        async for pb_event in stream:
            yield _proto_to_stream_event(pb_event)

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info(f"gRPC channel closed to {self._target}")


# ---------------------------------------------------------------------------
# Protobuf ↔ Wire dataclass conversion helpers
# ---------------------------------------------------------------------------


def _request_to_proto(request: AgentRequest, pb2):
    """Convert a wire ``AgentRequest`` to its protobuf representation."""
    pb_messages = []
    for msg in request.messages:
        pb_parts = []
        if msg.content_parts:
            for part in msg.content_parts:
                pb_part = pb2.ContentPart(type=part.type)
                if part.text is not None:
                    pb_part.text = part.text
                pb_parts.append(pb_part)

        pb_tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                pb_tool_calls.append(
                    pb2.ToolCallInfo(id=tc.id, name=tc.name, arguments=tc.arguments)
                )

        pb_msg = pb2.InputMessage(
            role=msg.role,
            content=msg.content,
            content_parts=pb_parts,
            tool_calls=pb_tool_calls,
        )
        if msg.tool_call_id is not None:
            pb_msg.tool_call_id = msg.tool_call_id
        pb_messages.append(pb_msg)

    pb_request = pb2.AgentRequest(
        response_id=request.response_id,
        conversation_id=request.conversation_id,
        stream=request.stream,
        messages=pb_messages,
        metadata=request.metadata,
    )
    if request.instructions is not None:
        pb_request.instructions = request.instructions
    if request.agent is not None:
        pb_agent = pb2.AgentInfo()
        if request.agent.name is not None:
            pb_agent.name = request.agent.name
        if request.agent.version is not None:
            pb_agent.version = request.agent.version
        if request.agent.type is not None:
            pb_agent.type = request.agent.type
        pb_request.agent.CopyFrom(pb_agent)

    return pb_request


def _proto_to_response(pb_response) -> AgentResponse:
    """Convert a protobuf ``AgentResponse`` to the wire dataclass."""
    output = [_proto_to_output_item(pb_item) for pb_item in pb_response.output]
    error = None
    if pb_response.HasField("error"):
        error = ErrorInfo(
            code=pb_response.error.code,
            message=pb_response.error.message,
        )
    return AgentResponse(
        status=pb_response.status,
        output=output,
        error=error,
    )


def _proto_to_output_item(pb_item) -> OutputItem:
    """Convert a protobuf ``OutputItem`` to the wire dataclass."""
    content_parts = None
    if pb_item.content_parts:
        content_parts = [
            ContentPart(
                type=p.type,
                text=p.text if p.HasField("text") else None,
            )
            for p in pb_item.content_parts
        ]

    return OutputItem(
        type=pb_item.type,
        role=pb_item.role if pb_item.HasField("role") else None,
        content=pb_item.content if pb_item.HasField("content") else None,
        content_parts=content_parts,
        call_id=pb_item.call_id if pb_item.HasField("call_id") else None,
        name=pb_item.name if pb_item.HasField("name") else None,
        arguments=pb_item.arguments if pb_item.HasField("arguments") else None,
        output=pb_item.output if pb_item.HasField("output") else None,
    )


def _proto_to_stream_event(pb_event) -> AgentStreamEvent:
    """Convert a protobuf ``AgentStreamEvent`` to the wire dataclass."""
    item = None
    if pb_event.HasField("item"):
        item = _proto_to_output_item(pb_event.item)
    response = None
    if pb_event.HasField("response"):
        response = _proto_to_response(pb_event.response)
    error = None
    if pb_event.HasField("error"):
        error = ErrorInfo(
            code=pb_event.error.code,
            message=pb_event.error.message,
        )
    return AgentStreamEvent(
        type=pb_event.type,
        output_index=pb_event.output_index,
        delta=pb_event.delta if pb_event.HasField("delta") else None,
        item=item,
        response=response,
        error=error,
    )
