# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Integration tests for the LocalBackend + GrpcAdapterServer round-trip.

These tests validate the end-to-end wire protocol flow:
  AgentRequest → protobuf → gRPC loopback → adapter.on_run → protobuf → AgentResponse

No external services or LLMs are needed — the adapter is a simple mock
that echoes input or produces fixed output.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from azure.ai.agentserver.core.server.adapter_server import GrpcAdapterServer
from azure.ai.agentserver.core.server.local_backend import LocalBackend
from azure.ai.agentserver.core.wire.models import (
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    ErrorInfo,
    InputMessage,
    OutputItem,
    OutputItemType,
    StreamEventType,
)


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class EchoAdapter(GrpcAdapterServer):
    """A trivial adapter that echoes the user's last message back."""

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        # Find last user message
        user_text = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_text = msg.content
                break

        return AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content=f"Echo: {user_text}",
                ),
            ],
        )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        user_text = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_text = msg.content
                break

        reply = f"Echo: {user_text}"
        item = OutputItem(
            type=OutputItemType.TEXT_MESSAGE,
            role="assistant",
            content=reply,
        )
        yield AgentStreamEvent(
            type=StreamEventType.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""),
        )
        yield AgentStreamEvent(
            type=StreamEventType.TEXT_DELTA,
            output_index=0,
            delta=reply,
        )
        yield AgentStreamEvent(
            type=StreamEventType.OUTPUT_ITEM_DONE,
            output_index=0,
            item=item,
        )
        yield AgentStreamEvent(
            type=StreamEventType.COMPLETED,
            output_index=0,
        )


class FunctionCallAdapter(GrpcAdapterServer):
    """Adapter that returns a function call output item."""

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.FUNCTION_CALL,
                    call_id="call_test_123",
                    name="get_weather",
                    arguments='{"city": "Seattle"}',
                ),
            ],
        )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        item = OutputItem(
            type=OutputItemType.FUNCTION_CALL,
            call_id="call_test_123",
            name="get_weather",
            arguments='{"city": "Seattle"}',
        )
        yield AgentStreamEvent(
            type=StreamEventType.OUTPUT_ITEM_ADDED,
            output_index=0,
            item=item,
        )
        yield AgentStreamEvent(
            type=StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA,
            output_index=0,
            delta='{"city": "Seattle"}',
        )
        yield AgentStreamEvent(
            type=StreamEventType.OUTPUT_ITEM_DONE,
            output_index=0,
            item=item,
        )
        yield AgentStreamEvent(
            type=StreamEventType.COMPLETED,
            output_index=0,
        )


class ErrorAdapter(GrpcAdapterServer):
    """Adapter that always returns an error."""

    async def on_run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(code="test_error", message="Something went wrong"),
        )

    async def on_run_stream(
        self, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        yield AgentStreamEvent(
            type=StreamEventType.ERROR,
            error=ErrorInfo(code="test_error", message="Stream failed"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(content: str = "Hello", stream: bool = False) -> AgentRequest:
    return AgentRequest(
        response_id="resp_test",
        conversation_id="conv_test",
        stream=stream,
        messages=[InputMessage(role="user", content=content)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalBackendEcho:
    """Test the full LocalBackend round-trip with EchoAdapter."""

    @pytest.mark.asyncio
    async def test_non_streaming(self):
        backend = LocalBackend(EchoAdapter())
        try:
            response = await backend.run(_make_request("Testing 123"))
            assert response.status == "completed"
            assert len(response.output) == 1
            assert response.output[0].content == "Echo: Testing 123"
            assert response.output[0].role == "assistant"
            assert response.output[0].type == OutputItemType.TEXT_MESSAGE
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_streaming(self):
        backend = LocalBackend(EchoAdapter())
        try:
            events = []
            async for event in backend.run_stream(_make_request("Stream test", stream=True)):
                events.append(event)

            assert len(events) == 4
            assert events[0].type == StreamEventType.OUTPUT_ITEM_ADDED
            assert events[1].type == StreamEventType.TEXT_DELTA
            assert events[1].delta == "Echo: Stream test"
            assert events[2].type == StreamEventType.OUTPUT_ITEM_DONE
            assert events[2].item.content == "Echo: Stream test"
            assert events[3].type == StreamEventType.COMPLETED
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_multiple_requests(self):
        """Verify the backend can handle sequential requests."""
        backend = LocalBackend(EchoAdapter())
        try:
            r1 = await backend.run(_make_request("First"))
            r2 = await backend.run(_make_request("Second"))
            assert r1.output[0].content == "Echo: First"
            assert r2.output[0].content == "Echo: Second"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        backend = LocalBackend(EchoAdapter())
        try:
            response = await backend.run(
                AgentRequest(
                    response_id="r1",
                    conversation_id="c1",
                    messages=[],
                )
            )
            assert response.status == "completed"
            # Echo of empty input
            assert response.output[0].content == "Echo: "
        finally:
            await backend.close()


class TestLocalBackendFunctionCall:
    """Test function call round-trip through LocalBackend."""

    @pytest.mark.asyncio
    async def test_non_streaming_function_call(self):
        backend = LocalBackend(FunctionCallAdapter())
        try:
            response = await backend.run(_make_request("Call a function"))
            assert response.status == "completed"
            assert len(response.output) == 1
            item = response.output[0]
            assert item.type == OutputItemType.FUNCTION_CALL
            assert item.call_id == "call_test_123"
            assert item.name == "get_weather"
            assert item.arguments == '{"city": "Seattle"}'
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_streaming_function_call(self):
        backend = LocalBackend(FunctionCallAdapter())
        try:
            events = []
            async for event in backend.run_stream(_make_request("Call", stream=True)):
                events.append(event)

            assert len(events) == 4
            assert events[0].type == StreamEventType.OUTPUT_ITEM_ADDED
            assert events[0].item.type == OutputItemType.FUNCTION_CALL
            assert events[1].type == StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA
            assert events[1].delta == '{"city": "Seattle"}'
            assert events[2].type == StreamEventType.OUTPUT_ITEM_DONE
            assert events[3].type == StreamEventType.COMPLETED
        finally:
            await backend.close()


class TestLocalBackendError:
    """Test error handling through LocalBackend."""

    @pytest.mark.asyncio
    async def test_non_streaming_error(self):
        backend = LocalBackend(ErrorAdapter())
        try:
            response = await backend.run(_make_request("Fail"))
            assert response.status == "failed"
            assert response.error is not None
            assert response.error.code == "test_error"
            assert response.error.message == "Something went wrong"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_streaming_error(self):
        backend = LocalBackend(ErrorAdapter())
        try:
            events = []
            async for event in backend.run_stream(_make_request("Fail", stream=True)):
                events.append(event)

            assert len(events) == 1
            assert events[0].type == StreamEventType.ERROR
            assert events[0].error.code == "test_error"
        finally:
            await backend.close()


class TestLocalBackendLifecycle:
    """Test LocalBackend lifecycle management."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with LocalBackend(EchoAdapter()) as backend:
            response = await backend.run(_make_request("Context"))
            assert response.output[0].content == "Echo: Context"

    @pytest.mark.asyncio
    async def test_lazy_start(self):
        """Backend should not start until first request."""
        backend = LocalBackend(EchoAdapter())
        assert not backend._started
        assert backend.port is None

        response = await backend.run(_make_request("Lazy"))
        assert backend._started
        assert backend.port is not None
        assert response.output[0].content == "Echo: Lazy"

        await backend.close()
        assert not backend._started

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        backend = LocalBackend(EchoAdapter())
        await backend.close()  # no-op when not started
        await backend.close()  # still no-op


class TestLocalBackendMetadata:
    """Test that request metadata survives the round-trip."""

    @pytest.mark.asyncio
    async def test_conversation_id_preserved(self):
        """The adapter receives the correct conversation_id."""

        class InspectingAdapter(GrpcAdapterServer):
            def __init__(self):
                self.last_request = None

            async def on_run(self, request: AgentRequest) -> AgentResponse:
                self.last_request = request
                return AgentResponse(status="completed", output=[
                    OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content="ok"),
                ])

            async def on_run_stream(self, request):
                yield AgentStreamEvent(type=StreamEventType.COMPLETED)

        adapter = InspectingAdapter()
        backend = LocalBackend(adapter)
        try:
            req = AgentRequest(
                response_id="resp_meta",
                conversation_id="conv_meta_123",
                stream=False,
                instructions="Be helpful",
                messages=[InputMessage(role="user", content="test")],
                metadata={"key1": "value1"},
            )
            await backend.run(req)

            assert adapter.last_request is not None
            assert adapter.last_request.response_id == "resp_meta"
            assert adapter.last_request.conversation_id == "conv_meta_123"
            assert adapter.last_request.instructions == "Be helpful"
            assert adapter.last_request.metadata == {"key1": "value1"}
            assert len(adapter.last_request.messages) == 1
            assert adapter.last_request.messages[0].role == "user"
            assert adapter.last_request.messages[0].content == "test"
        finally:
            await backend.close()
