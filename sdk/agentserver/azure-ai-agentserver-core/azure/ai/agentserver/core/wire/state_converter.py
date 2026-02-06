# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wire-format state converter protocol.

For adapters whose frameworks don't use a standard chat-message interface
(e.g., LangGraph with a custom state schema), a ``WireStateConverter``
bridges between the wire format and the framework's native state
representation.

Unlike the legacy ``LanggraphStateConverter`` (which operated on OpenAI
``Response`` / ``ResponseStreamEvent`` types), this converter works
entirely in wire-format types — no OpenAI or Azure SDK imports needed.

Usage::

    class MyCustomStateConverter(WireStateConverter):
        def request_to_state(self, request):
            return {"my_field": request.messages[-1].content}

        def state_to_response(self, state):
            return AgentResponse(
                status="completed",
                output=[OutputItem(type="text_message", role="assistant", content=state["result"])],
            )

        async def state_to_stream(self, stream, request):
            async for chunk in stream:
                yield AgentStreamEvent(type="text_delta", delta=chunk["text"])
            yield AgentStreamEvent(type="completed")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator

from .models import AgentRequest, AgentResponse, AgentStreamEvent


class WireStateConverter(ABC):
    """Abstract converter between wire format and framework-native state.

    Adapter authors implement this to handle non-standard state schemas.
    All inputs and outputs are wire-format types — the converter never
    needs to import OpenAI SDK or Azure SDK models.

    :param request: The wire-format request from the core server.
    """

    @abstractmethod
    def request_to_state(self, request: AgentRequest) -> Any:
        """Convert a wire ``AgentRequest`` into framework-native input state.

        :param request: The wire-format request.
        :return: Framework-native input (dict, dataclass, etc.).
        """
        ...

    @abstractmethod
    def state_to_response(self, state: Any) -> AgentResponse:
        """Convert completed framework-native state into a wire ``AgentResponse``.

        :param state: The framework's output after execution.
        :return: A wire-format response.
        """
        ...

    @abstractmethod
    async def state_to_stream(
        self, stream: AsyncIterator, request: AgentRequest
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Convert framework-native streaming output into wire stream events.

        :param stream: An async iterator of framework-native stream chunks.
        :param request: The original wire-format request (for context).
        :return: An async generator of wire-format stream events.
        """
        ...

    def get_stream_mode(self, request: AgentRequest) -> str:
        """Return the framework-specific stream mode string.

        Override this to control how the framework's streaming API is
        invoked.  The default returns ``"messages"`` for streaming
        requests and ``"updates"`` for non-streaming.

        :param request: The wire-format request.
        :return: A stream mode string passed to the framework.
        """
        return "messages" if request.stream else "updates"
