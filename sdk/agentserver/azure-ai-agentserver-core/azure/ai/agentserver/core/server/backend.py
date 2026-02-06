# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Abstract backend client for out-of-process adapter communication.

``BackendClient`` defines the interface that the ``FoundryCBAgent`` server
uses to delegate work to an adapter.  Concrete implementations handle
the transport details (gRPC, HTTP/SSE, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..wire.models import AgentRequest, AgentResponse, AgentStreamEvent


class BackendClient(ABC):
    """Transport-agnostic interface for communicating with an adapter backend.

    Implementations must support both non-streaming and streaming invocation
    and provide a ``close()`` hook for resource cleanup.
    """

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentResponse:
        """Execute a non-streaming agent invocation.

        :param request: The wire-format request.
        :type request: AgentRequest
        :return: The wire-format response.
        :rtype: AgentResponse
        """
        ...

    @abstractmethod
    async def run_stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
        """Execute a streaming agent invocation.

        :param request: The wire-format request.
        :type request: AgentRequest
        :return: An async iterator of wire-format stream events.
        :rtype: AsyncIterator[AgentStreamEvent]
        """
        ...

    async def close(self) -> None:
        """Release any underlying resources (channels, connections, etc.).

        Subclasses should override to clean up transport-specific state.
        """

    async def __aenter__(self) -> "BackendClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
