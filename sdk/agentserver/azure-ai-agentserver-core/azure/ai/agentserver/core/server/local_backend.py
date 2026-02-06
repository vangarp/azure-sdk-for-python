# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
"""Local (in-process) backend that starts an adapter server on loopback.

``LocalBackend`` provides a single-process convenience mode: it launches
an adapter's gRPC server on ``localhost`` with an ephemeral port, then
connects to it via ``GrpcBackendClient``.  This preserves the protocol
boundary (adapter code runs through the wire format) while avoiding the
need to manage two separate processes.

Usage::

    from azure.ai.agentserver.core import FoundryCBAgent
    from azure.ai.agentserver.core.server.local_backend import LocalBackend
    from my_adapter import MyGrpcAdapter

    adapter = MyGrpcAdapter(graph=my_graph)
    backend = LocalBackend(adapter)
    server = FoundryCBAgent(backend=backend)
    server.run(port=8088)
"""
from __future__ import annotations

import asyncio  # pylint: disable=do-not-import-asyncio
import socket
from typing import AsyncIterator, Optional

from ..logger import get_logger
from ..wire.models import AgentRequest, AgentResponse, AgentStreamEvent
from .adapter_server import GrpcAdapterServer
from .backend import BackendClient
from .grpc_backend import GrpcBackendClient

logger = get_logger()


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalBackend(BackendClient):
    """In-process backend that starts a gRPC adapter server on loopback.

    The adapter runs on ``localhost:<port>`` in the same event loop.
    Requests go through the full wire protocol (gRPC), ensuring the same
    serialization/deserialization path as a true out-of-process deployment.

    :param adapter: The adapter server to run locally.
    :type adapter: GrpcAdapterServer
    :param port: Port for the adapter gRPC server.  ``0`` (default) means
        auto-select a free port.
    :type port: int
    """

    def __init__(self, adapter: GrpcAdapterServer, port: int = 0):
        self._adapter = adapter
        self._requested_port = port
        self._actual_port: Optional[int] = None
        self._grpc_server = None
        self._client: Optional[GrpcBackendClient] = None
        self._started = False

    @property
    def port(self) -> Optional[int]:
        """The actual port the adapter server is listening on (after start)."""
        return self._actual_port

    async def start(self) -> None:
        """Start the local adapter gRPC server and connect the client.

        Called automatically on first ``run`` / ``run_stream`` if not
        already started.
        """
        if self._started:
            return

        self._actual_port = self._requested_port or _find_free_port()
        logger.info(f"Starting local adapter server on port {self._actual_port}")

        self._grpc_server = await self._adapter.serve_background(
            port=self._actual_port
        )
        self._client = GrpcBackendClient(f"localhost:{self._actual_port}")
        self._started = True
        logger.info(f"Local adapter server started on port {self._actual_port}")

    async def run(self, request: AgentRequest) -> AgentResponse:
        await self.start()
        return await self._client.run(request)

    async def run_stream(self, request: AgentRequest) -> AsyncIterator[AgentStreamEvent]:
        await self.start()
        async for event in self._client.run_stream(request):
            yield event

    async def close(self) -> None:
        """Stop the local adapter server and close the client channel."""
        if self._client is not None:
            await self._client.close()
            self._client = None

        if self._grpc_server is not None:
            await self._grpc_server.stop(grace=2.0)
            self._grpc_server = None

        self._started = False
        logger.info("Local adapter server stopped")
