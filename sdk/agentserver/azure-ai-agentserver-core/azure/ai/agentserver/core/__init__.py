# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from ._version import VERSION
from .logger import configure as config_logging
from .server.adapter_server import AdapterServer, GrpcAdapterServer
from .server.backend import BackendClient
from .server.base import FoundryCBAgent
from .server.common.agent_run_context import AgentRunContext
from .server.grpc_backend import GrpcBackendClient
from .server.local_backend import LocalBackend
from .wire.state_converter import WireStateConverter

config_logging()

__all__ = [
    "AdapterServer",
    "AgentRunContext",
    "BackendClient",
    "FoundryCBAgent",
    "GrpcAdapterServer",
    "GrpcBackendClient",
    "LocalBackend",
    "WireStateConverter",
]
__version__ = VERSION
