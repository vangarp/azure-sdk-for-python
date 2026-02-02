# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import TYPE_CHECKING

from ._version import VERSION

if TYPE_CHECKING:  # pragma: no cover
    from claude_agent_sdk import ClaudeSDKClient


def from_claude(agent: "ClaudeSDKClient"):
    from .claude import ClaudeAdapter

    return ClaudeAdapter(agent)


__all__ = ["from_claude"]
__version__ = VERSION
