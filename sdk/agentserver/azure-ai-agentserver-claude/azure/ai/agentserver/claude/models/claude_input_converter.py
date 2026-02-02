# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from __future__ import annotations

from typing import Dict, List

from azure.ai.agentserver.core.logger import get_logger

logger = get_logger()


class ClaudeInputConverter:
    """Normalize inputs for Claude agent.

    Accepts: str | List | None
    Returns: str (message to send to Claude)
    """

    def transform_input(
        self,
        input: str | List[Dict] | None,
    ) -> str:
        logger.debug("Transforming input of type: %s", type(input))

        if input is None:
            return ""

        if isinstance(input, str):
            return input

        try:
            if isinstance(input, list):
                messages: list[str] = []

                for item in input:
                    # Case 1: ImplicitUserMessage with content as str or list of ItemContentInputText
                    if self._is_implicit_user_message(item):
                        content = item.get("content", None)
                        if isinstance(content, str):
                            messages.append(content)
                        elif isinstance(content, list):
                            text_parts: list[str] = []
                            for content_item in content:
                                text_content = self._extract_input_text(content_item)
                                if text_content:
                                    text_parts.append(text_content)
                            if text_parts:
                                messages.append(" ".join(text_parts))

                    # Case 2: Explicit message params (user/assistant/system)
                    elif (
                        item.get("type") == "message"
                        and item.get("role") is not None
                        and item.get("content") is not None
                    ):
                        item_content = item.get("content", None)
                        if item_content and isinstance(item_content, list):
                            text_parts: list[str] = []
                            for content_item in item_content:
                                item_text = self._extract_input_text(content_item)
                                if item_text:
                                    text_parts.append(item_text)
                            content_text = " ".join(text_parts) if text_parts else ""
                        elif item_content and isinstance(item_content, str):
                            content_text = str(item_content)
                        else:
                            content_text = ""

                        if content_text:
                            messages.append(content_text)

                # Return combined messages as a single string
                return " ".join(messages) if messages else ""

            raise TypeError(f"Unsupported input type: {type(input)}")
        except Exception as e:
            logger.error("Error processing messages: %s", e, exc_info=True)
            raise Exception(f"Error processing messages: {e}") from e  # pylint: disable=broad-exception-raised

    def _is_implicit_user_message(self, item: Dict) -> bool:
        return "content" in item and "role" not in item and "type" not in item

    def _extract_input_text(self, content_item: Dict) -> str:
        if content_item.get("type") == "input_text" and "text" in content_item:
            text_content = content_item.get("text")
            if isinstance(text_content, str):
                return text_content
        return ""  # type: ignore
