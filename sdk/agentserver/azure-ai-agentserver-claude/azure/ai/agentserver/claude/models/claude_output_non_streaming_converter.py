# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from __future__ import annotations

import datetime
from typing import Any, List

from azure.ai.agentserver.core import AgentRunContext
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import Response as OpenAIResponse
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponsesAssistantMessageItemResource,
)

logger = get_logger()


class ClaudeOutputNonStreamingConverter:  # pylint: disable=name-too-long
    """Non-streaming converter: Claude response -> OpenAIResponse."""

    def __init__(self, context: AgentRunContext):
        self._context = context
        self._response_id = None
        self._response_created_at = None

    def _ensure_response_started(self) -> None:
        if not self._response_id:
            self._response_id = self._context.response_id  # type: ignore
        if not self._response_created_at:
            self._response_created_at = int(datetime.datetime.now(datetime.timezone.utc).timestamp())  # type: ignore

    def _build_item_content_output_text(self, text: str) -> ItemContentOutputText:
        return ItemContentOutputText(text=text, annotations=[])

    def _new_assistant_message_item(self, message_text: str) -> ResponsesAssistantMessageItemResource:
        item_content = self._build_item_content_output_text(message_text)
        return ResponsesAssistantMessageItemResource(
            id=self._context.id_generator.generate_message_id(), status="completed", content=[item_content]
        )

    def transform_output_for_response(self, response_chunks: List[Any]) -> OpenAIResponse:
        """Build an OpenAIResponse from Claude response chunks.

        :param response_chunks: The list of response chunks from Claude.
        :type response_chunks: List[Any]

        :return: The constructed OpenAIResponse.
        :rtype: OpenAIResponse
        """
        logger.debug("Transforming non-streaming response (chunks=%d)", len(response_chunks))
        self._ensure_response_started()

        completed_items: List[dict] = []

        # Combine all text from response chunks
        full_text = ""
        for chunk in response_chunks:
            if hasattr(chunk, "content"):
                full_text += str(chunk.content)
            elif isinstance(chunk, str):
                full_text += chunk
            elif isinstance(chunk, dict) and "content" in chunk:
                full_text += str(chunk["content"])

        # Create a single assistant message item with the combined text
        if full_text:
            assistant_message = self._new_assistant_message_item(full_text)
            completed_items.append(assistant_message)

        response_data = self._construct_response_data(completed_items)
        openai_response = OpenAIResponse(response_data)
        logger.info("Non-streaming response constructed with %d items", len(completed_items))
        return openai_response

    def _construct_response_data(self, completed_items: List[dict]) -> dict:
        return {
            "id": self._response_id,
            "object": "realtime.response",
            "status": "completed",
            "created_at": self._response_created_at,
            "output": completed_items,
        }
