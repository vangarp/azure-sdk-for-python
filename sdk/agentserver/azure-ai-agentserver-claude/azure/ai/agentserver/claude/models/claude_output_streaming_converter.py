# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from __future__ import annotations

import datetime
from typing import Any, List

from azure.ai.agentserver.core import AgentRunContext
from azure.ai.agentserver.core.logger import get_logger
from azure.ai.agentserver.core.models import ResponseStreamEvent
from azure.ai.agentserver.core.models.projects import (
    ItemContentOutputText,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsesAssistantMessageItemResource,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

logger = get_logger()


class ClaudeOutputStreamingConverter:  # pylint: disable=name-too-long
    """Streaming converter: Claude streaming updates -> ResponseStreamEvent."""

    def __init__(self, context: AgentRunContext):
        self._context = context
        self._response_id = context.response_id
        self._response_created_at = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        self._sequence_number = 0
        self._output_index = 0
        self._item_id = None
        self._content_index = 0
        self._text_buffer = ""

    def next_sequence(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def initial_events(self) -> List[ResponseStreamEvent]:
        """Generate the initial events for streaming (created, in_progress)."""
        return [
            ResponseCreatedEvent(
                sequence_number=self.next_sequence(),
            ),
            ResponseInProgressEvent(
                sequence_number=self.next_sequence(),
            ),
        ]

    def completion_events(self) -> List[ResponseStreamEvent]:
        """Generate the final events for streaming (done, completed)."""
        events = []

        # If we have accumulated text, emit final events
        if self._text_buffer and self._item_id:
            events.append(
                ResponseTextDoneEvent(
                    sequence_number=self.next_sequence(),
                    output_index=self._output_index,
                    content_index=self._content_index,
                    text=self._text_buffer,
                )
            )
            events.append(
                ResponseContentPartDoneEvent(
                    sequence_number=self.next_sequence(),
                    output_index=self._output_index,
                    content_index=self._content_index,
                    part=ItemContentOutputText(text=self._text_buffer, annotations=[]),
                )
            )
            events.append(
                ResponseOutputItemDoneEvent(
                    sequence_number=self.next_sequence(),
                    output_index=self._output_index,
                    item=ResponsesAssistantMessageItemResource(
                        id=self._item_id,
                        status="completed",
                        content=[ItemContentOutputText(text=self._text_buffer, annotations=[])],
                    ),
                )
            )

        events.append(
            ResponseCompletedEvent(
                sequence_number=self.next_sequence(),
            )
        )
        return events

    def transform_output_for_streaming(self, update: Any) -> List[ResponseStreamEvent]:
        """Transform a Claude update into a list of ResponseStreamEvent.

        :param update: A single update from Claude streaming response.
        :type update: Any

        :return: List of ResponseStreamEvent to yield.
        :rtype: List[ResponseStreamEvent]
        """
        events = []

        # Extract text from the update
        text = ""
        if hasattr(update, "content"):
            text = str(update.content)
        elif isinstance(update, str):
            text = update
        elif isinstance(update, dict) and "content" in update:
            text = str(update["content"])

        if not text:
            return events

        # Start a new message item if needed
        if self._item_id is None:
            self._item_id = self._context.id_generator.generate_message_id()
            events.append(
                ResponseOutputItemAddedEvent(
                    sequence_number=self.next_sequence(),
                    output_index=self._output_index,
                    item=ResponsesAssistantMessageItemResource(
                        id=self._item_id,
                        status="in_progress",
                        content=[],
                    ),
                )
            )
            events.append(
                ResponseContentPartAddedEvent(
                    sequence_number=self.next_sequence(),
                    output_index=self._output_index,
                    content_index=self._content_index,
                    part=ItemContentOutputText(text="", annotations=[]),
                )
            )

        # Accumulate text and emit delta
        self._text_buffer += text
        events.append(
            ResponseTextDeltaEvent(
                sequence_number=self.next_sequence(),
                output_index=self._output_index,
                content_index=self._content_index,
                delta=text,
            )
        )

        return events
