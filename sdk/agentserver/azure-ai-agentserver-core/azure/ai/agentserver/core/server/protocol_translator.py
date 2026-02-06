# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# pylint: disable=logging-fstring-interpolation
"""Protocol translator: OpenAI Responses API ↔ Wire format.

This module is the critical bridge that converts between the OpenAI Responses
API types used on the HTTP boundary and the simplified wire format used for
adapter communication.

The core server owns the full complexity of the OpenAI event lifecycle.
Adapters only produce simplified ``AgentResponse`` / ``AgentStreamEvent``
objects.  This translator expands them into the rich event sequence that
OpenAI-compatible clients expect.

Translation responsibilities:
  - ``request_to_wire``: ``AgentRunContext`` → ``AgentRequest``
  - ``wire_response_to_openai``: ``AgentResponse`` → ``Response``
  - ``wire_stream_to_openai``: ``AsyncIterator[AgentStreamEvent]``
    → ``AsyncGenerator[ResponseStreamEvent]``
"""
from __future__ import annotations

import time
from typing import Any, AsyncGenerator, AsyncIterator, List, Optional

from ..logger import get_logger
from ..models import Response as OpenAIResponse, ResponseStreamEvent
from ..models import projects as project_models
from ..wire.models import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    InputMessage,
    OutputItem,
    OutputItemType,
    StreamEventType,
    ToolCallInfo,
)
from .common.agent_run_context import AgentRunContext

logger = get_logger()


# ---------------------------------------------------------------------------
# OpenAI request → Wire request
# ---------------------------------------------------------------------------


def request_to_wire(context: AgentRunContext) -> AgentRequest:
    """Convert an ``AgentRunContext`` (OpenAI Responses API format) into a
    wire-format ``AgentRequest`` for sending to an adapter backend.

    :param context: The run context built from the incoming HTTP request.
    :type context: AgentRunContext
    :return: A wire-format request.
    :rtype: AgentRequest
    """
    request = context.request
    messages = _convert_input_to_messages(request)

    agent_info = None
    agent_ref = request.get("agent")
    if agent_ref:
        agent_info = AgentInfo(
            name=getattr(agent_ref, "name", None),
            version=getattr(agent_ref, "version", None),
            type=getattr(agent_ref, "type", None),
        )

    return AgentRequest(
        response_id=context.response_id,
        conversation_id=context.conversation_id,
        stream=context.stream,
        instructions=request.get("instructions"),
        messages=messages,
        metadata=request.get("metadata") or {},
        agent=agent_info,
    )


def _convert_input_to_messages(request) -> List[InputMessage]:
    """Extract messages from the CreateResponse ``input`` field."""
    messages: List[InputMessage] = []
    raw_input = request.get("input")
    if raw_input is None:
        return messages

    # Simple string input → single user message
    if isinstance(raw_input, str):
        messages.append(InputMessage(role="user", content=raw_input))
        return messages

    # List of items
    if isinstance(raw_input, list):
        for item in raw_input:
            msg = _convert_input_item(item)
            if msg is not None:
                messages.append(msg)
        return messages

    logger.warning(f"Unsupported input type: {type(raw_input)}")
    return messages


def _convert_input_item(item) -> Optional[InputMessage]:
    """Convert a single OpenAI input item to an ``InputMessage``."""
    if isinstance(item, str):
        return InputMessage(role="user", content=item)

    if isinstance(item, dict):
        item_type = item.get("type", "message")

        if item_type == "message":
            return _convert_message_item(item)
        if item_type == "function_call":
            return _convert_function_call_item(item)
        if item_type == "function_call_output":
            return _convert_function_call_output_item(item)

        # Fallback: treat as user message with JSON content
        logger.warning(f"Unknown input item type '{item_type}', treating as user message")
        return InputMessage(role="user", content=str(item))

    logger.warning(f"Skipping unsupported input item: {type(item)}")
    return None


def _convert_message_item(item: dict) -> InputMessage:
    """Convert an OpenAI message item."""
    role = item.get("role", "user")
    content = item.get("content", "")

    if isinstance(content, str):
        return InputMessage(role=role, content=content)

    if isinstance(content, list):
        # Multi-part content: extract text and represent as content_parts
        from ..wire.models import ContentPart

        text_parts = []
        parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type", "text")
                if part_type in ("input_text", "text"):
                    text = part.get("text", "")
                    text_parts.append(text)
                    parts.append(ContentPart(type="text", text=text))
                else:
                    parts.append(ContentPart(type=part_type, extra=part))
            elif isinstance(part, str):
                text_parts.append(part)
                parts.append(ContentPart(type="text", text=part))

        return InputMessage(
            role=role,
            content=" ".join(text_parts) if text_parts else "",
            content_parts=parts if parts else None,
        )

    return InputMessage(role=role, content=str(content))


def _convert_function_call_item(item: dict) -> InputMessage:
    """Convert an OpenAI function_call input item → assistant message with tool_calls."""
    return InputMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCallInfo(
                id=item.get("call_id", ""),
                name=item.get("name", ""),
                arguments=item.get("arguments", ""),
            )
        ],
    )


def _convert_function_call_output_item(item: dict) -> InputMessage:
    """Convert an OpenAI function_call_output input item → tool message."""
    output = item.get("output", "")
    if isinstance(output, list):
        # Multi-part output: serialize to string
        import json

        output = json.dumps(output)
    return InputMessage(
        role="tool",
        content=str(output),
        tool_call_id=item.get("call_id", ""),
    )


# ---------------------------------------------------------------------------
# Wire response → OpenAI response (non-streaming)
# ---------------------------------------------------------------------------


def wire_response_to_openai(wire_resp: AgentResponse, context: AgentRunContext) -> OpenAIResponse:
    """Convert a wire-format ``AgentResponse`` into the full OpenAI
    ``Response`` object.

    :param wire_resp: The adapter's wire-format response.
    :type wire_resp: AgentResponse
    :param context: The original run context (provides IDs, agent info, etc.).
    :type context: AgentRunContext
    :return: An OpenAI-compatible Response model.
    :rtype: OpenAIResponse
    """
    output_items = []
    for idx, wire_item in enumerate(wire_resp.output):
        openai_item = _wire_output_to_openai_item(wire_item, context, idx)
        if openai_item is not None:
            output_items.append(openai_item)

    agent_id = context.get_agent_id_object()
    conversation = context.get_conversation_object()

    return OpenAIResponse(
        object="response",
        id=context.response_id,
        status=wire_resp.status,
        created_at=int(time.time()),
        output=output_items,
        agent=agent_id,
        conversation=conversation,
        metadata=context.request.get("metadata"),
    )


def _wire_output_to_openai_item(
    wire_item: OutputItem, context: AgentRunContext, index: int
) -> Optional[project_models.ItemResource]:
    """Convert a wire ``OutputItem`` into the corresponding OpenAI ``ItemResource`` subtype."""
    item_type = wire_item.type

    if item_type == OutputItemType.TEXT_MESSAGE:
        return _wire_text_to_openai(wire_item, context)
    if item_type == OutputItemType.FUNCTION_CALL:
        return _wire_function_call_to_openai(wire_item, context)
    if item_type == OutputItemType.FUNCTION_CALL_OUTPUT:
        return _wire_function_output_to_openai(wire_item, context)

    logger.warning(f"Unknown wire output item type: {item_type}")
    return None


def _wire_text_to_openai(
    wire_item: OutputItem, context: AgentRunContext
) -> project_models.ItemResource:
    """Convert a text_message wire item to the appropriate OpenAI message resource."""
    role = wire_item.role or "assistant"
    text = wire_item.content or ""

    if role == "assistant":
        content_type = project_models.ItemContentType.OUTPUT_TEXT
        content = [project_models.ItemContent({"type": content_type, "text": text, "annotations": []})]
        return project_models.ResponsesAssistantMessageItemResource(
            content=content,
            id=context.id_generator.generate_message_id(),
            status="completed",
        )
    elif role == "user":
        content_type = project_models.ItemContentType.INPUT_TEXT
        content = [project_models.ItemContent({"type": content_type, "text": text})]
        return project_models.ResponsesUserMessageItemResource(
            content=content,
            id=context.id_generator.generate_message_id(),
            status="completed",
        )
    elif role == "system":
        content_type = project_models.ItemContentType.INPUT_TEXT
        content = [project_models.ItemContent({"type": content_type, "text": text})]
        return project_models.ResponsesSystemMessageItemResource(
            content=content,
            id=context.id_generator.generate_message_id(),
            status="completed",
        )
    else:
        # Fallback: treat as assistant
        content_type = project_models.ItemContentType.OUTPUT_TEXT
        content = [project_models.ItemContent({"type": content_type, "text": text, "annotations": []})]
        return project_models.ResponsesAssistantMessageItemResource(
            content=content,
            id=context.id_generator.generate_message_id(),
            status="completed",
        )


def _wire_function_call_to_openai(
    wire_item: OutputItem, context: AgentRunContext
) -> project_models.FunctionToolCallItemResource:
    return project_models.FunctionToolCallItemResource(
        call_id=wire_item.call_id or "",
        name=wire_item.name or "",
        arguments=wire_item.arguments or "",
        id=context.id_generator.generate_function_call_id(),
        status="completed",
    )


def _wire_function_output_to_openai(
    wire_item: OutputItem, context: AgentRunContext
) -> project_models.FunctionToolCallOutputItemResource:
    return project_models.FunctionToolCallOutputItemResource(
        call_id=wire_item.call_id or "",
        output=wire_item.output or "",
        id=context.id_generator.generate_function_output_id(),
    )


# ---------------------------------------------------------------------------
# Wire stream → OpenAI streaming events
# ---------------------------------------------------------------------------


async def wire_stream_to_openai(
    events: AsyncIterator[AgentStreamEvent],
    context: AgentRunContext,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """Expand simplified wire-format stream events into the full OpenAI
    Responses API event lifecycle.

    The adapter only needs to emit simplified events (``text_delta``,
    ``output_item_added``, ``output_item_done``, ``completed``, ``error``).
    This function injects the boilerplate lifecycle events that
    OpenAI-compatible clients expect:

      1. ``response.created`` (initial Response with status=in_progress)
      2. ``response.in_progress``
      3. Per output item:
         a. ``response.output_item.added``
         b. ``response.content_part.added`` (for text items)
         c. ``response.output_text.delta`` / ``response.function_call_arguments.delta``
         d. ``response.output_text.done`` / ``response.function_call_arguments.done``
         e. ``response.content_part.done`` (for text items)
         f. ``response.output_item.done``
      4. ``response.completed`` (final Response with status=completed)

    :param events: Async iterator of wire-format events from the adapter.
    :param context: The original run context.
    :yields: OpenAI ``ResponseStreamEvent`` subclasses.
    """
    seq = _SequenceCounter()

    # State tracking for the streaming lifecycle
    created_emitted = False
    # Track accumulated text/args per output_index for *done events
    text_accumulators: dict[int, str] = {}
    args_accumulators: dict[int, str] = {}
    # Track which output items have been "added" (for auto-adding on first delta)
    items_added: set[int] = set()
    # Track item IDs and content_index per output
    item_ids: dict[int, str] = {}
    content_indices: dict[int, int] = {}
    # Collect output items for the final completed response
    completed_items: list[OutputItem] = []

    agent_id = context.get_agent_id_object()
    conversation = context.get_conversation_object()
    metadata = context.request.get("metadata")

    def _make_in_progress_response() -> OpenAIResponse:
        return OpenAIResponse(
            object="response",
            id=context.response_id,
            status="in_progress",
            created_at=int(time.time()),
            output=[],
            agent=agent_id,
            conversation=conversation,
            metadata=metadata,
        )

    async for event in events:
        # Emit response.created + response.in_progress on first event
        if not created_emitted:
            created_emitted = True
            initial_resp = _make_in_progress_response()
            yield project_models.ResponseCreatedEvent(
                sequence_number=seq.next(),
                response=initial_resp,
            )
            yield project_models.ResponseInProgressEvent(
                sequence_number=seq.next(),
                response=initial_resp,
            )

        event_type = event.type
        idx = event.output_index

        # --- output_item_added ---
        if event_type == StreamEventType.OUTPUT_ITEM_ADDED:
            if event.item is not None:
                openai_item = _wire_output_to_openai_item(event.item, context, idx)
                if openai_item is not None:
                    item_id = openai_item.id
                    item_ids[idx] = item_id
                    items_added.add(idx)
                    completed_items.append(event.item)
                    yield project_models.ResponseOutputItemAddedEvent(
                        sequence_number=seq.next(),
                        output_index=idx,
                        item=openai_item,
                    )
                    # For text messages, also emit content_part.added
                    if event.item.type == OutputItemType.TEXT_MESSAGE:
                        content_indices[idx] = 0
                        text_accumulators[idx] = ""
                        yield project_models.ResponseContentPartAddedEvent(
                            sequence_number=seq.next(),
                            item_id=item_id,
                            output_index=idx,
                            content_index=0,
                            part=project_models.ItemContent(
                                {"type": project_models.ItemContentType.OUTPUT_TEXT, "text": "", "annotations": []}
                            ),
                        )
                    elif event.item.type == OutputItemType.FUNCTION_CALL:
                        args_accumulators[idx] = ""
            continue

        # --- text_delta ---
        if event_type == StreamEventType.TEXT_DELTA:
            delta = event.delta or ""
            text_accumulators.setdefault(idx, "")
            text_accumulators[idx] += delta

            # Auto-add item if adapter didn't send explicit output_item_added
            if idx not in items_added:
                _auto_add_text_item(idx, items_added, item_ids, content_indices,
                                    text_accumulators, completed_items, context, seq)

            item_id = item_ids.get(idx, "")
            c_idx = content_indices.get(idx, 0)
            yield project_models.ResponseTextDeltaEvent(
                sequence_number=seq.next(),
                item_id=item_id,
                output_index=idx,
                content_index=c_idx,
                delta=delta,
            )
            continue

        # --- function_call_arguments_delta ---
        if event_type == StreamEventType.FUNCTION_CALL_ARGUMENTS_DELTA:
            delta = event.delta or ""
            args_accumulators.setdefault(idx, "")
            args_accumulators[idx] += delta

            item_id = item_ids.get(idx, "")
            yield project_models.ResponseFunctionCallArgumentsDeltaEvent(
                sequence_number=seq.next(),
                item_id=item_id,
                output_index=idx,
                delta=delta,
            )
            continue

        # --- output_item_done ---
        if event_type == StreamEventType.OUTPUT_ITEM_DONE:
            item_id = item_ids.get(idx, "")

            # Emit text_done + content_part_done for text items
            if idx in text_accumulators:
                full_text = text_accumulators.pop(idx)
                c_idx = content_indices.get(idx, 0)
                yield project_models.ResponseTextDoneEvent(
                    sequence_number=seq.next(),
                    item_id=item_id,
                    output_index=idx,
                    content_index=c_idx,
                    text=full_text,
                )
                yield project_models.ResponseContentPartDoneEvent(
                    sequence_number=seq.next(),
                    item_id=item_id,
                    output_index=idx,
                    content_index=c_idx,
                    part=project_models.ItemContent(
                        {
                            "type": project_models.ItemContentType.OUTPUT_TEXT,
                            "text": full_text,
                            "annotations": [],
                        }
                    ),
                )

            # Emit function_call_arguments_done for function call items
            if idx in args_accumulators:
                full_args = args_accumulators.pop(idx)
                yield project_models.ResponseFunctionCallArgumentsDoneEvent(
                    sequence_number=seq.next(),
                    item_id=item_id,
                    output_index=idx,
                    arguments=full_args,
                )

            # Get the final output item (from the event or from stored state)
            if event.item is not None:
                openai_item = _wire_output_to_openai_item(event.item, context, idx)
            else:
                # Reconstruct from accumulated state
                openai_item = _reconstruct_done_item(idx, item_ids, text_accumulators,
                                                     args_accumulators, completed_items, context)

            if openai_item is not None:
                yield project_models.ResponseOutputItemDoneEvent(
                    sequence_number=seq.next(),
                    output_index=idx,
                    item=openai_item,
                )
            continue

        # --- completed ---
        if event_type == StreamEventType.COMPLETED:
            # Build the final Response from the wire response or accumulated state
            if event.response is not None:
                final_resp = wire_response_to_openai(event.response, context)
            else:
                # Build from accumulated items
                openai_items = []
                for i, ci in enumerate(completed_items):
                    oi = _wire_output_to_openai_item(ci, context, i)
                    if oi is not None:
                        openai_items.append(oi)
                final_resp = OpenAIResponse(
                    object="response",
                    id=context.response_id,
                    status="completed",
                    created_at=int(time.time()),
                    output=openai_items,
                    agent=agent_id,
                    conversation=conversation,
                    metadata=metadata,
                )
            yield project_models.ResponseCompletedEvent(
                sequence_number=seq.next(),
                response=final_resp,
            )
            continue

        # --- error ---
        if event_type == StreamEventType.ERROR:
            error = event.error
            yield project_models.ResponseErrorEvent(
                sequence_number=seq.next(),
                code=error.code if error else "unknown",
                message=error.message if error else "Unknown error",
            )
            continue

        logger.debug(f"Unknown wire stream event type: {event_type}")

    # If the adapter stream ended without a completed event, emit one
    if created_emitted:
        # Flush any remaining accumulators (items that had deltas but no explicit done)
        for idx in list(text_accumulators.keys()):
            item_id = item_ids.get(idx, "")
            full_text = text_accumulators.pop(idx)
            c_idx = content_indices.get(idx, 0)
            yield project_models.ResponseTextDoneEvent(
                sequence_number=seq.next(),
                item_id=item_id,
                output_index=idx,
                content_index=c_idx,
                text=full_text,
            )
            yield project_models.ResponseContentPartDoneEvent(
                sequence_number=seq.next(),
                item_id=item_id,
                output_index=idx,
                content_index=c_idx,
                part=project_models.ItemContent(
                    {"type": project_models.ItemContentType.OUTPUT_TEXT, "text": full_text, "annotations": []}
                ),
            )
            yield project_models.ResponseOutputItemDoneEvent(
                sequence_number=seq.next(),
                output_index=idx,
                item=_wire_text_to_openai(
                    OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=full_text),
                    context,
                ),
            )

        for idx in list(args_accumulators.keys()):
            item_id = item_ids.get(idx, "")
            full_args = args_accumulators.pop(idx)
            yield project_models.ResponseFunctionCallArgumentsDoneEvent(
                sequence_number=seq.next(),
                item_id=item_id,
                output_index=idx,
                arguments=full_args,
            )

        # Final completed event
        openai_items = []
        for i, ci in enumerate(completed_items):
            oi = _wire_output_to_openai_item(ci, context, i)
            if oi is not None:
                openai_items.append(oi)
        final_resp = OpenAIResponse(
            object="response",
            id=context.response_id,
            status="completed",
            created_at=int(time.time()),
            output=openai_items,
            agent=agent_id,
            conversation=conversation,
            metadata=metadata,
        )
        yield project_models.ResponseCompletedEvent(
            sequence_number=seq.next(),
            response=final_resp,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_add_text_item(idx, items_added, item_ids, content_indices,
                        text_accumulators, completed_items, context, seq):
    """Auto-generate output_item_added events when adapter sends deltas
    without an explicit output_item_added event."""
    items_added.add(idx)
    item_id = context.id_generator.generate_message_id()
    item_ids[idx] = item_id
    content_indices[idx] = 0
    text_accumulators.setdefault(idx, "")
    completed_items.append(OutputItem(type=OutputItemType.TEXT_MESSAGE, role="assistant", content=""))


def _reconstruct_done_item(idx, item_ids, text_accumulators, args_accumulators,
                           completed_items, context):
    """Reconstruct a done item from accumulated state when the adapter
    doesn't include the full item in the output_item_done event."""
    if idx < len(completed_items):
        return _wire_output_to_openai_item(completed_items[idx], context, idx)
    return None


class _SequenceCounter:
    """Monotonically increasing sequence number generator."""

    def __init__(self):
        self._value = 0

    def next(self) -> int:
        self._value += 1
        return self._value
