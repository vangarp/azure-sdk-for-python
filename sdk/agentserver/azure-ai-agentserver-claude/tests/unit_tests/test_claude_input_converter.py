import pytest

from azure.ai.agentserver.claude.models.claude_input_converter import ClaudeInputConverter


@pytest.fixture()
def converter() -> ClaudeInputConverter:
    return ClaudeInputConverter()


@pytest.mark.unit
def test_transform_none_returns_empty(converter: ClaudeInputConverter) -> None:
    assert converter.transform_input(None) == ""


@pytest.mark.unit
def test_transform_string_returns_same(converter: ClaudeInputConverter) -> None:
    assert converter.transform_input("hello") == "hello"


@pytest.mark.unit
def test_transform_implicit_user_message_with_string(converter: ClaudeInputConverter) -> None:
    payload = [{"content": "How are you?"}]

    result = converter.transform_input(payload)

    assert result == "How are you?"


@pytest.mark.unit
def test_transform_implicit_user_message_with_input_text_list(converter: ClaudeInputConverter) -> None:
    payload = [
        {
            "content": [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": "world"},
            ]
        }
    ]

    result = converter.transform_input(payload)

    assert result == "Hello world"


@pytest.mark.unit
def test_transform_explicit_message_returns_text(converter: ClaudeInputConverter) -> None:
    payload = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "Hi there"},
            ],
        }
    ]

    result = converter.transform_input(payload)

    assert result == "Hi there"


@pytest.mark.unit
def test_transform_multiple_explicit_messages_returns_combined(converter: ClaudeInputConverter) -> None:
    payload = [
        {
            "type": "message",
            "role": "user",
            "content": "Hello",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "Greetings"},
            ],
        },
    ]

    result = converter.transform_input(payload)

    assert result == "Hello Greetings"


@pytest.mark.unit
def test_transform_mixed_messages_combines_to_string(converter: ClaudeInputConverter) -> None:
    payload = [
        {"content": "First"},
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "Second"},
            ],
        },
    ]

    result = converter.transform_input(payload)

    assert result == "First Second"


@pytest.mark.unit
def test_transform_invalid_input_type_raises(converter: ClaudeInputConverter) -> None:
    with pytest.raises(Exception) as exc_info:
        converter.transform_input({"content": "invalid"})

    assert "Unsupported input type" in str(exc_info.value)


@pytest.mark.unit
def test_transform_skips_non_text_entries(converter: ClaudeInputConverter) -> None:
    payload = [
        {
            "content": [
                {"type": "input_text", "text": 123},
                {"type": "image", "url": "https://example.com"},
            ]
        }
    ]

    result = converter.transform_input(payload)

    assert result == ""
