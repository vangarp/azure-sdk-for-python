from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from langgraph_adapter import from_langgraph  # noqa: E402

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from openai import OpenAI, OpenAIError

from azure.ai.agentserver.core.wire.models import (
    AgentRequest,
    AgentResponse,
    AgentStreamEvent,
    OutputItem,
    OutputItemType,
)
from azure.ai.agentserver.core.wire.state_converter import WireStateConverter

load_dotenv()

API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
BASE_URL = os.environ.get("AZURE_OPENAI_ENDPOINT") + "openai/v1"
DEPLOYMENT = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")  # optional override
DEFAULT_MODEL = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Simple in-memory knowledge base (replace with real vector DB in production)
# ---------------------------------------------------------------------------
@dataclass
class KBEntry:
    id: str
    text: str
    tags: List[str]


KNOWLEDGE_BASE: List[KBEntry] = [
    KBEntry(
        id="doc1",
        text="LangGraph enables stateful AI workflows via graphs of nodes.",
        tags=["langgraph", "workflow"],
    ),
    KBEntry(
        id="doc2",
        text="Retrieval augmented generation improves answer grounding by injecting documents.",
        tags=["rag", "retrieval", "grounding"],
    ),
    KBEntry(
        id="doc3",
        text="Streaming responses send partial model outputs for lower latency user experience.",
        tags=["streaming", "latency"],
    ),
]


# ---------------------------------------------------------------------------
# LangGraph State definition
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    query: str
    messages: List[Dict[str, Any]]  # simplified message records
    needs_retrieval: bool
    retrieved: List[Dict[str, Any]]  # selected documents
    answer_parts: List[str]  # incremental answer assembly
    final_answer: str  # final answer text
    _stream_events: List[Any]  # buffered upstream model delta events (if any)
    stream: bool  # whether streaming was requested


# ---------------------------------------------------------------------------
# Utility: naive keyword scoring retrieval
# ---------------------------------------------------------------------------
KEYWORDS = {
    "langgraph": ["langgraph", "graph"],
    "retrieval": ["retrieval", "rag", "ground"],
    "stream": ["stream", "latency", "partial"],
}


def retrieve_docs(question: str, k: int = 2) -> List[Dict[str, Any]]:
    scores: List[tuple[float, KBEntry]] = []
    lower_q = question.lower()
    for entry in KNOWLEDGE_BASE:
        score = 0
        for token in entry.tags:
            if token in lower_q:
                score += 2
        for kw_group in KEYWORDS.values():
            for kw in kw_group:
                if kw in lower_q and kw in entry.text.lower():
                    score += 1
        if score > 0:
            scores.append((score, entry))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [{"id": e.id, "text": e.text, "score": s} for s, e in scores[:k]]


# ---------------------------------------------------------------------------
# Custom Converter
# ---------------------------------------------------------------------------
class RAGStateConverter(WireStateConverter):
    """Converter implementing mini RAG logic (non‑streaming only)."""

    def get_stream_mode(self, request: AgentRequest) -> str:  # noqa: D401
        if request.stream:
            raise NotImplementedError("Streaming not supported in this sample.")
        return "values"

    def request_to_state(self, request: AgentRequest) -> Dict[str, Any]:  # noqa: D401
        # Extract user input from wire messages
        user_input = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_input = msg.content
                break

        messages = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        messages.append({"role": "user", "content": user_input})
        res = {
            "query": user_input,
            "messages": messages,
            "needs_retrieval": False,
            "retrieved": [],
            "answer_parts": [],
            "stream": False,
        }
        print("initial state:", res)
        return res

    def state_to_response(
        self, state: Dict[str, Any]
    ) -> AgentResponse:  # noqa: D401
        final_answer = state.get("final_answer") or "(no answer generated)"
        print(f"convert state to response, state: {state}")

        return AgentResponse(
            status="completed",
            output=[
                OutputItem(
                    type=OutputItemType.TEXT_MESSAGE,
                    role="assistant",
                    content=final_answer,
                )
            ],
        )

    async def state_to_stream(  # noqa: D401
        self,
        stream: AsyncIterator,
        request: AgentRequest,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        raise NotImplementedError("Streaming not supported in this sample.")
        # Make this an async generator
        yield  # type: ignore  # pragma: no cover


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------


def _normalize_query(val: Any) -> str:
    """Extract a lowercase text query from varied structures.

    Accepts:
      * str
      * dict with 'content' or 'text'
      * list of mixed items (recursively extracts first textual segment)
    Falls back to JSON stringification for unknown structures.
    """
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, dict):
        for k in ("content", "text", "value"):
            v = val.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        # flatten simple dict string values
        parts = [str(v) for v in val.values() if isinstance(v, (str, int, float))]
        if parts:
            return " ".join(parts).lower()
    if isinstance(val, list):
        for item in val:  # take first meaningful piece
            extracted = _normalize_query(item)
            if extracted:
                return extracted
        return ""
    try:
        return str(val).strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def analyze_intent(state: RAGState) -> RAGState:
    raw_q = state.get("query", "")
    q = _normalize_query(raw_q)
    keywords = ("what", "how", "explain", "retrieval", "langgraph", "stream")
    needs = any(kw in q for kw in keywords)
    state["needs_retrieval"] = needs
    # Also store normalized form for downstream nodes if different
    if isinstance(raw_q, (dict, list)):
        state["query"] = q
    return state


def retrieve_if_needed(state: RAGState) -> RAGState:
    if state.get("needs_retrieval"):
        state["retrieved"] = retrieve_docs(state.get("query", ""))
    return state


def generate_answer(state: RAGState) -> RAGState:
    query = state.get("query", "")
    retrieved = state.get("retrieved", [])

    model_name = DEPLOYMENT or DEFAULT_MODEL

    def synthesize_answer() -> tuple[str, List[str]]:
        if not retrieved:
            text = f"Answer: {query}" if query else "No question provided."
            return text, [text]
        doc_summaries = "; ".join(r["text"] for r in retrieved)
        answer = f"Based on docs: {doc_summaries}\n\nAnswer: {query}"[:4000]
        return answer, [answer]

    if API_KEY and BASE_URL:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        try:
            resp = client.responses.create(model=model_name, input=query)
            text = getattr(resp, "output_text", None)
            if not text:
                text = json.dumps(resp.model_dump(mode="json", exclude_none=True))[:500]
            state["final_answer"] = text
            state["answer_parts"] = [text]
            return state
        except OpenAIError:  # fallback
            state["final_answer"], state["answer_parts"] = synthesize_answer()
            return state
    state["final_answer"], state["answer_parts"] = synthesize_answer()
    return state


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def _build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("analyze", analyze_intent)
    graph.add_node("retrieve", retrieve_if_needed)
    graph.add_node("answer", generate_answer)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = _build_graph()
    converter = RAGStateConverter()
    from_langgraph(graph, converter).run()
