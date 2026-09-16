import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import PortfolioAgent
from app.models.requests import AgentResponse
from app.rag.retriever import KnowledgeRetriever
from app.rag.vectorstore import RetrievedChunk
from app.utils.time import now_tz


class FakeStore:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []

    def reset(self) -> None:
        self.chunks = []

    def upsert(self, ids, documents, metadatas) -> None:
        self.chunks = [
            RetrievedChunk(
                text=document,
                source=str(meta.get("source", "unknown")),
                category=str(meta.get("category", "general")),
                distance=0.1,
            )
            for document, meta in zip(documents, metadatas, strict=False)
        ]

    def similarity_search(self, query: str, k: int = 4, category: str | None = None) -> list[RetrievedChunk]:
        items = self.chunks
        if category:
            items = [chunk for chunk in items if chunk.category == category] or items
        return items[:k]

    def all_chunks(self) -> list[RetrievedChunk]:
        return list(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.chunks})


class FakeGemini:
    def generate(self, *, system_instruction: str, contents: list, tools=None):
        blob = " ".join(str(item) for item in contents).lower()
        if "system prompt" in blob or "ignore" in blob and "instructions" in blob:
            text = "Nice try. I can tell you about boss, but I'm not exposing my backstage instructions."
        elif "favorite food" in blob or "(no relevant knowledge retrieved)" in blob and "food" in blob:
            text = "I don't think boss has given me that lore yet."
        elif "datetime=" in blob and "authoritative" in blob:
            text = "Right now it's the time from the tool, not a guess."
        elif "metaconnect" in blob or "projects.md" in blob:
            text = "Boss has built MetaConnect and Pneumo.AI. MetaConnect is the one with WebRTC."
        elif "what is rag" in blob:
            text = "RAG is retrieval-augmented generation — pull facts, then generate."
        else:
            text = "Yeah, for sure. I can tell you more about boss."
        return type("Response", (), {"text": text, "candidates": []})()

    def extract_text(self, response) -> str:
        return response.text

    def stream(self, *, system_instruction: str, contents: list):
        yield "Yo "
        yield "what's up?"


class ScriptedAgent:
    def reply(self, conversation_id: str, message: str, history=None) -> AgentResponse:
        now = now_tz()
        return AgentResponse(
            conversation_id=conversation_id,
            message=f"echo:{message}",
            response_type="text",
            sources=[],
            timestamp=now,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=48),
        )

    def stream_text(self, conversation_id: str, message: str, history=None):
        yield "streamed"


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore(
        [
            RetrievedChunk(
                text="MetaConnect uses WebRTC for real-time communication. Pneumo.AI is an AI project.",
                source="projects.md",
                category="projects",
                distance=0.1,
            ),
            RetrievedChunk(
                text="Languages: JavaScript, Python, Java. Frontend React. Backend Node/Express and FastAPI.",
                source="skills.md",
                category="skills",
                distance=0.1,
            ),
        ]
    )


@pytest.fixture
def agent(fake_store: FakeStore) -> PortfolioAgent:
    return PortfolioAgent(
        retriever=KnowledgeRetriever(fake_store),
        gemini=FakeGemini(),
    )


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as test_client:
        test_client.app.state.agent = ScriptedAgent()
        yield test_client
