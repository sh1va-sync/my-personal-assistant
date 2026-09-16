import re

from app.config.settings import get_settings
from app.rag.vectorstore import RetrievedChunk, VectorStore

CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "projects": ("project", "built", "app", "webrtc", "metaconnect", "pneumo", "portfolio work"),
    "skills": ("skill", "tech stack", "language", "framework", "python", "react", "node"),
    "education": ("education", "college", "university", "degree", "student", "cs"),
    "experience": ("experience", "work", "job", "intern"),
    "contact": ("contact", "email", "github", "linkedin", "hire", "reach"),
    "interests": ("interest", "hobby", "like", "into"),
    "goals": ("goal", "future", "headed", "next"),
    "about": ("who is", "about shiva", "who are you"),
    "faq": ("faq", "often asked"),
}

_TOKEN = re.compile(r"[a-z0-9]+")


def infer_category(query: str) -> str | None:
    lowered = query.lower()
    for category, hints in CATEGORY_HINTS.items():
        if any(hint in lowered for hint in hints):
            return category
    return None


def lexical_score(query: str, text: str) -> float:
    query_tokens = set(_TOKEN.findall(query.lower()))
    if not query_tokens:
        return 0.0
    text_tokens = set(_TOKEN.findall(text.lower()))
    return len(query_tokens & text_tokens) / len(query_tokens)


class KnowledgeRetriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        settings = get_settings()
        top_k = k or settings.retrieval_top_k
        category = infer_category(query)

        vector_hits = self.store.similarity_search(query, k=max(top_k * 2, 6), category=category)
        if category and not vector_hits:
            vector_hits = self.store.similarity_search(query, k=max(top_k * 2, 6), category=None)

        corpus = [chunk for chunk in vector_hits if chunk.visibility == "public"]
        if hasattr(self.store, "all_chunks") and self.store.count() <= 400:
            seen = {chunk.text for chunk in corpus}
            for chunk in self.store.all_chunks():
                if category and chunk.category != category:
                    continue
                if chunk.visibility != "public":
                    continue
                if chunk.text not in seen:
                    corpus.append(chunk)
                    seen.add(chunk.text)

        ranked: list[tuple[float, RetrievedChunk]] = []
        for chunk in corpus:
            vector_score = 0.0
            if chunk.distance is not None:
                vector_score = max(0.0, 1.0 - chunk.distance)
            score = vector_score + lexical_score(query, f"{chunk.source} {chunk.category} {chunk.text}")
            if category and chunk.category == category:
                score += 0.15
            ranked.append((score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for score, chunk in ranked if score > 0][:top_k]
        if selected:
            return selected
        return [chunk for _, chunk in ranked[:top_k]]
