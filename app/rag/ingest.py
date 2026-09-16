from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.rag.embeddings import build_embeddings
from app.rag.vectorstore import ChromaVectorStore, VectorStore
from app.utils.logging import logger

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_knowledge_documents(knowledge_path: str | Path | None = None) -> list[Document]:
    settings = get_settings()
    root = Path(knowledge_path or settings.knowledge_path)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "category": path.stem.lower(),
                },
            )
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


def ingest_knowledge(
    knowledge_path: str | Path | None = None,
    store: VectorStore | None = None,
    force_hash_embeddings: bool = False,
) -> dict[str, int | list[str]]:
    documents = load_knowledge_documents(knowledge_path)
    chunks = chunk_documents(documents)
    vector_store = store or ChromaVectorStore(embeddings=build_embeddings(force_hash=force_hash_embeddings))
    vector_store.reset()

    ids = [f"{chunk.metadata.get('source', 'doc')}-{index}" for index, chunk in enumerate(chunks)]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source": str(chunk.metadata.get("source", "unknown")),
            "category": str(chunk.metadata.get("category", "general")),
        }
        for chunk in chunks
    ]
    if ids:
        vector_store.upsert(ids, texts, metadatas)

    sources = sorted({str(item.metadata.get("source")) for item in documents})
    logger.info(
        "Knowledge ingestion complete",
        extra={
            "request_id": "-",
            "conversation_id": "-",
            "endpoint": "ingest",
            "latency_ms": "-",
        },
    )
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "sources": sources,
    }
