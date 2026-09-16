from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import ingest_knowledge


def main() -> None:
    result = ingest_knowledge()
    print("Ingestion complete")
    print(f"Documents: {result['documents']}")
    print(f"Chunks: {result['chunks']}")
    print("Sources:")
    for source in result["sources"]:
        print(f"  - {source}")


if __name__ == "__main__":
    main()
