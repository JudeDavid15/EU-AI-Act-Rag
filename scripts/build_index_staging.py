import json
import pickle
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/processed/chunks_consolidated_preview.json"
BM25_PATH = ROOT / "data/processed/bm25_consolidated_preview.pkl"
QDRANT_PATH = ROOT / "data/qdrant_db_consolidated_preview"

COLLECTION = "eu_ai_act_clauses_preview"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    if BM25_PATH.exists() or QDRANT_PATH.exists():
        raise FileExistsError(
            "Staging index already exists. Stop and inspect it first."
        )

    with SOURCE.open("r", encoding="utf-8") as f:
        parents = json.load(f)

    children = [
        child
        for parent in parents
        for child in parent.get("children", [])
    ]

    if not children or any(
        not child.get("citation_label") or not child.get("text", "").strip()
        for child in children
    ):
        raise ValueError("Missing text or citation labels.")

    print(f"Staging {len(children)} chunks")

    bm25 = BM25Okapi([
        child["text"].lower().split()
        for child in children
    ])

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": children}, f)

    model = SentenceTransformer(MODEL_NAME)
    texts = [
        f'{c["article_number"]}-{c["title"]}:{c["text"]}'
        for c in children
    ]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE,
            ),
        )

        points = []
        for idx, (child, vector) in enumerate(zip(children, embeddings)):
            points.append(PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    key: child.get(key)
                    for key in (
                        "chunk_id",
                        "parent_id",
                        "article_number",
                        "title",
                        "text",
                        "citation_label",
                        "paragraph_number",
                    )
                },
            ))

        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Qdrant staging points: {len(points)}")
    finally:
        client.close()

    print("STAGING BUILD COMPLETE")
    print(f"BM25: {BM25_PATH}")
    print(f"Qdrant: {QDRANT_PATH}")


if __name__ == "__main__":
    main()