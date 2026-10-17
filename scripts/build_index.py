import json
import pickle
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PROJECT_ROOT=Path(__file__).resolve().parent.parent
PROCESSED_JSON_PATH=PROJECT_ROOT/"data"/"processed"/"chunks.json"
QDRANT_DB_PATH=PROJECT_ROOT/"data"/"qdrant_db"
BM25_INDEX_PATH=PROJECT_ROOT/"data"/"processed"/"bm25.pkl"

COLLECTION_NAME="eu_ai_act_clauses"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"


def tokenize(text:str)->list[str]:
    return text.lower().split()

def build_indices():
    if not PROCESSED_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing data at {PROCESSED_JSON_PATH}. Run parser first")
    print("loading processed chunks")
    with open(PROCESSED_JSON_PATH,'r',encoding='utf-8') as f:
        parents=json.load(f)
    all_children=[]
    for p in parents:
        for child in p.get('children',[]):
            all_children.append(child)
    print(f"total child clauses to index:{len(all_children)}")
    if not all_children:
        print("Error: No child clauses found in chunks.json")
    print("building bm25 sparse index")
    corpus_tokens=[tokenize(child['text']) for child in all_children]
    bm25=BM25Okapi(corpus_tokens)

    BM25_INDEX_PATH.parent.mkdir(parents=True,exist_ok=True)
    with open(BM25_INDEX_PATH,'wb') as f:
        pickle.dump({"bm25":bm25,"chunks":all_children},f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

    print(f"loading embedding model {EMBEDDING_MODEL_NAME}....")
    model=SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts_to_embed=[
        f"{child['article_number']}-{child['title']}:{child['text']}"
        for child in all_children
    ]
    print("Generating dense vector embedding....")
    embeddings=model.encode(texts_to_embed,show_progress_bar=True,normalize_embeddings=True)

    print(f"Initalizing qdrant DB at {QDRANT_DB_PATH}.......")
    client=QdrantClient(path=str(QDRANT_DB_PATH))

    vector_size=embeddings.shape[1]
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size,distance=Distance.COSINE)
    )

    points=[]
    for idx,(child,vector) in enumerate(zip(all_children,embeddings)):
        points.append(
            PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={
                "chunk_id":child["chunk_id"],
                "parent_id":child["parent_id"],
                "article_number":child["article_number"],
                "title":child["title"],
                "text":child["text"]
            }
            )
        )
    client.upsert(collection_name=COLLECTION_NAME,points=points)
    print(f"Successfully indexed {len(points)} points in qdrant collection : {COLLECTION_NAME}")

if __name__=="__main__":
    build_indices()
