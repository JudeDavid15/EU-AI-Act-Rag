import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List,Dict,Any
from qdrant_client import QdrantClient

PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
QDRANT_DB_PATH=PROJECT_ROOT/"data"/"qdrant_db"
BM25_INDEX_PATH=PROJECT_ROOT/"data"/"processed"/"bm25.pkl"

COLLECTION_NAME="eu_ai_act_clauses"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"

class HybridRetriever:
    def __init__(self):
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError("BM25 index missing. build indices first!")
        with open(BM25_INDEX_PATH,"rb") as f:
            data=pickle.load(f)
            self.bm25=data["bm25"]
            self.bm25_chunks=data["chunks"]
        self.encoder=SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.qdrant=QdrantClient(path=str(QDRANT_DB_PATH))
    def _sparse_search(self,query:str,top_k:int=30)->List[Dict[str,Any]]:
        tokenized_query=query.lower().split()
        scores=self.bm25.get_scores(tokenized_query)

        top_indices=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]
        results=[]
        for idx in top_indices:
            if scores[idx]<=0:
                continue
            chunk=self.bm25_chunks[idx]
            results.append({
                "chunk_id":chunk["chunk_id"],
                "text":chunk["text"],
                "article_number":chunk["article_number"],
                "title":chunk["title"],
                "parent_id":chunk["parent_id"],
                "score":float(scores[idx])
            })
        return results

    def _dense_search(self,query:str,top_k:int=30)->List[Dict[str,Any]]:
        query_vector=self.encoder.encode(query,normalize_embeddings=True).tolist()

        search_result=self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )

        results=[]
        for hit in search_result.points:
            payload=hit.payload
            results.append({
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "article_number": payload["article_number"],
                "title": payload["title"],
                "parent_id": payload["parent_id"],
                "score": float(hit.score)
            })
        return results

    def reciprocal_rank_fusion(self,
                               sparse_results:List[Dict[str,Any]],
                               dense_results:List[Dict[str,Any]],
                               k:int=60,
                               top_n:int=10,
                               )->List[Dict[str,Any]]:
        rrf_scores:Dict[str,float]={}
        chunk_map:Dict[str,Dict[str,any]]={}
        for rank,item in enumerate(sparse_results,start=1):
            cid=item["chunk_id"]
            rrf_scores[cid]=rrf_scores.get(cid,0.0)+(1.0/(k+rank))
            chunk_map[cid]=item

        for rank,item in enumerate(dense_results,start=1):
            cid=item["chunk_id"]
            rrf_scores[cid]=rrf_scores.get(cid,0.0)+(1.0/(k+rank))
            chunk_map[cid]=item
        sorted_cid=sorted(rrf_scores.keys(),key=lambda cid:rrf_scores[cid],reverse=True)[:top_n]

        fused_results=[]
        for cid in sorted_cid:
            item=chunk_map[cid]
            item["rrf_score"]=rrf_scores[cid]
            fused_results.append(item)
        return fused_results

    def search(self,query:str,top_n:int=10)->List[Dict[str,Any]]:
        spares_res=self._sparse_search(query,top_k=30)
        dense_res=self._dense_search(query,top_k=30)
        return self.reciprocal_rank_fusion(spares_res,dense_res,k=60,top_n=top_n)
    def close(self):
        if hasattr(self, "qdrant") and self.qdrant is not None:
            self.qdrant.close()
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ =="__main__":
    retriever=HybridRetriever()
    query="what are the transparency obligations for high risk AI systems?"
    results=retriever.search(query=query,top_n=3)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} merged results: \n")

    for r in results:
        print(f"[{r['article_number']}] {r['title']} (RRF Score: {r['rrf_score']:.4f})")
        print(f"Snippet: {r['text'][:120]}...\n")
