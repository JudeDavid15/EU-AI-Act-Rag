from typing import List,Dict,Any
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME="BAAI/bge-reranker-large"

class Reranker:
    def __init__(self,model_name:str=RERANKER_MODEL_NAME):
        print(f"Loading CrossEncoder model {model_name}")
        self.model=CrossEncoder(model_name)
    def rerank(
        self,query:str,candidates:List[Dict[str,Any]],top_n:int=3
    )->List[Dict[str,Any]]:
        if not candidates:
            return []
        pairs=[[query,candidate["text"]] for candidate in candidates]

        scores=self.model.predict(pairs)
        for candidate,score in zip(candidates,scores):
            candidate["rerank_score"]=float(score)
        reranked=sorted(candidates,key=lambda x:x["rerank_score"],reverse=True)
        return reranked[:top_n]


if __name__=="__main__":
    from src.retrieval.hybrid_search import HybridRetriever
    query="what are the transperancy obligations for high risk in AI system"
    with HybridRetriever() as retriever:
        candidates=retriever.search(query=query,top_n=15)
    reranker=Reranker()
    final_results=reranker.rerank(query=query,candidates=candidates,top_n=3)
    print(f"\nQuery: {query}")
    print(f"Top {len(final_results)} Reranked Results")

    for r in final_results:
        print(f"[{r['article_number']}] [{r['title']}] (Rerank Score: {r['rerank_score']:.4f})")
        print(f"Snippet: {r['text'][:140]}...\n")
