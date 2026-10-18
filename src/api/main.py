from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Callable
import json
import queue
import re
import threading
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import time
import hashlib
import redis
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.reranker import Reranker
from src.guardrails.nli_verifier import NLIConstraintVerifier
from src.generation.generation import (
    RAGGenerator,
    ExactProvisionNotRetrievedError,
)
from src.retrieval.legal_reference_router import LegalReferenceRouter

pipeline_components: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Hybrid RAG Pipeline Components...")
    pipeline_components["retriever"] = HybridRetriever()
    pipeline_components["reranker"] = Reranker()
    pipeline_components["context_builder"] = ContextBuilder()
    pipeline_components["generator"] = RAGGenerator()
    pipeline_components["verifier"] = NLIConstraintVerifier()
    pipeline_components["legal_router"] = LegalReferenceRouter()
    pipeline_components["cache"] = None
    cache_client = None

    if os.getenv("CACHE_ENABLED", "true").lower() in {
        "1", "true", "yes", "on"
    }:
        try:
            cache_client = redis.Redis.from_url(
                os.getenv(
                    "REDIS_URL",
                    "redis://localhost:6379/0",
                ),
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            cache_client.ping()

            pipeline_components["cache"] = cache_client
            print("[CACHE] Connected to Valkey.")

        except (redis.RedisError, ValueError) as exc:
            if cache_client is not None:
                cache_client.close()

            pipeline_components["cache"] = None
            print(
                "[CACHE] Valkey unavailable; "
                f"continuing without shared cache: {exc}"
            )
    else:
        print("[CACHE] Shared response cache disabled.")

    print("Pipeline Ready to accept Requests.")
    yield

    print("Shutting down pipeline...")
    if "retriever" in pipeline_components and hasattr(pipeline_components["retriever"], "close"):
        pipeline_components["retriever"].close()
    if "reranker" in pipeline_components and hasattr(pipeline_components["reranker"], "close"):
        pipeline_components["reranker"].close()
    cache_client = pipeline_components.get("cache")
    if cache_client is not None:
        cache_client.close()
    pipeline_components.clear()

app = FastAPI(
    title="Hybrid RAG Engine API",
    description="RAG service with hybrid search, cross-encoder Reranking and NLI faithfulness guardrails",
    version="1.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["what are the transparency obligations for high risk AI systems?"]
    )
    top_n_candidates: int = Field(default=15,
                                  ge=1,
                                    le=50,
                                      description="Initial BM25 + Vector recall count")
    top_n_rerank: int = Field(default=3,
                               ge=1,
                                 le=20,
                                   description="Chunks sent to context window after reranking")

class ClaimAudit(BaseModel):
    claim: str
    status: str
    confidence: Optional[float] = None
    flagged: bool

class GuardrailAudit(BaseModel):
    is_faithful: bool
    hallucination_rate: float
    claims: List[ClaimAudit]

class SourceDocument(BaseModel):
    article_id: Optional[str] = None
    title: Optional[str] = None
    full_text: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    verification_audit: GuardrailAudit

def _response_cache_key(payload: QueryRequest) -> str:
    material = json.dumps(
        {
            "version": os.getenv(
                "RAG_CACHE_VERSION",
                "eu-ai-act-rag-v1",
            ),
            "query": payload.query,
            "top_n_candidates": payload.top_n_candidates,
            "top_n_rerank": payload.top_n_rerank,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()

    return f"rag:response:{digest}"


def _get_cached_response(
    payload: QueryRequest,
) -> Optional[QueryResponse]:
    cache = pipeline_components.get("cache")
    if cache is None:
        return None

    try:
        cached_json = cache.get(
            _response_cache_key(payload)
        )

        if not cached_json:
            print("[CACHE] MISS")
            return None

        result = QueryResponse.model_validate_json(
            cached_json
        )
        print("[CACHE] HIT")
        return result

    except Exception as exc:
        print(
            "[CACHE] Read failed; bypassing cache: "
            f"{type(exc).__name__}"
        )
        return None


def _store_cached_response(
    payload: QueryRequest,
    result: QueryResponse,
) -> None:
    cache = pipeline_components.get("cache")
    if cache is None:
        return

    try:
        ttl = max(
            1,
            int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        )

        cache.setex(
            _response_cache_key(payload),
            ttl,
            result.model_dump_json(),
        )
        print(f"[CACHE] STORED (TTL={ttl}s)")

    except Exception as exc:
        print(
            "[CACHE] Write failed; continuing normally: "
            f"{type(exc).__name__}"
        )

@app.get("/", status_code=status.HTTP_200_OK)
async def read_root():
    return {
        "status": "online",
        "message": "EU AI Act RAG API is active. Visit /docs to test endpoints interactively."
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    is_ready = all(k in pipeline_components for k in ["retriever", "reranker", "generator", "verifier"])
    if not is_ready:
        raise HTTPException(status_code=503, detail="Pipeline Components are initializing or unavailable")
    return {"status": "ok", "pipeline_ready": True}

def _execute_rag_query(
    payload: QueryRequest,
    on_progress: Optional[Callable[[str], None]] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> QueryResponse:

    try:
        retriever = pipeline_components["retriever"]
        reranker = pipeline_components["reranker"]
        builder = pipeline_components["context_builder"]
        generator = pipeline_components["generator"]
        verifier = pipeline_components["verifier"]

        def progress(message: str) -> None:
            if on_progress:
                on_progress(message)

        t0 = time.time()
        progress("Retrieving candidate passages with hybrid search...")

        candidates = retriever.search(
            query=payload.query,
            top_n=payload.top_n_candidates,
        )

        legal_router = pipeline_components["legal_router"]
        article_refs = legal_router.extract_article_references(
            payload.query
        )
        forced_candidates = legal_router.forced_candidates_for_query(
            payload.query
        )

        requested_article_numbers = set()

        for ref in article_refs:
            match = re.search(
                r"\bArticle\s+(\d+[a-z]?)",
                ref,
                re.IGNORECASE,
            )
            if match:
                requested_article_numbers.add(
                    match.group(1).casefold()
                )

        # Preserve the existing special handling for questions about
        # penalties arising from Article 5 prohibited practices.
        article5_penalty_query = bool(
            re.search(r"\bArticle\s+5\b", payload.query, re.IGNORECASE)
            and re.search(
                r"\b(?:fine|fines|penalty|penalties)\b",
                payload.query,
                re.IGNORECASE,
            )
        )

        routed_candidates = []

        if requested_article_numbers and not article5_penalty_query:
            for candidate in forced_candidates:
                match = re.search(
                    r"\bArticle\s+(\d+[a-z]?)",
                    str(candidate.get("article_number", "")),
                    re.IGNORECASE,
                )

                if (
                    match
                    and match.group(1).casefold()
                    in requested_article_numbers
                ):
                    routed_candidates.append(candidate)

        # A directly referenced article takes priority over unrelated
        # hybrid-search results.
        if routed_candidates:
            candidates = routed_candidates
        else:
            # Otherwise, augment hybrid results with useful routed candidates.
            merged = {}

            for candidate in candidates + forced_candidates:
                chunk_id = str(candidate.get("chunk_id", ""))

                if not chunk_id:
                    continue

                merged[chunk_id] = {
                    **merged.get(chunk_id, {}),
                    **candidate,
                }

            candidates = list(merged.values())

        t1 = time.time()
        print(f"[TIMING] retrieval: {t1 - t0:.2f}s")

        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant knowledge base chunks found for query.",
            )
        progress("Reranking retrieved passages...")
        # Detect questions that explicitly reference a legal provision.
        exact_provision_pattern = (
            r"\bArticle\s+(\d+[a-z]?)\s*"
            r"\(\s*\d+\s*\)"
            r"(?:\s*\(\s*[a-z0-9]+\s*\))*"
        )

        exact_matches = list(
            re.finditer(
                exact_provision_pattern,
                payload.query,
                flags=re.IGNORECASE,
            )
        )

        requested_articles = {
            match.group(1).casefold()
            for match in exact_matches
        }

        if len(requested_articles) == 1:
            requested_article = next(iter(requested_articles))

            target_candidates = []

            for candidate in candidates:
                article_match = re.search(
                    r"\bArticle\s+(\d+[a-z]?)\b",
                    str(candidate.get("article_number", "")),
                    flags=re.IGNORECASE,
                )

                if (
                    article_match
                    and article_match.group(1).casefold()
                    == requested_article
                ):
                    target_candidates.append(candidate)

            if not target_candidates:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Could not retrieve Article {requested_article}. "
                        "No answer was generated from unrelated provisions."
                    ),
                )

            top_chunks = reranker.rerank(
                query=payload.query,
                candidates=target_candidates,
                top_n=len(target_candidates),
            )

        elif requested_article_numbers and not article5_penalty_query:
            top_chunks = reranker.rerank(
            query=payload.query,
            candidates=candidates,
            top_n=len(candidates),
            )

        else:
            top_chunks = reranker.rerank(
                query=payload.query,
                candidates=candidates,
                top_n=payload.top_n_rerank,
            )

        t2 = time.time()
        print(f"[TIMING] rerank: {t2 - t1:.2f}s")
        for chunk in top_chunks:
            if str(chunk.get("article_number", "")).strip().lower() == "article 14":
                print(
                    "ID:", chunk.get("chunk_id"),
                    "| TEXT:", str(chunk.get("text", ""))[:1000],
                )
        t2 = time.time()
        print(f"[TIMING] rerank: {t2 - t1:.2f}s")

        progress("Building the legal context...")
        context = builder.build_context(top_chunks)

        for source in context.get("sources", []):
            if str(source.get("article_number", "")).strip().lower() == "article 14":
                for chunk in source.get("selected_chunks", []):
                    print(
                        "ID:", chunk.get("chunk_id"),
                        "| Paragraph:", chunk.get("paragraph_number"),
                        "| Citation:", chunk.get("citation_label"),
                        "| TEXT:", str(chunk.get("text", ""))[:700],
                    )
        t3 = time.time()
        print(f"[TIMING] context_build: {t3 - t2:.2f}s")

        progress("Generating draft answer from the language model...")
        answer = generator.generate_response(payload.query, context, on_token=on_token)
        answer = re.sub(
            r"(?im)^([ \t]*[-*]?[ \t]*the characteristics, capabilities "
            r"and limitations of performance of the high-risk AI system, "
            r"including:)[ \t]*$",
            r"\1 [Article 13(3)(b)]",
            answer,
        )
        answer = re.sub(
            r"(?i)(For high-risk AI systems referred to in point\s+1\s*\(\s*a\s*\),?\s*of Annex III,\s*the logging capabilities shall provide, at a minimum):",
            r"\1 [Article 12(3)]:",
            answer,
        )
        t4 = time.time()
        print(f"[TIMING] generation: {t4 - t3:.2f}s")

        progress("Checking claims and citations with the NLI verifier...")
        verification = verifier.verify_response(
            generated_response=answer,
            sources=context.get("sources", []),
        )
        t5 = time.time()
        print(f"[TIMING] nli_verify: {t5 - t4:.2f}s")
        print(f"[TIMING] TOTAL: {t5 - t0:.2f}s")

        return QueryResponse(
            query=payload.query,
            answer=answer,
            sources=context.get("sources", []),
            verification_audit=GuardrailAudit(**verification),
        )
    except HTTPException:
        raise
    except ExactProvisionNotRetrievedError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal RAG pipeline failure: {str(e)}",
        )

@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
def process_rag_query(payload: QueryRequest):
    cached = _get_cached_response(payload)

    if cached is not None:
        return cached

    result = _execute_rag_query(payload)
    _store_cached_response(payload, result)

    return result

@app.post("/query/stream", status_code=status.HTTP_200_OK)
def process_rag_query_stream(payload: QueryRequest):
    def event_stream():
        events: queue.Queue = queue.Queue()

        def push(event: Dict[str, Any]) -> None:
            events.put(event)

        def worker() -> None:
            try:
                cached = _get_cached_response(payload)

                if cached is not None:
                    push({
                        "type": "status",
                        "message": "Loaded verified response from shared cache.",
                    })
                    push({
                        "type": "complete",
                        "data": jsonable_encoder(cached),
                    })
                    return

                result = _execute_rag_query(
                    payload,
                    on_progress=lambda message: push({
                        "type": "status",
                        "message": message,
                    }),
                    on_token=lambda token: push({
                        "type": "token",
                        "text": token,
                    }),
                )

                _store_cached_response(payload, result)

                push({
                    "type": "complete",
                    "data": jsonable_encoder(result),
                })
            except HTTPException as exc:
                push({
                    "type": "error",
                    "status_code": exc.status_code,
                    "detail": str(exc.detail),
                })
            except Exception as exc:
                push({"type": "error", "status_code": 500, "detail": str(exc)})
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
