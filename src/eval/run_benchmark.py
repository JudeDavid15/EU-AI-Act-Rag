
from __future__ import annotations

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.legal_reference_router import LegalReferenceRouter
from src.generation.generation import RAGGenerator
from src.guardrails.nli_verifier import NLIConstraintVerifier
import argparse

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "benchmark_results.json"

RETRIEVAL_TOP_N = 15
DEFAULT_RERANK_TOP_N = 3
REFERENCE_RERANK_TOP_N = 6


def normalize_article(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip()).lower()


def base_article(value: str) -> str:
    match = re.search(r"Article\s+(\d+)", str(value), flags=re.IGNORECASE)
    return normalize_article(f"Article {match.group(1)}") if match else ""


def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation file: {QUESTIONS_PATH}")
    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions.json must contain a non-empty 'questions' list")
    return questions


def candidate_article(candidate: Dict[str, Any]) -> str:
    for key in ("article_number", "article", "source_article"):
        value = candidate.get(key)
        if value:
            return str(value)
    return ""


def dedupe_articles(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        key = normalize_article(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def filter_candidates(
    candidates: List[Dict[str, Any]],
    allowed_articles: List[str],
) -> List[Dict[str, Any]]:
    allowed = {normalize_article(a) for a in allowed_articles}
    return [
        c for c in candidates
        if normalize_article(candidate_article(c)) in allowed
    ]


def detected_articles(verification: Dict[str, Any]) -> List[str]:
    return dedupe_articles(
        claim.get("cited_article", "")
        for claim in verification.get("claims", [])
    )


def retrieved_expected_match(
    expected_articles: List[str],
    retrieved_articles: List[str],
) -> bool:
    if not expected_articles:
        return True
    retrieved_base = {base_article(a) for a in retrieved_articles}
    return all(base_article(expected) in retrieved_base for expected in expected_articles)


def expected_articles_match(
    expected_articles: List[str],
    verification: Dict[str, Any],
) -> bool:

    if not expected_articles:
        return True

    detected = detected_articles(verification)
    detected_exact = {normalize_article(a) for a in detected}
    detected_base = {base_article(a) for a in detected}

    for expected in expected_articles:
        expected_norm = normalize_article(expected)
        if "(" in expected_norm:
            if expected_norm not in detected_exact:
                return False
        else:
            if expected_norm not in detected_base:
                return False

    return True


def detect_abstention(answer: str) -> bool:
    text = answer.lower()

    patterns = [
        r"\bnot\s+(?:directly\s+)?(?:explicitly\s+)?(?:stated|specified|provided|contained)\b",
        r"\bdoes\s+not\s+(?:state|specify|provide|contain)\b",
        r"\bno\s+.*\b(?:stated|specified|provided|contained)\b",
        r"\bcannot\s+be\s+determined\b",
        r"\binsufficient\s+(?:information|context)\b",
        r"\bnot\s+enough\s+information\b",
        r"\bthe\s+(?:supplied|provided)\s+context\b.*\bdoes\s+not\b",
        r"\bthere\s+is\s+no\b.*\b(?:stated|specified|provided|contained)\b",
    ]

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def choose_rerank_top_n(query: str) -> int:
    if re.search(r"\bArticle\s+\d+(?:(?:\s*\([^)]+\))+)?\b", query, re.IGNORECASE):
        return REFERENCE_RERANK_TOP_N
    if re.search(
        r"\b(?:fine|fines|penalty|penalties|administrative fines?)\b",
        query,
        re.IGNORECASE,
    ):
        return REFERENCE_RERANK_TOP_N
    return DEFAULT_RERANK_TOP_N


def ensure_exact_paragraphs(
    top_chunks: List[Dict[str, Any]],
    query: str,
    router: LegalReferenceRouter,
) -> List[Dict[str, Any]]:
    result = [dict(c) for c in top_chunks]

    seen = {
        str(c.get("chunk_id", ""))
        for c in result
        if c.get("chunk_id")
    }

    for ref in router.extract_article_references(query):
        match = re.search(
            r"Article\s+(\d+)(?:\s*\((\d+)\))",
            ref,
            re.IGNORECASE,
        )

        if not match:
            continue

        paragraph = match.group(2)
        article = f"Article {match.group(1)}"

        candidates = router.force_article_candidates(article)

        exact = [
            c
            for c in candidates
            if re.match(
                rf"^\s*{re.escape(paragraph)}\.\s+",
                str(c.get("text", "")),
            )
        ]

        for candidate in exact[:1]:
            cid = str(candidate.get("chunk_id", ""))

            if cid and cid not in seen:
                candidate["exact_reference_boost"] = 2000.0
                result.append(candidate)
                seen.add(cid)

    return result

def run_question(
    item: Dict[str, Any],
    retriever: HybridRetriever,
    reranker: Reranker,
    builder: ContextBuilder,
    router: LegalReferenceRouter,
    generator: RAGGenerator,
    verifier: NLIConstraintVerifier,
) -> Dict[str, Any]:
    total_start = time.perf_counter()

    qid = item["id"]
    query = item["question"]
    expected = item.get("expected_articles", [])
    context_articles = item.get("context_articles", [])

    start = time.perf_counter()
    raw_candidates = retriever.search(query=query, top_n=RETRIEVAL_TOP_N)
    retrieval_time = time.perf_counter() - start

    retrieved_articles = dedupe_articles(
        candidate_article(c) for c in raw_candidates
    )

    augmented = router.augment_candidates(query, raw_candidates)
    augmented_articles = dedupe_articles(
        candidate_article(c) for c in augmented
    )

    if context_articles:
        augmented = filter_candidates(augmented, context_articles)
        if not augmented:
            raise RuntimeError(
                f"{qid}: no candidates matched context_articles={context_articles}"
            )

    rerank_top_n = choose_rerank_top_n(query)
    print(
    f"[RERANK INPUT] raw={len(raw_candidates)}, "
    f"augmented={len(augmented)}, "
    f"rerank_top_n={rerank_top_n}"
    )

    start = time.perf_counter()
    top_chunks = reranker.rerank(
        query=query,
        candidates=augmented,
        top_n=rerank_top_n,
    )
    rerank_time = time.perf_counter() - start

    forced_chunks = router.forced_candidates_for_query(query)

    top_chunks = top_chunks + forced_chunks

    top_chunks = ensure_exact_paragraphs(
        top_chunks,
        query,
        router,
    )

    top_chunks.sort(
        key=lambda c: float(c.get("score", 0.0)),
        reverse=True,
    )
    start = time.perf_counter()
    context = builder.build_context(top_chunks)
    context_time = time.perf_counter() - start

    start = time.perf_counter()
    answer = generator.generate_response(
        query=query,
        context_payload=context,
    )
    generation_time = time.perf_counter() - start

    start = time.perf_counter()
    verification = verifier.verify_response(
        generated_response=answer,
        sources=context.get("sources", []),
    )
    nli_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start
    claims = verification.get("claims", [])

    hallucinated = sum(bool(c.get("hallucination_flag")) for c in claims)
    citation_valid = sum(bool(c.get("citation_valid")) for c in claims)
    cited_claims = sum(c.get("cited_article") is not None for c in claims)
    citation_errors = int(verification.get("citation_errors", 0))
    uncited_claims = int(verification.get("uncited_claims", 0))
    abstention_required = bool(item.get("requires_abstention", False))

    abstention_detected = False
    if abstention_required:
        abstention_detected = (
            detect_abstention(answer)
            or bool(verification.get("abstention_claim", False))
            or any(
                str(c.get("status", "")).lower() == "abstention"
                for c in claims
            )
        )
    article_match = expected_articles_match(expected, verification)
    raw_retrieval_match = retrieved_expected_match(
            expected,
            retrieved_articles,
        )
    retrieval_match = retrieved_expected_match(
            expected,
            augmented_articles,
        )


    if abstention_required:
        task_pass = abstention_detected and hallucinated == 0
    else:
        task_pass = (
            bool(verification.get("is_faithful", False))
            and article_match
            and citation_errors == 0
            and uncited_claims == 0
        )

    return {
        "id": qid,
        "category": item.get("category", "unspecified"),
        "question": query,
        "expected_articles": expected,
        "context_articles": context_articles,
        "retrieved_articles": retrieved_articles,
        "retrieved_articles": retrieved_articles,
        "retrieved_expected_match": raw_retrieval_match,
        "raw_retrieval_expected_match": raw_retrieval_match,
        "routed_articles": augmented_articles,
        "routed_expected_match": retrieval_match,
        "forced_candidate_count": len(forced_chunks),
        "detected_articles": detected_articles(verification),
        "expected_articles_matched": article_match,
        "requires_abstention": abstention_required,
        "abstention_detected": abstention_detected,
        "task_pass": task_pass,
        "answer": answer,
        "is_faithful": bool(verification.get("is_faithful", False)),
        "hallucination_rate": float(verification.get("hallucination_rate", 0.0)),
        "claim_count": len(claims),
        "hallucinated_claim_count": hallucinated,
        "citation_valid_claim_count": citation_valid,
        "cited_claim_count": cited_claims,
        "citation_errors": citation_errors,
        "uncited_claims": uncited_claims,
        "claims": claims,
        "timings": {
            "retrieval": round(retrieval_time, 3),
            "reranking": round(rerank_time, 3),
            "context_building": round(context_time, 3),
            "generation": round(generation_time, 3),
            "nli": round(nli_time, 3),
            "total": round(total_time, 3),
        },
        "benchmark_config": {
            "retrieval_top_n": RETRIEVAL_TOP_N,
            "rerank_top_n": rerank_top_n,
        },
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    total_claims = sum(r["claim_count"] for r in results)

    citation_eligible_claims = sum(
        1
        for result in results
        for claim in result.get("claims", [])
        if not (
            str(claim.get("status", "")).lower() == "abstention"
            and claim.get("cited_article") is None
        )
    )

    hallucinated = sum(r["hallucinated_claim_count"] for r in results)
    citation_valid = sum(r["citation_valid_claim_count"] for r in results)
    cited_claims = sum(r["cited_claim_count"] for r in results)
    citation_errors = sum(r["citation_errors"] for r in results)
    uncited = sum(r["uncited_claims"] for r in results)

    faithful = sum(bool(r["is_faithful"]) for r in results)
    task_passes = sum(bool(r.get("task_pass")) for r in results)

    article_tests = [r for r in results if r["expected_articles"]]

    retrieval_passes = sum(
        bool(r["retrieved_expected_match"]) for r in article_tests
    )
    routed_passes = sum(
        bool(r["routed_expected_match"]) for r in article_tests
    )

    article_passes = sum(
        bool(r["expected_articles_matched"]) for r in article_tests
    )
    abstention_tests = [r for r in results if r.get("requires_abstention", False)]
    abstention_passes = sum(bool(r.get("abstention_detected")) for r in abstention_tests)

    total_time = sum(r["timings"]["total"] for r in results)

    return {
        "questions": total,
        "task_passes": task_passes,
        "task_pass_rate": task_passes / total if total else 0.0,
        "faithful_responses": faithful,
        "faithfulness_rate": faithful / total if total else 0.0,
        "total_claims": total_claims,
        "citation_eligible_claims": citation_eligible_claims,

        "hallucinated_claims": hallucinated,
        "hallucination_rate": hallucinated / total_claims if total_claims else 0.0,
        "citation_valid_claims": citation_valid,
        "cited_claims": cited_claims,
        "citation_accuracy_among_cited": citation_valid / cited_claims if cited_claims else 0.0,
        "citation_coverage": (
                    cited_claims / citation_eligible_claims
                    if citation_eligible_claims else 0.0
                ),
        "citation_errors": citation_errors,
        "uncited_claims": uncited,
        "uncited_claim_rate": (
                uncited / citation_eligible_claims
                if citation_eligible_claims else 0.0
            ),
        "retrieval_article_tests": len(article_tests),
        "retrieval_article_passes": retrieval_passes,
       "retrieval_article_recall": (
            retrieval_passes / len(article_tests) if article_tests else 0.0
        ),
        "routed_article_passes": routed_passes,
        "routed_article_recall": (
            routed_passes / len(article_tests) if article_tests else 0.0
        ),
        "expected_article_tests": len(article_tests),
        "expected_article_passes": article_passes,
        "expected_article_coverage": article_passes / len(article_tests) if article_tests else 0.0,
        "abstention_tests": len(abstention_tests),
        "abstention_passes": abstention_passes,
        "abstention_pass_rate": abstention_passes / len(abstention_tests) if abstention_tests else 0.0,
        "average_total_latency_seconds": total_time / total if total else 0.0,
    }


def main() -> None:
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.retrieval.context_builder import ContextBuilder
    from src.retrieval.legal_reference_router import LegalReferenceRouter
    from src.generation.generation import RAGGenerator
    from src.guardrails.nli_verifier import NLIConstraintVerifier

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Run only selected question IDs, e.g. --ids Q01 Q12 Q20"
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Use consolidated preview indexes instead of production indexes",
    )
    args = parser.parse_args()

    questions = load_questions()

    if args.ids:
        wanted = set(args.ids)
        questions = [
            q for q in questions
            if str(q.get("id", "")).upper() in wanted
        ]

        if not questions:
            raise ValueError(
                f"No matching question IDs found: {args.ids}"
            )

    print("=" * 70)
    print("EU AI ACT RAG BENCHMARK — IMPROVED")
    print("=" * 70)
    print(f"Questions: {len(questions)}")

    if args.staging:
        root = Path.cwd()

        retriever = HybridRetriever(
            bm25_index_path=root / "data/processed/bm25_consolidated_preview.pkl",
            qdrant_db_path=root / "data/qdrant_db_consolidated_preview",
            collection_name="eu_ai_act_clauses_preview",
        )

        builder = ContextBuilder(
            processed_json_path=root / "data/processed/chunks_consolidated_preview.json"
        )
    else:
        retriever = HybridRetriever()
        builder = ContextBuilder()

    reranker = Reranker()
    router = LegalReferenceRouter()
    generator = RAGGenerator()
    verifier = NLIConstraintVerifier()

    results: List[Dict[str, Any]] = []

    try:
        for index, item in enumerate(questions, start=1):
            qid = item.get("id", f"Q{index:02d}")
            print("\n" + "=" * 70)
            print(f"[{index}/{len(questions)}] {qid}")
            print(item["question"])
            print("=" * 70)

            try:
                result = run_question(
                    item,
                    retriever,
                    reranker,
                    builder,
                    router,
                    generator,
                    verifier,
                )
                results.append(result)
                print(
                    f"[RESULT] Pass={result['task_pass']} | "
                    f"Faithful={result['is_faithful']} | "
                    f"Hallucinations={result['hallucinated_claim_count']}/"
                    f"{result['claim_count']} | "
                    f"CitationErrors={result['citation_errors']} | "
                    f"Uncited={result['uncited_claims']} | "
                    f"RetrievalExpected={result['retrieved_expected_match']} | "
                    f"Total={result['timings']['total']:.2f}s"
                )
            except Exception as exc:
                print(f"[ERROR] {qid}: {type(exc).__name__}: {exc}")
                results.append({
                    "id": qid,
                    "category": item.get("category", "unspecified"),
                    "question": item["question"],
                    "expected_articles": item.get("expected_articles", []),
                    "context_articles": item.get("context_articles", []),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        retriever.close()

    successful = [r for r in results if "error" not in r]
    summary = summarize(successful)

    output = {
        "benchmark": {
            "name": "EU AI Act RAG Evaluation Suite",
            "version": "2.0",
            "retrieval_top_n": RETRIEVAL_TOP_N,
            "default_rerank_top_n": DEFAULT_RERANK_TOP_N,
            "reference_rerank_top_n": REFERENCE_RERANK_TOP_N,
            "citation_matching": "hierarchical base-Article + exact subsection",
            "standalone_citation_headers": "supported",
            "exact_reference_augmentation": True,
            "title_aware_augmentation": True,
        },
        "summary": summary,
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for label, key, suffix in [
        ("Task pass rate", "task_pass_rate", "%"),
        ("Faithfulness rate", "faithfulness_rate", "%"),
        ("Hallucination rate", "hallucination_rate", "%"),
        ("Citation accuracy (cited)", "citation_accuracy_among_cited", "%"),
        ("Citation coverage", "citation_coverage", "%"),
        ("Uncited claim rate", "uncited_claim_rate", "%"),
        ("Retrieval Article recall", "retrieval_article_recall", "%"),
        ("Expected Article coverage", "expected_article_coverage", "%"),
        ("Abstention pass rate", "abstention_pass_rate", "%"),
    ]:
        print(f"{label:<30} {summary[key] * 100:.1f}{suffix}")

    print(f"{'Task passes':<30} {summary['task_passes']}/{summary['questions']}")
    print(f"{'Faithful responses':<30} {summary['faithful_responses']}/{summary['questions']}")
    print(f"{'Citation errors':<30} {summary['citation_errors']}")
    print(f"{'Average latency':<30} {summary['average_total_latency_seconds']:.2f}s")
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
