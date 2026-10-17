from __future__ import annotations

import json
from pathlib import Path

import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


fake_ollama = types.ModuleType("ollama")
fake_ollama.Client = object
fake_ollama.chat = lambda *args, **kwargs: None
sys.modules.setdefault("ollama", fake_ollama)

from src.guardrails.nli_verifier import NLIConstraintVerifier
from src.generation.generation import RAGGenerator
from src.retrieval.legal_reference_router import LegalReferenceRouter
from src.eval.run_benchmark import (
    detect_abstention,
    expected_articles_match,
    normalize_article,
    retrieved_expected_match,
)


def make_verifier() -> NLIConstraintVerifier:
    verifier = NLIConstraintVerifier.__new__(NLIConstraintVerifier)
    verifier.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    return verifier


def test_standalone_citation_is_attached_to_following_claim() -> None:
    verifier = make_verifier()
    claims = verifier._extract_claims_with_citations(
        "[Article 100(5)]\n"
        "The rights of defence of the parties concerned shall be fully respected in the proceedings."
    )
    assert claims == [(
        "The rights of defence of the parties concerned shall be fully respected in the proceedings.",
        "Article 100(5)",
    )]


def test_inline_citation_is_preserved_exactly() -> None:
    verifier = make_verifier()
    claims = verifier._extract_claims_with_citations(
        "High-risk AI systems shall achieve an appropriate level of accuracy [Article 15(1)]."
    )
    assert claims == [
        ("High-risk AI systems shall achieve an appropriate level of accuracy.", "Article 15(1)")
    ]




def test_generator_moves_standalone_citation_inline() -> None:
    assert RAGGenerator._normalize_standalone_citations(
        "[Article 100(5)]\nThe rights of defence shall be fully respected."
    ) == "The rights of defence shall be fully respected. [Article 100(5)]"


def test_generator_does_not_duplicate_inline_citation() -> None:
    source = "The rights of defence shall be fully respected [Article 100(5)]."
    assert RAGGenerator._normalize_standalone_citations(source) == source

def test_exact_provision_indexing_supports_nested_citations() -> None:
    verifier = make_verifier()
    provisions = verifier._split_article_provisions(
        "Article 5",
        "Article 5\nThe following AI practices shall be prohibited:\n"
        "1. The following practices are prohibited:\n"
        "(a) manipulating behaviour through subliminal techniques;\n"
        "(c) social scoring leading to either or both of the following:\n"
        "(i) detrimental treatment in unrelated social contexts;\n"
        "(ii) unjustified or disproportionate treatment;",
    )
    assert "article5(1)" in provisions
    assert "article5(1)(a)" in provisions
    assert "article5(1)(c)" in provisions
    assert "article5(1)(c)(i)" in provisions




def test_nested_roman_labels_are_not_promoted_to_top_level_letters() -> None:
    verifier = make_verifier()
    provisions = verifier._split_article_provisions(
        "Article 5",
        "Article 5\n1. The following AI practices shall be prohibited:\n"
        "(c) social scoring leading to either or both of the following:\n"
        "(i) detrimental treatment in unrelated social contexts;\n"
        "(ii) unjustified or disproportionate treatment;\n"
        "(d) risk assessment based on profiling;",
    )
    assert "article5(1)(c)" in provisions
    assert "article5(1)(c)(i)" in provisions
    assert "article5(1)(c)(ii)" in provisions
    assert "article5(1)(i)" not in provisions

def test_base_article_matching_is_hierarchical() -> None:
    verification = {
        "claims": [
            {"cited_article": "Article 9(5)"},
        ]
    }
    assert expected_articles_match(["Article 9"], verification)




def test_expected_article_matching_accepts_nested_reference_for_base_article() -> None:
    verification = {"claims": [{"cited_article": "Article 10(2)(f)"}]}
    assert expected_articles_match(["Article 10"], verification)

def test_exact_article_matching_is_not_hierarchical() -> None:
    verification = {
        "claims": [
            {"cited_article": "Article 99(4)"},
        ]
    }
    assert not expected_articles_match(["Article 99(5)"], verification)


def test_retrieval_matching_uses_base_article() -> None:
    assert retrieved_expected_match(
        ["Article 100(5)"],
        ["Article 100", "Article 99"],
    )


def test_abstention_detection_catches_context_limitation() -> None:
    assert detect_abstention(
        "The supplied context does not specify an administrative fine for Article 9."
    )


def test_article_normalization() -> None:
    assert normalize_article(" Article 99 ( 4 ) ") == "article99(4)"


def test_router_extracts_exact_references(tmp_path: Path) -> None:
    chunks = [
        {
            "parent_id": "art_100",
            "article_number": "Article 100",
            "title": "Administrative fines on Union institutions, bodies, offices and agencies",
            "children": [
                {"chunk_id": "art_100_p2", "parent_id": "art_100", "article_number": "Article 100", "title": "Article 100", "text": "2. Non-compliance with prohibited AI practices shall be subject to administrative fines."},
                {"chunk_id": "art_100_p5", "parent_id": "art_100", "article_number": "Article 100", "title": "Article 100", "text": "5. The rights of defence shall be respected."},
            ],
        },
        {
            "parent_id": "art_12",
            "article_number": "Article 12",
            "title": "Record keeping",
            "children": [
                {"chunk_id": "art_12_p2", "parent_id": "art_12", "article_number": "Article 12", "title": "Article 12", "text": "1. Records shall be automatically generated and kept."},
            ],
        },
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")

    router = LegalReferenceRouter(chunks_path=str(path))
    assert router.extract_article_references("What is Article 100(5)?") == ["Article 100(5)"]

    forced = router.forced_candidates_for_query("What is Article 100(5)?")
    assert any(c["chunk_id"] == "art_100_p5" for c in forced)

    rescued = router.augment_candidates("What record keeping obligations apply?", [])
    assert any(c["chunk_id"] == "art_12_p2" for c in rescued)



def test_generator_derives_exact_penalty_citation_targets() -> None:
    assert RAGGenerator._citation_targets(
        "What is the exact fine for prohibited AI practices under Article 100(2)?"
    ) == ["Article 100(2)"]


def test_generator_derives_multi_provision_article99_targets() -> None:
    assert RAGGenerator._citation_targets(
        "What administrative fines can apply to operators for other specified infringements under Article 99?"
    ) == ["Article 99(4)", "Article 99(5)"]


def test_generator_derives_article99_commission_targets() -> None:
    assert RAGGenerator._citation_targets(
        "According to Article 99, can the Commission itself impose the penalties laid down by Member States?"
    ) == ["Article 99(1)", "Article 99(2)"]


def test_abstention_claim_is_classified_without_nli_entailment() -> None:
    verifier = make_verifier()
    assert verifier.is_abstention_claim(
        "The administrative fine is not explicitly stated in Article 9."
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
