from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARTICLE_REF_RE = re.compile(
    r"\bArticle\s+(\d+)(?:(?:\s*\([^)]+\))+)?",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "under", "on",
    "in", "what", "which", "does", "do", "are", "is", "be", "can", "how",
    "with", "by", "according", "regarding", "concerning", "apply", "have",
    "has", "their", "this", "that", "these", "those", "shall", "must",
    "may", "high", "risk", "ai", "systems", "system", "eu", "act",
    "obligations", "requirement", "requirements", "provisions", "rules", "duties",
}


class LegalReferenceRouter:
    def __init__(self, chunks_path: Optional[str] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.chunks_path = Path(
            chunks_path or project_root / "data" / "processed" / "chunks.json"
        )
        self.parents: List[Dict[str, Any]] = []
        self.by_article: Dict[str, Dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _normalize_article(value: str) -> str:
        return re.sub(r"\s+", "", str(value).strip()).lower()

    @staticmethod
    def _base_article(value: str) -> str:
        match = re.search(r"Article\s+(\d+)", str(value), re.IGNORECASE)
        return f"Article {match.group(1)}" if match else ""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}

    @staticmethod
    def _copy_child(child: Dict[str, Any], score: float) -> Dict[str, Any]:
        candidate = dict(child)
        candidate.setdefault("score", score)
        candidate.setdefault("article_number", child.get("article_number", ""))
        candidate.setdefault("title", child.get("title", ""))
        candidate.setdefault("parent_id", child.get("parent_id", ""))
        candidate.setdefault("chunk_id", child.get("chunk_id", ""))
        candidate.setdefault("text", child.get("text", ""))
        return candidate

    def _load(self) -> None:
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {self.chunks_path}. "
                "Pass chunks_path=... when constructing LegalReferenceRouter."
            )

        payload = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw = payload.get("documents") or payload.get("chunks") or payload.get("data")
        else:
            raw = payload

        if not isinstance(raw, list):
            raise ValueError("chunks.json must contain a list of article records.")

        for parent in raw:
            if not isinstance(parent, dict):
                continue
            article = str(parent.get("article_number", "")).strip()
            if not article:
                continue
            self.parents.append(parent)
            self.by_article[self._normalize_article(article)] = parent

    def extract_article_references(self, query: str) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for match in ARTICLE_REF_RE.finditer(query):
            ref = match.group(0).strip()
            key = self._normalize_article(ref)
            if key not in seen:
                seen.add(key)
                result.append(ref)
        return result

    def _article_children(self, article: str) -> List[Dict[str, Any]]:
        parent = self.by_article.get(self._normalize_article(article))
        if not parent:
            return []

        children = parent.get("children", [])
        base_score = 1000.0
        return [
            self._copy_child(child, base_score)
            for child in children
            if isinstance(child, dict) and str(child.get("text", "")).strip()
        ]

    def _title_matches(self, query: str, limit: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        q = self._tokens(query)
        if not q:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for parent in self.parents:
            title = str(parent.get("title", ""))
            full_text = str(parent.get("full_text", ""))

            heading = " ".join(full_text.splitlines()[:4])
            searchable = f"{title} {heading}"
            title_tokens = self._tokens(searchable)
            if not title_tokens:
                continue

            overlap = q & title_tokens
            if not overlap:
                continue

            score = len(overlap) / max(1, len(title_tokens))
            if score >= 0.20 or len(overlap) >= 2:
                scored.append((score, parent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def augment_candidates(
        self,
        query: str,
        candidates: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        merged: List[Dict[str, Any]] = [dict(c) for c in candidates]
        seen = {
            str(c.get("chunk_id", ""))
            for c in merged
            if c.get("chunk_id")
        }

        for ref in self.extract_article_references(query):
            article = self._base_article(ref)
            for child in self._article_children(article):
                cid = str(child.get("chunk_id", ""))
                if cid and cid not in seen:
                    child["legal_reference_boost"] = 1000.0
                    merged.append(child)
                    seen.add(cid)

        for title_score, parent in self._title_matches(query):
            article = str(parent.get("article_number", ""))
            boost = 700.0 + 100.0 * title_score
            for child in self._article_children(article):
                cid = str(child.get("chunk_id", ""))
                if cid and cid not in seen:
                    child["legal_title_boost"] = boost
                    merged.append(child)
                    seen.add(cid)


        lowered = query.lower()
        penalty_intent = bool(
            re.search(r"\b(?:fine|fines|penalt(?:y|ies)|administrative fines?)\b", lowered)
        )
        if penalty_intent:
            for article in ("Article 99", "Article 100"):
                for child in self._article_children(article):
                    cid = str(child.get("chunk_id", ""))
                    if cid and cid not in seen:
                        child["penalty_boost"] = 650.0
                        merged.append(child)
                        seen.add(cid)

        return merged


    def _paragraph_children(self, article: str, paragraphs: Iterable[str], score: float = 2000.0) -> List[Dict[str, Any]]:
        wanted = {str(p) for p in paragraphs}
        out: List[Dict[str, Any]] = []
        for child in self._article_children(article):
            text = str(child.get("text", ""))
            match = re.match(r"^\s*(\d+)\.\s+", text)
            if match and match.group(1) in wanted:
                child["exact_reference_boost"] = score
                out.append(child)
        return out

    def forced_candidates_for_query(self, query: str) -> List[Dict[str, Any]]:

        forced: List[Dict[str, Any]] = []
        seen: set[str] = set()

        TOPIC_ARTICLE_HINTS = {
            "technical documentation": ["Article 11"],
            "record-keeping": ["Article 12"],
            "risk management": ["Article 9"],
            "data and data governance": ["Article 10"],
            "data governance": ["Article 10"],
            "human oversight": ["Article 14"],
            "accuracy": ["Article 15"],
            "robustness": ["Article 15"],
            "cybersecurity": ["Article 15"],
            "providers of high-risk ai systems": ["Article 16"],
            "deployers of high-risk ai systems": ["Article 26"],
            "fundamental rights impact assessment": ["Article 27"],
            "conformity assessment": ["Article 43"],
            "transparency obligations": ["Article 50"],
            "general-purpose ai models": ["Article 53"],
            "systemic risk": ["Article 55"],
        }

        def add(candidates: Iterable[Dict[str, Any]]) -> None:
            for candidate in candidates:
                cid = str(candidate.get("chunk_id", ""))
                if cid and cid not in seen:
                    candidate = dict(candidate)
                    candidate["score"] = max(
                        float(candidate.get("score", 0.0)),
                        5000.0,
                    )
                    forced.append(candidate)
                    seen.add(cid)


        refs = self.extract_article_references(query)
        exact_ref_present = any("(" in ref for ref in refs)

        for ref in refs:
            article = self._base_article(ref)

            match = re.search(
                r"Article\s+\d+(?:\s*\((\d+)\))",
                ref,
                re.IGNORECASE,
            )

            if match:

                add(
                    self._paragraph_children(
                        article,
                        [match.group(1)],
                        score=7000.0,
                    )
                )
            else:

                add(self._article_children(article)[:8])


        lowered = query.lower()

        topic_matches: List[str] = []

        for phrase, articles in sorted(
            TOPIC_ARTICLE_HINTS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if phrase in lowered:
                topic_matches.extend(articles)

        topic_articles: List[str] = []
        topic_seen: set[str] = set()

        for article in topic_matches:
            key = self._normalize_article(article)
            if key not in topic_seen:
                topic_seen.add(key)
                topic_articles.append(article)

        for article in topic_articles:

            add(self._article_children(article)[:8])


        penalty_intent = bool(
            re.search(
                r"\b(?:fine|fines|penalty|penalties|administrative fines?)\b",
                lowered,
            )
        )

        if (
            not exact_ref_present
            and penalty_intent
            and re.search(r"Article\s+5\b", query, re.IGNORECASE)
        ):
            if re.search(r"\boperators?\b", lowered):
                add(
                    self._paragraph_children(
                        "Article 99",
                        ["3"],
                        score=6500.0,
                    )
                )
            else:
                add(
                    self._paragraph_children(
                        "Article 99",
                        ["3"],
                        score=6500.0,
                    )
                )
                add(
                    self._paragraph_children(
                        "Article 100",
                        ["2"],
                        score=6500.0,
                    )
                )

        if (
            not exact_ref_present
            and penalty_intent
            and re.search(r"Article\s+99\b", query, re.IGNORECASE)
        ):
            add(
                self._paragraph_children(
                    "Article 99",
                    ["3", "4", "5"],
                    score=6500.0,
                )
            )

        if (
            not exact_ref_present
            and penalty_intent
            and re.search(r"Article\s+100\b", query, re.IGNORECASE)
        ):
            add(
                self._paragraph_children(
                    "Article 100",
                    ["2", "3"],
                    score=6500.0,
                )
            )


        if (
            re.search(r"Article\s+99\b", query, re.IGNORECASE)
            and re.search(r"\bcommission\b", lowered)
        ):
            add(
                self._paragraph_children(
                    "Article 99",
                    ["1", "2", "9"],
                    score=6200.0,
                )
            )

        return forced
    def force_article_candidates(
        self,
        article: str,
    ) -> List[Dict[str, Any]]:
        return self._article_children(article)
