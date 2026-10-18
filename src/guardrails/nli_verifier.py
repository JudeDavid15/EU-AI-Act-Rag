from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

CITATION_RE = re.compile(
    r"\[\s*(Article\s+\d+(?:(?:\s*\([^)]+\))+)?)\s*\]",
    re.IGNORECASE,
)
CITATION_ONLY_RE = re.compile(
    r"^\s*\[\s*Article\s+\d+(?:(?:\s*\([^)]+\))+)?\s*\]\s*$",
    re.IGNORECASE,
)

class NLIConstraintVerifier:
    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        print(
            f"Loading NLI Guardrail model '{model_name}' on {device}..."
        )
        self.model = CrossEncoder(model_name, device=device)

        config = getattr(self.model.model, "config", None)
        if config and hasattr(config, "id2label"):
            self.id2label = {
                int(k): v.lower()
                for k, v in config.id2label.items()
            }
        else:
            self.id2label = {
                0: "contradiction",
                1: "entailment",
                2: "neutral",
            }

    def _is_evaluable_claim(self, claim: str) -> bool:
        ignore_patterns = [
            r"^the .*(?:obligations?|requirements?|fines?|penalties?|procedures?)"
            r".*\b(?:are|include|consist of)\s*:?$",
            r"^the .+ (?:are|include|consist of) as follows:?$",
            r"^in summary",
            r"^overall[, ]",
            r"^the following are",
            r"^in addition to the above",
            r"^key requirements?:",
            r"^here are the",
            r"^the main requirements are as follows",
            r"^the penalties for$",
            r"^transparency obligations$",
            r"^compliance requirements$",
            r"^key transparency requirements$",
            r"^additional transparency requirements$",
            r"^penalties for$",
            r"^prohibited ai practices$",
            r"^according to the legal context[:.]?$",
            r"^according to the supplied context[:.]?$",
            r"^legal context[:.]?$",
            r"^user query[:.]?$",
        ]

        value = claim.strip()
        if not value:
            return False

        lowered = value.lower()
        if any(re.search(p, lowered) for p in ignore_patterns):
            return False

        if len(value.split()) < 5:
            return False
        if value.endswith(":") and len(value.split()) < 12:
            return False
        if re.search(r"(?:,|\b(?:and|or|the|which|that|including|as)\s*)$", value, flags=re.IGNORECASE):
            return False

        return True

    def _extract_claims_with_citations(
        self,
        text: str,
    ) -> List[Tuple[str, Optional[str]]]:

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        extracted: List[Tuple[str, Optional[str]]] = []
        pending_citation: Optional[str] = None

        for raw_line in lines:
            line = raw_line

            if line.startswith("#"):
                continue
            if line.startswith("**") and line.endswith("**"):
                continue

            if CITATION_ONLY_RE.fullmatch(line):
                match = CITATION_RE.search(line)
                pending_citation = match.group(1).strip() if match else None
                continue

            clean_line = re.sub(r"^\s*[*•\d]+[.)-]?\s*", "", line).strip()
            clean_line = re.sub(r"\*\*|__|#+", "", clean_line).strip()

            if CITATION_ONLY_RE.fullmatch(clean_line):
                match = CITATION_RE.search(clean_line)
                pending_citation = (
                    match.group(1).strip()
                    if match
                    else None
                )
                continue

            inline_matches = list(CITATION_RE.finditer(clean_line))
            cited_article: Optional[str] = None

            if inline_matches:

                cited_article = inline_matches[-1].group(1).strip()
                claim_text = CITATION_RE.sub("", clean_line).strip(" :-")
                claim_text = re.sub(r"\s+([.!?,;:])", r"\1", claim_text)
            else:
                claim_text = clean_line
                if pending_citation:
                    cited_article = pending_citation
                    pending_citation = None

            claim_text = re.sub(
                r"^According to Article\s+\d+(?:(?:\([^)]+\))+)(?:\s+of the EU AI Act)?[:,]?\s*",
                "",
                claim_text,
                flags=re.IGNORECASE,
            )

            if not claim_text:
                continue
            if not self._is_evaluable_claim(claim_text):
                continue

            extracted.append((claim_text, cited_article))

        return extracted

    @staticmethod
    def is_abstention_claim(claim: str) -> bool:
        lowered = claim.lower().strip()
        patterns = [
            r"\bnot\s+(?:directly\s+)?(?:explicitly\s+)?(?:stated|specified|provided|contained)\b",
            r"\bdoes\s+not\s+(?:explicitly\s+)?(?:state|specify|contain)\s+(?:the\s+)?(?:requested\s+)?(?:information|details?|amount|fine|fines|penalty|penalties|sanction|sanctions|answer|provision)\b",
            r"\bdoes\s+not\s+provide\s+(?:any\s+)?(?:information|details?|amount|fine|fines|penalty|penalties|sanction|sanctions|answer)\b",
            r"\bno\s+.*\b(?:stated|specified|provided|contained)\b",
            r"\bcannot\s+be\s+determined\b",
            r"\binsufficient\s+(?:information|context)\b",
            r"\bno\s+.*\b(?:fine|fines|penalty|penalties|amount)\b.*\b(?:stated|specified|provided|contained)\b",
        ]
        return any(re.search(p, lowered) for p in patterns)

    @staticmethod
    def _base_article_number(citation: str) -> str:
        match = re.search(r"Article\s+(\d+)", citation, flags=re.IGNORECASE)
        return f"Article {match.group(1)}".lower() if match else ""

    @staticmethod
    def _citation_key(citation: str) -> str:
        return re.sub(r"\s+", "", citation.strip()).lower() if citation else ""
    @staticmethod
    def _normalize_claim_for_nli(
        claim: str
    ) -> str:


        normalized = claim.strip()

        prefixes = [
            r"^according to the eu ai act[:,]?\s*",
            r"^under the eu ai act[:,]?\s*",
            r"^the eu ai act requires[:,]?\s*",
            r"^the eu ai act states[:,]?\s*",
            r"^the regulation requires[:,]?\s*",
            r"^the regulation states[:,]?\s*",
            r"^under article \d+(?:\([^)]+\))*[:,]?\s*",
            r"^article \d+(?:\([^)]+\))* provides that\s*",
            r"^article \d+(?:\([^)]+\))* requires that\s*",
            r"^article \d+(?:\([^)]+\))* states that\s*",
            r"^providers?\s+of\s+high-risk\s+ai\s+systems\s+"
            r"(?:shall|must|should)\s+",
            r"^additionally,\s*",
            r"^furthermore,\s*",
            r"^moreover,\s*",
            r"^it is also worth noting that\s*",
            r"^it is worth noting that\s*",
            r"^however,\s*",
            r"^providers?\s+of\s+general-purpose\s+ai\s+models\s+"
            r"with\s+systemic\s+risk\s+"
            r"(?:shall|must|should)\s+ensure\s+that\s+",
            r"^also,\s*",
        ]

        for pattern in prefixes:
            normalized = re.sub(
                pattern,
                "",
                normalized,
                flags=re.IGNORECASE,
            )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

    @classmethod
    def _has_exact_text_support(
        cls,
        claim: str,
        source_texts: List[str],
    ) -> bool:
        normalized_claim = cls._normalize_for_exact_match(
            cls._normalize_claim_for_nli(claim)
        )

        if len(normalized_claim.split()) < 8:
            return False

        for source_text in source_texts:
            normalized_source = cls._normalize_for_exact_match(
                source_text
            )

            if normalized_claim in normalized_source:
                return True
        return False


    @classmethod
    def _has_lexical_legal_support(
        cls,
        claim: str,
        source_texts: List[str],
    ) -> bool:
        stopwords = {
            "the", "a", "an", "of", "to", "and", "or", "that",
            "shall", "must", "be", "is", "are", "with", "for",
            "on", "in", "by", "as", "them", "those", "their",
            "this", "these", "from", "such",
        }

        def tokens(text: str) -> set[str]:
            normalized = cls._normalize_for_exact_match(text)
            return {
                token
                for token in normalized.split()
                if token not in stopwords
                and len(token) >= 3
            }

        claim_tokens = tokens(claim)

        if len(claim_tokens) < 8:
            return False

        for source_text in source_texts:
            source_tokens = tokens(source_text)

            if not source_tokens:
                continue

            overlap = len(claim_tokens & source_tokens) / len(claim_tokens)


            if overlap >= 0.85:
                return True

        return False

    @staticmethod
    def _normalize_source_text(text: str) -> str:
        value = text or ""
        value = value.replace("\\r\\n", "\n")
        value = value.replace("\\n", "\n")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value.strip()

    def _split_article_provisions(
        self,
        article_number: str,
        full_text: str,
    ) -> Dict[str, str]:

        normalized = self._normalize_source_text(full_text)

        lines = [
            line.strip()
            for line in normalized.split("\n")
            if line.strip()
        ]

        paragraph_pattern = re.compile(r"^(\d+)\.\s+(.*)$")

        paragraphs: List[Tuple[str, List[str]]] = []
        current_number: Optional[str] = None
        current_lines: List[str] = []

        def flush_paragraph() -> None:
            nonlocal current_number, current_lines

            if current_number is not None and current_lines:
                paragraphs.append(
                    (
                        current_number,
                        current_lines,
                    )
                )

            current_number = None
            current_lines = []


        for line in lines:
            match = paragraph_pattern.match(line)

            if match:
                flush_paragraph()

                current_number = match.group(1)
                current_lines = [
                    match.group(2).strip()
                ]

            elif current_number is not None:
                current_lines.append(line)

        flush_paragraph()

        base = article_number.strip()
        result: Dict[str, str] = {}


        letter_pattern = re.compile(
            r"\(([a-z])\)",
            re.IGNORECASE,
        )

        roman_pattern = re.compile(
            r"\(([ivxlcdm]+)\)",
            re.IGNORECASE,
        )


        for paragraph_number, paragraph_lines in paragraphs:

            paragraph_text = " ".join(
                paragraph_lines
            ).strip()

            if not paragraph_text:
                continue

            paragraph_key = self._citation_key(
                f"{base}({paragraph_number})"
            )

            result[paragraph_key] = paragraph_text

            subpoints: List[Tuple[str, List[str]]] = []

            current_letter: Optional[str] = None
            current_sub_lines: List[str] = []

            letter_with_text = re.compile(
                r"^\(([a-z])\)\s+(.+)$",
                re.IGNORECASE,
            )

            letter_only = re.compile(
                r"^\(([a-z])\)$",
                re.IGNORECASE,
            )

            def flush_subpoint() -> None:
                nonlocal current_letter, current_sub_lines

                if current_letter is not None and current_sub_lines:
                    subpoints.append(
                        (
                            current_letter,
                            current_sub_lines.copy(),
                        )
                    )

                current_letter = None
                current_sub_lines = []

            for line_index, line in enumerate(paragraph_lines):
                match_with_text = letter_with_text.match(line)
                match_only = letter_only.match(line)

                # Labels such as (i) are ambiguous:
                # they can be a letter or a nested Roman numeral.
                nested_roman = False

                if match_with_text and current_letter is not None:
                    label = match_with_text.group(1).lower()

                    if label in {"i", "v", "x", "l", "c", "d", "m"}:
                        previous_text = (
                            current_sub_lines[-1].rstrip()
                            if current_sub_lines
                            else ""
                        )

                        next_line = (
                            paragraph_lines[line_index + 1]
                            if line_index + 1 < len(paragraph_lines)
                            else ""
                        )

                        next_is_roman_continuation = bool(
                            re.match(
                                r"^\((?:ii|iii|iv|vi|vii|viii|ix|xi|xii)\)\s+",
                                next_line,
                                re.IGNORECASE,
                            )
                        )

                        nested_roman = (
                            previous_text.endswith(":")
                            or next_is_roman_continuation
                        )

                if nested_roman:
                    current_sub_lines.append(line)

                elif match_with_text:
                    flush_subpoint()

                    current_letter = match_with_text.group(1).lower()
                    current_sub_lines = [
                        match_with_text.group(2).strip()
                    ]

                elif match_only:
                    flush_subpoint()

                    current_letter = match_only.group(1).lower()
                    current_sub_lines = []

                elif current_letter is not None:
                    current_sub_lines.append(line)
            flush_subpoint()

            for letter, sub_lines in subpoints:

                sub_text = " ".join(
                    sub_lines
                ).strip()

                if not sub_text:
                    continue

                sub_text = re.sub(
                    r"\s+",
                    " ",
                    sub_text,
                ).strip()


                sub_key = self._citation_key(
                    f"{base}({paragraph_number})({letter})"
                )

                result[sub_key] = sub_text
                roman_parts: List[Tuple[str, List[str]]] = []

                current_roman: Optional[str] = None
                current_roman_lines: List[str] = []

                def flush_roman() -> None:
                    nonlocal current_roman, current_roman_lines

                    if current_roman is not None:
                        text = " ".join(
                            current_roman_lines
                        ).strip()

                        if text:
                            roman_parts.append(
                                (
                                    current_roman,
                                    [text],
                                )
                            )

                    current_roman = None
                    current_roman_lines = []

                for line in sub_lines:

                    rmatch = re.match(
                        r"^\(([ivxlcdm]+)\)\s+(.+)$",
                        line,
                        re.IGNORECASE,
                    )

                    rlabel = re.match(
                        r"^\(([ivxlcdm]+)\)$",
                        line,
                        re.IGNORECASE,
                    )

                    if rmatch:
                        flush_roman()

                        current_roman = (
                            rmatch.group(1)
                            .lower()
                        )

                        current_roman_lines = [
                            rmatch.group(2).strip()
                        ]

                    elif rlabel:
                        flush_roman()

                        current_roman = (
                            rlabel.group(1)
                            .lower()
                        )

                        current_roman_lines = []

                    elif current_roman is not None:
                        current_roman_lines.append(line)

                flush_roman()

                for roman, roman_lines in roman_parts:

                    roman_text = " ".join(
                        roman_lines
                    ).strip()

                    if not roman_text:
                        continue

                    roman_text = re.sub(
                        r"\s+",
                        " ",
                        roman_text,
                    ).strip()

                    roman_key = self._citation_key(
                        f"{base}({paragraph_number})"
                        f"({letter})({roman})"
                    )

                    result[roman_key] = roman_text

        return result

    def _best_scores(self, scores_list: List[Any]) -> Tuple[float, float, float]:
        if not scores_list:
            return 0.0, 0.0, 0.0

        best_scores = (0.0, 0.0, 0.0)
        best_entailment = -1.0

        for scores in scores_list:
            prob_dict = {
                self.id2label[i]: float(scores[i])
                for i in range(len(scores))
            }
            entailment = prob_dict.get("entailment", 0.0)
            if entailment > best_entailment:
                best_entailment = entailment
                best_scores = (
                    entailment,
                    prob_dict.get("contradiction", 0.0),
                    prob_dict.get("neutral", 0.0),
                )

        return best_scores

    @staticmethod
    def _classify(
        entailment: float,
        contradiction: float,
        neutral: float,
        threshold: float,
    ) -> str:
        if entailment >= threshold and entailment > contradiction:
            return "supported"
        if contradiction > neutral:
            return "contradicted"
        return "unsupported"
    @staticmethod
    def _normalize_for_exact_match(text: str) -> str:
        value = text.lower().strip()
        value = re.sub(r"\[.*?\]", " ", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()


    @staticmethod
    def _content_candidates(
        claim: str,
        source_chunks: List[str],
        limit: int = 12,
    ) -> List[str]:

        claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
        if not claim_tokens or len(source_chunks) <= limit:
            return source_chunks

        scored: List[Tuple[int, int, str]] = []
        for index, source in enumerate(source_chunks):
            source_tokens = set(re.findall(r"[a-z0-9]+", source.lower()))
            overlap = len(claim_tokens & source_tokens)
            scored.append((overlap, -len(source), source))

        scored.sort(reverse=True)
        selected = [source for _, _, source in scored[:limit]]
        return selected or source_chunks[:limit]


    def verify_response(
        self,
        generated_response: str,
        sources: List[Dict[str, Any]],
        entailment_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        claims_with_citations = self._extract_claims_with_citations(
            generated_response
        )

        for idx, (claim, citation) in enumerate(claims_with_citations, start=1):
                logger.debug("Claim %d | Citation: %s | Text: %s", idx, citation, claim)
        if not claims_with_citations:
            return {
                "is_faithful": True,
                "hallucination_rate": 0.0,
                "citation_errors": 0,
                "uncited_claims": 0,
                "claims": [],
            }


        all_source_chunks = []
        citation_source_map = {}
        article_source_map = {}

        for source in sources:
            article_number = str(
            source.get("article_number", "")
        ).strip()

            if not article_number:
                continue

            article_key = self._citation_key(article_number)

            selected_chunks = source.get(
                "selected_chunks",
                []
            ) or []
            selected_chunks = source.get("selected_chunks", [])

            paragraph_groups: Dict[str, List[Dict[str, Any]]] = {}

            for chunk in selected_chunks:
                paragraph_number = chunk.get("paragraph_number")

                if paragraph_number is None:
                    continue

                paragraph_key = str(paragraph_number)

                paragraph_groups.setdefault(
                    paragraph_key,
                    []
                ).append(chunk)
            debug_key = self._citation_key("Article 13(3)(b)")


            for paragraph_number, group in paragraph_groups.items():

                if any(
                    chunk.get("citation_label")
                    for chunk in selected_chunks
                ):
                    continue

                if len(group) < 2:
                    continue
                first_text = str(
                    group[0].get("text", "")
                ).strip()

                is_structured_list = bool(
                    re.search(
                        r"\b(?:the following steps|"
                        r"the following provisions|"
                        r"the following obligations|"
                        r"the following)\b",
                        first_text,
                        flags=re.IGNORECASE,
                    )
                )

                if not is_structured_list:
                    continue

                letter_index = 0


                for chunk in group[1:]:

                    chunk_text = str(
                        chunk.get("text", "")
                    ).strip()

                    if not chunk_text:
                        continue


                    if re.match(
                        r"^\s*\d+\.\s+",
                        chunk_text,
                    ):
                        continue

                    letter = chr(
                        ord("a") + letter_index
                    )

                    nested_key = self._citation_key(
                        f"{article_number}"
                        f"({paragraph_number})"
                        f"({letter})"
                    )

                    nli_text = (
                        f"{article_number}\n"
                        f"{chunk_text}"
                    )

                    citation_source_map.setdefault(
                        nested_key,
                        []
                    )

                    if nli_text not in citation_source_map[nested_key]:
                        citation_source_map[nested_key].append(
                            nli_text
                        )

                    letter_index += 1
            for chunk in selected_chunks:
                chunk_text = str(
                    chunk.get("text", "")
                ).strip()

                if not chunk_text:
                    continue

                nli_text = f"{article_number}\n{chunk_text}"
                provision_label = str(
                    chunk.get("citation_label", "")
                ).strip()

                if provision_label:
                    provision_key = self._citation_key(
                        provision_label
                    )
                    provision_text = (
                        f"{provision_label}\n{chunk_text}"
                    )

                    citation_source_map.setdefault(
                        provision_key,
                        [],
                    )

                    if provision_text not in citation_source_map[provision_key]:
                        citation_source_map[provision_key].append(
                            provision_text
                        )

                if nli_text not in all_source_chunks:
                    all_source_chunks.append(nli_text)
                article_source_map.setdefault(
                    article_key,
                    []
                )

                if nli_text not in article_source_map[article_key]:
                    article_source_map[article_key].append(nli_text)

                base_citation_key = self._citation_key(
                    article_number
                )

                citation_source_map.setdefault(
                    base_citation_key,
                    []
                )

                if nli_text not in citation_source_map[base_citation_key]:
                    citation_source_map[base_citation_key].append(nli_text)


                paragraph_number = chunk.get(
                    "paragraph_number"
                )

                if paragraph_number is None:
                    match = re.match(
                        r"^\s*(\d+)\.\s+",
                        chunk_text
                    )
                    paragraph_number = (
                        match.group(1)
                        if match
                        else None
                    )

                if paragraph_number is not None:
                    key = self._citation_key(
                        f"{article_number}({paragraph_number})"
                    )

                    citation_source_map.setdefault(
                        key,
                        []
                    )

                    if nli_text not in citation_source_map[key]:
                        citation_source_map[key].append(nli_text)


            if not selected_chunks:
                fallback_text = str(
                    source.get("full_text", "")
                ).strip()

                if fallback_text:
                    nli_text = (
                        f"{article_number}\n"
                        f"{fallback_text}"
                    )

                    if nli_text not in all_source_chunks:
                        all_source_chunks.append(nli_text)

                    citation_source_map.setdefault(
                        article_key,
                        []
                    )

                    if nli_text not in citation_source_map[article_key]:
                        citation_source_map[article_key].append(nli_text)

                    article_source_map.setdefault(
                        article_key,
                        []
                    )

                    if nli_text not in article_source_map[article_key]:
                        article_source_map[article_key].append(nli_text)

            full_text = str(
                source.get("full_text", "")
            ).strip()

            if not full_text:
                continue

            provisions = self._split_article_provisions(
                article_number,
                full_text,
            )

            for key, provision_text in provisions.items():
                citation_source_map.setdefault(
                    key,
                    []
                )

                if provision_text not in citation_source_map[key]:
                    citation_source_map[key].append(provision_text)

                article_source_map.setdefault(
                    article_key,
                    []
                )

                if provision_text not in article_source_map[article_key]:
                    article_source_map[article_key].append(provision_text)


        if not all_source_chunks:
            return {
                "is_faithful": False,
                "hallucination_rate": 1.0,
                "citation_errors": 0,
                "uncited_claims": len(claims_with_citations),
                "claims": [],
            }

        content_pairs: List[List[str]] = []
        content_claim_index: List[int] = []
        for index, (claim, cited_article) in enumerate(claims_with_citations):
            normalized_claim = self._normalize_claim_for_nli(claim)
            candidate_chunks = self._content_candidates(
                normalized_claim,
                all_source_chunks,
                limit=12,
            )

            if cited_article:
                cited_key = self._citation_key(cited_article)
                preferred = citation_source_map.get(
                    cited_key,
                    [],
                )

                if not preferred:
                    nested_match = re.match(
                        r"^(article\d+\(\d+\))\([^)]+\)$",
                        cited_key,
                        flags=re.IGNORECASE,
                    )

                    if nested_match:
                        parent_key = nested_match.group(1)
                        preferred = citation_source_map.get(
                            parent_key,
                            [],
                        )

                if not preferred and "(" not in cited_key:
                    preferred = article_source_map.get(
                        self._base_article_number(cited_article),
                        [],
                    )
                merged = []
                for item in preferred + candidate_chunks:
                    if item not in merged:
                        merged.append(item)

                if "(" in cited_key:
                    preferred_sorted = sorted(preferred, key=len, reverse=True)

                    candidate_chunks = []
                    for item in preferred_sorted + candidate_chunks:
                        if item not in candidate_chunks:
                            candidate_chunks.append(item)

                    candidate_chunks = candidate_chunks[:12]
                else:
                    candidate_chunks = list(preferred)

                    remaining = 12 - len(candidate_chunks)
                    if remaining > 0:
                        for item in merged:
                            if item not in candidate_chunks:
                                candidate_chunks.append(item)
                                if len(candidate_chunks) >= 12:
                                    break

            for source_text in candidate_chunks:
                content_pairs.append([source_text, normalized_claim])
                content_claim_index.append(index)

        for citation_label in (
            "Article 55(1)(a)",
            "Article 55(1)(b)",
            "Article 55(1)(c)",
            "Article 55(1)(d)",
            "Article 53(3)",
        ):
            target_key = self._citation_key(citation_label)
            sources = citation_source_map.get(target_key, [])

            for index, source_text in enumerate(sources, start=1):
                logger.debug(
                    "%s source %d retrieved (%d characters)",
                    citation_label,
                    index,
                    len(source_text),
                )
        content_scores = self.model.predict(
            content_pairs,
            apply_softmax=True,
            batch_size=64,
        )

        for claim_index, score in zip(content_claim_index, content_scores):
            claim_text = claims_with_citations[claim_index][0]

            if "1 500 000" in claim_text or "750 000" in claim_text:
                print(
                    "[Q19 CONTENT SCORE]",
                    claim_text,
                    "RAW_SCORE=",
                    score,
                )

        content_scores_map: Dict[int, List[Any]] = {
            i: [] for i in range(len(claims_with_citations))
        }

        for claim_index, score in zip(
            content_claim_index,
            content_scores,
        ):
            content_scores_map[claim_index].append(score)

        for claim_index, (claim, cited_article) in enumerate(
            claims_with_citations
        ):
            entailment, contradiction, neutral = self._best_scores(
                content_scores_map[claim_index]
            )

            status = self._classify(
                entailment,
                contradiction,
                neutral,
                entailment_threshold,
            )

            print(
                f"Claim {claim_index + 1}: {status} | "
                f"entailment={entailment:.4f} | "
                f"contradiction={contradiction:.4f} | "
                f"neutral={neutral:.4f}\n"
                f"  {claim}"
            )

        citation_pairs: List[List[str]] = []
        citation_claim_index: List[int] = []
        citation_lookup_meta: Dict[int, str] = {}
        citation_exact_match: Dict[int, bool] = {
            i: False
            for i in range(len(claims_with_citations))
        }
        for claim_index, (_, cited_article) in enumerate(claims_with_citations):
            if not cited_article:
                continue

            citation_key = self._citation_key(cited_article)
            cited_chunks = citation_source_map.get(citation_key, [])
            lookup_type = "exact"

            if not cited_chunks:
                nested_match = re.match(
                    r"^(article\d+\(\d+\))\([^)]+\)$",
                    citation_key,
                    flags=re.IGNORECASE,
                )

                if nested_match:
                    parent_key = nested_match.group(1)
                    parent_chunks = citation_source_map.get(
                        parent_key,
                        [],
                    )

                    if parent_chunks:
                        cited_chunks = parent_chunks
                        lookup_type = "parent_paragraph_fallback"

            if not cited_chunks and "(" not in citation_key:
                base_key = self._base_article_number(cited_article)
                cited_chunks = article_source_map.get(base_key, [])
                lookup_type = "article"

            if cited_chunks:
                citation_lookup_meta[claim_index] = lookup_type

                normalized_claim = self._normalize_for_exact_match(
                    claims_with_citations[claim_index][0]
                )

                for provision in cited_chunks:

                    citation_pairs.append([
                        provision,
                        self._normalize_claim_for_nli(
                            claims_with_citations[claim_index][0]
                        )
                    ])

                    citation_claim_index.append(
                        claim_index
                    )


                    normalized_provision = (
                        self._normalize_for_exact_match(
                            provision
                        )
                    )

                    if (
                        len(normalized_claim) >= 40
                        and normalized_claim
                        in normalized_provision
                    ):
                        citation_exact_match[claim_index] = True

        citation_scores_map: Dict[int, List[Any]] = {
            i: [] for i in range(len(claims_with_citations))
        }
        if citation_pairs:
            citation_scores = self.model.predict(
                citation_pairs,
                apply_softmax=True,
                batch_size=64,
            )


            for claim_index, score in zip(
                citation_claim_index,
                citation_scores,
            ):
                claim_text = claims_with_citations[claim_index][0]

                if "1 500 000" in claim_text or "750 000" in claim_text:
                    print(
                        "[Q19 CITATION SCORE]",
                        claim_text,
                        "CITATION=",
                        claims_with_citations[claim_index][1],
                        "RAW_SCORE=",
                        score,
                    )
            for claim_index, score in zip(citation_claim_index, citation_scores):
                citation_scores_map[claim_index].append(score)

        verified_claims: List[Dict[str, Any]] = []
        hallucination_count = 0
        citation_error_count = 0
        uncited_count = 0

        for claim_index, (claim, cited_article) in enumerate(claims_with_citations):
            abstention_claim = self.is_abstention_claim(claim)

            content_entailment, content_contradiction, content_neutral = self._best_scores(
                content_scores_map[claim_index]
            )
            content_status = self._classify(
                content_entailment,
                content_contradiction,
                content_neutral,
                entailment_threshold,
            )


            exact_text_support = False
            lexical_legal_support = False
            deterministic_support = False

            if cited_article:
                citation_key = self._citation_key(cited_article)

                exact_sources = citation_source_map.get(
                    citation_key,
                    [],
                )
                if cited_article == "Article 55(3)":
                    print(
                        "[Q15 DEBUG] exact_sources count:",
                        len(exact_sources),
                    )
                    for i, source_text in enumerate(exact_sources):
                        print(
                            f"[Q15 DEBUG] source {i}:",
                            repr(source_text),
                        )

                exact_text_support = self._has_exact_text_support(
                    claim,
                    exact_sources,
                )

                lexical_legal_support = self._has_lexical_legal_support(
                    claim,
                    exact_sources,
                )

                deterministic_support = (
                exact_text_support
                or lexical_legal_support
                or citation_exact_match.get(claim_index, False)
                 )
                if cited_article == "Article 55(3)":
                        print(
                            "[Q15 DEBUG] exact=",
                            exact_text_support,
                            "lexical=",
                            lexical_legal_support,
                            "citation_exact=",
                            citation_exact_match.get(claim_index, False),
                            "deterministic=",
                            deterministic_support,
                        )
                if (
                    deterministic_support
                    and content_status == "unsupported"
                    and content_contradiction < 0.50
                ):
                    content_status = "supported"

            if content_status != "supported":
                    logger.warning(
                        "NLI claim %d failed: status=%s, "
                        "entailment=%.4f, contradiction=%.4f, neutral=%.4f",
                        claim_index + 1,
                        content_status,
                        content_entailment,
                        content_contradiction,
                        content_neutral,
                    )
            citation_valid = False
            citation_status = "uncited"
            citation_entailment = 0.0
            citation_contradiction = 0.0
            citation_neutral = 0.0

            citation_scores_list = citation_scores_map.get(
                claim_index,
                [],
            )

            if cited_article is None:
                if not abstention_claim:
                    uncited_count += 1
            else:
                if citation_scores_list:
                    (
                        citation_entailment,
                        citation_contradiction,
                        citation_neutral,
                    ) = self._best_scores(citation_scores_list)

                    citation_status = self._classify(
                        citation_entailment,
                        citation_contradiction,
                        citation_neutral,
                        entailment_threshold,
                    )

                    citation_valid = (
                        citation_status == "supported"
                        or (
                            deterministic_support
                            and content_status == "supported"
                        )
                    )
                    if citation_valid and citation_status != "supported":
                        citation_status = "supported_deterministic"
                else:
                    citation_status = "citation_not_indexed"

                if not citation_valid and not abstention_claim:
                    citation_error_count += 1

            if abstention_claim and content_status != "contradicted":
                status = "abstention"
                hallucination_flag = False
            elif content_status == "supported":
                status = (
                    "supported_uncited"
                    if cited_article is None
                    else "supported_cited"
                    if citation_valid
                    else "supported_wrong_citation"
                )
                hallucination_flag = False
            elif content_status == "contradicted":
                status = (
                    "contradicted_uncited"
                    if cited_article is None
                    else "contradicted_cited"
                )
                hallucination_flag = True
            else:
                status = (
                    "unsupported_uncited"
                    if cited_article is None
                    else "unsupported_cited"
                )
                hallucination_flag = True

            if hallucination_flag:
                hallucination_count += 1

            citation_flag = (not abstention_claim) and (cited_article is None or not citation_valid)
            verified_claims.append({
                "claim": claim,
                "cited_article": cited_article,
                "status": status,
                "support_status": content_status,
                "support_confidence": round(content_entailment, 4),
                "citation_valid": citation_valid,
                "citation_status": citation_status,
                "citation_confidence": round(citation_entailment, 4),
                "entailment_score": round(content_entailment, 4),
                "contradiction_score": round(content_contradiction, 4),
                "neutral_score": round(content_neutral, 4),
                "citation_entailment_score": round(citation_entailment, 4),
                "citation_contradiction_score": round(citation_contradiction, 4),
                "citation_neutral_score": round(citation_neutral, 4),
                "abstention_claim": abstention_claim,
                "hallucination_flag": hallucination_flag,
                "citation_flag": citation_flag,
                "flagged": hallucination_flag or citation_flag,
                "exact_text_support": exact_text_support,
                "lexical_legal_support": lexical_legal_support,
                "deterministic_support": deterministic_support,
            })

        hallucination_rate = (
            hallucination_count / len(verified_claims)
            if verified_claims
            else 0.0
        )

        return {
            "is_faithful": hallucination_count == 0,
            "hallucination_rate": round(hallucination_rate, 2),
            "citation_errors": citation_error_count,
            "uncited_claims": uncited_count,
            "claims": verified_claims,
        }


if __name__ == "__main__":
    print("NLIConstraintVerifier module loaded. Use the application/benchmark runner to execute it.")
