import re
import json
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


class ContextBuilder:
    def __init__(
        self,
        max_total_chars: int = 6000,
        max_source_chars: int = 2500,
        expansion_radius: int = 2,
    ):
        if not PROCESSED_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Missing chunk data at {PROCESSED_JSON_PATH}"
            )

        with open(PROCESSED_JSON_PATH, "r", encoding="utf-8") as f:
            parents_data = json.load(f)

        self.parent_map: Dict[str, Dict[str, Any]] = {
            parent["parent_id"]: parent
            for parent in parents_data
        }


        self.max_total_chars = max_total_chars
        self.max_source_chars = max_source_chars
        self.expansion_radius = expansion_radius

        self.child_index_map: Dict[str, Dict[str, int]] = {}

        for parent in parents_data:
            parent_id = parent["parent_id"]
            children = parent.get("children", [])

            self.child_index_map[parent_id] = {
                child["chunk_id"]: index
                for index, child in enumerate(children)
                if child.get("chunk_id")
            }

    def _get_local_chunk_indices(
        self,
        parent_id: str,
        matched_indices: List[int],
    ) -> List[int]:

        selected = set()

        for index in matched_indices:
            selected.add(index)

            for distance in range(1, self.expansion_radius + 1):
                previous_index = index - distance
                next_index = index + distance

                if previous_index >= 0:
                    selected.add(previous_index)

                next_index_exists = (
                    next_index
                    < len(self.parent_map[parent_id].get("children", []))
                )

                if next_index_exists:
                    selected.add(next_index)

        return sorted(selected)

    def _build_source_text(
    self,
    parent: Dict[str, Any],
    matched_indices: List[int],
) -> tuple[str, List[str], List[Dict[str, str]]]:

        children = parent.get("children", [])

        if not children or not matched_indices:
            return "", [], []

        paragraph_by_index = {}
        current_paragraph = None

        for child_index, child in enumerate(children):

            child_text = child.get(
                "text",
                ""
            ).strip()

            paragraph_match = re.match(
                r"^(\d+)\.\s+",
                child_text
            )

            if paragraph_match:
                current_paragraph = (
                    paragraph_match.group(1)
                )

            paragraph_by_index[child_index] = (
                current_paragraph
            )

        paragraph_indices = {}

        for index, paragraph in paragraph_by_index.items():
            if paragraph is not None:
                paragraph_indices.setdefault(
                    paragraph, []
                ).append(index)

        structured_list_paragraphs = set()

        for paragraph, indices in paragraph_indices.items():
            if len(indices) < 2:
                continue

            first_text = str(
                children[indices[0]].get("text", "")
            ).strip()

            if re.search(
                r"\bthe following\b",
                first_text,
                flags=re.IGNORECASE,
            ):
                structured_list_paragraphs.add(paragraph)

        parent_id = str(parent.get("parent_id", ""))

        if parent_id in self.parent_map:
            required_indices = set(
                self._get_local_chunk_indices(
                    parent_id,
                    matched_indices,
                )
            )
        else:
            required_indices = set(matched_indices)

        for paragraph in structured_list_paragraphs:
            if any(
                paragraph_by_index.get(index) == paragraph
                for index in matched_indices
            ):
                required_indices.update(
                    paragraph_indices[paragraph]
                )

        parent_full_text = str(parent.get("full_text", ""))
        source_lines = []
        source_events = []
        source_paragraph = None
        expected_letter = "a"
        previous_content = ""
        running_length = 0

        for raw_line in parent_full_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue

            position = running_length + (1 if source_lines else 0)

            paragraph_match = re.match(
                r"^(\d+)\.\s+(.*)$",
                line,
            )

            if paragraph_match:
                source_paragraph = paragraph_match.group(1)
                expected_letter = "a"
                previous_content = paragraph_match.group(2).strip()
                source_events.append(
                    (position, source_paragraph, None)
                )
            else:
                marker_match = re.fullmatch(
                    r"\(([a-z]+)\)",
                    line,
                    flags=re.IGNORECASE,
                )

                if marker_match:
                    label = marker_match.group(1).lower()

                    nested_intro = bool(
                        re.search(
                            r"(?:either\s+or\s+both\s+of\s+the\s+following|"
                            r"one\s+of\s+the\s+following\s+objectives|"
                            r"following\s+objectives)\s*:?\s*$",
                            previous_content,
                            flags=re.IGNORECASE,
                        )
                    )

                    if (
                        len(label) == 1
                        and label == expected_letter
                        and not nested_intro
                    ):
                        source_events.append(
                            (position, source_paragraph, label)
                        )
                        expected_letter = (
                            chr(ord(label) + 1)
                            if label < "z"
                            else None
                        )
                else:
                    previous_content = line

            source_lines.append(line)
            running_length = position + len(line)

        source_flat = " ".join(source_lines).lower()
        search_from = 0
        child_positions = {}

        for child_index, child in enumerate(children):
            child_text = re.sub(
                r"\s+",
                " ",
                str(child.get("text", "")),
            ).strip()

            if not child_text:
                continue

            needle = child_text.lower()
            position = source_flat.find(needle, search_from)

            if position < 0:
                position = source_flat.find(needle)

            if position >= 0:
                child_positions[child_index] = position
                search_from = position + len(needle)

        citation_label_by_index = {}

        for child_index, position in child_positions.items():
            active_paragraph = None
            active_letter = None

            for event_position, paragraph, letter in source_events:
                if event_position > position:
                    break

                active_paragraph = paragraph
                active_letter = letter

            if (
                active_letter
                and str(active_paragraph)
                == str(paragraph_by_index.get(child_index))
            ):
                article = str(
                    parent.get("article_number", "")
                ).strip()

                if article and not article.lower().startswith("article "):
                    article = f"Article {article}"

                if article:
                    if active_paragraph is None:
                        citation_label_by_index[child_index] = (
                            f"{article}({active_letter})"
                        )
                    else:
                        citation_label_by_index[child_index] = (
                            f"{article}"
                            f"({active_paragraph})"
                            f"({active_letter})"
                        )
        candidate_indices = []

        for index in range(len(children)):
            distance = min(
                abs(index - matched_index)
                for matched_index in matched_indices
            )
            candidate_indices.append((distance, index))

        candidate_indices.sort(
            key=lambda item: (item[0], item[1])
        )

        selected_indices = sorted(required_indices)
        selected_set = set(selected_indices)

        current_chars = sum(
            len(children[index].get("text", "").strip())
            for index in selected_indices
        )
        current_chars += max(0, len(selected_indices) - 1) * 2

        for _, index in candidate_indices:
            if index in selected_set:
                continue

            text = str(
                children[index].get("text", "")
            ).strip()

            if not text:
                continue

            additional_chars = len(text)

            if selected_indices:
                additional_chars += 2

            if (
                current_chars + additional_chars
                > self.max_source_chars
            ):
                continue

            selected_indices.append(index)
            selected_set.add(index)
            current_chars += additional_chars

        selected_indices.sort()
        selected_texts = []
        selected_ids = []
        selected_chunks = []

        for index in selected_indices:

            child = children[index]
            text = child.get("text", "").strip()

            if not text:
                continue

            chunk_id = child.get("chunk_id", "")

            selected_texts.append(text)
            selected_ids.append(chunk_id)

            paragraph_number = paragraph_by_index.get(
                index
            )

            selected_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "paragraph_number": paragraph_number,
                "citation_label": citation_label_by_index.get(index),
            })
        return (
            "\n\n".join(selected_texts),
            selected_ids,
            selected_chunks,
        )
    def build_context(
        self,
        reranked_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        grouped: Dict[str, List[int]] = {}
        score_map: Dict[str, float] = {}
        matched_ids_map: Dict[str, List[str]] = {}

        for chunk in reranked_chunks:
            parent_id = chunk.get("parent_id")
            chunk_id = chunk.get("chunk_id")

            if not parent_id or not chunk_id:
                continue

            if parent_id not in self.parent_map:
                continue

            index_map = self.child_index_map.get(parent_id, {})

            if chunk_id not in index_map:
                continue

            chunk_index = index_map[chunk_id]

            grouped.setdefault(parent_id, []).append(chunk_index)

            chunk_priority = max(
                float(chunk.get("rerank_score", 0.0)),
                float(chunk.get("score", 0.0)),
                float(chunk.get("exact_reference_boost", 0.0)),
            )

            score_map[parent_id] = max(
                score_map.get(parent_id, float("-inf")),
                chunk_priority,
            )

            matched_ids_map.setdefault(parent_id, []).append(chunk_id)

        ordered_parents = sorted(
            grouped.keys(),
            key=lambda pid: score_map.get(pid, 0.0),
            reverse=True,
        )

        sources = []
        total_chars = 0

        for parent_id in ordered_parents:

            parent = self.parent_map[parent_id]

            matched_indices = sorted(
                set(grouped[parent_id])
            )

            local_text, selected_child_ids, selected_chunks = self._build_source_text(
                            parent=parent,
                            matched_indices=matched_indices,
                        )

            if not local_text:
                continue

            remaining_budget = (
                self.max_total_chars - total_chars
            )

            if remaining_budget <= 0:
                break

            if len(local_text) > remaining_budget:
                local_text = local_text[:remaining_budget]

            sources.append({
                    "parent_id": parent_id,
                    "article_number": parent.get(
                        "article_number", ""
                    ),
                    "title": parent.get(
                        "title", ""
                    ),
                    "full_text": local_text,
                    "matched_child_ids": matched_ids_map[parent_id],
                    "selected_child_ids": selected_child_ids,
                    "selected_chunks": selected_chunks,
                    "rerank_score": score_map[parent_id],
                })

            total_chars += len(local_text)

        formatted_parts = []

        for idx, source in enumerate(sources, start=1):
            evidence_parts = []
            current_paragraph = None
            current_letter = None

            for chunk in source.get("selected_chunks", []):
                paragraph_number = chunk.get("paragraph_number")
                article_number = source["article_number"]
                text = str(chunk.get("text", "")).strip()

                if not text:
                    continue

                if paragraph_number != current_paragraph:
                    current_paragraph = paragraph_number
                    current_letter = None

                marker = re.match(
                    r"^\s*\(([a-z])\)(?:\s|$)",
                    text,
                    flags=re.IGNORECASE,
                )

                if marker:
                    current_letter = marker.group(1).lower()

                provided_label = chunk.get("citation_label")

                if provided_label:
                    citation_label = f"[{provided_label}]"
                elif paragraph_number:
                    citation_label = (
                        f"[{article_number}({paragraph_number})]"
                    )
                else:
                    citation_label = f"[{article_number}]"
                evidence_parts.append(
                    f"{citation_label}\n{text}"
                )

            formatted_parts.append(
                f"--- DOCUMENT SOURCE {idx} ---\n"
                f"ARTICLE: {source['article_number']} - "
                f"{source['title']}\n"
                f"CONTENT:\n"
                f"{chr(10).join(evidence_parts)}"
            )

        formatted_context = "\n\n".join(
            formatted_parts
        ).strip()
        return {
            "sources": sources,
            "formatted_context": formatted_context,
        }


if __name__ == "__main__":

    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.reranker import Reranker

    query = (
        "what are the transparency obligations "
        "for high risk AI systems"
    )

    retriever = HybridRetriever()

    try:
        candidates = retriever.search(
            query=query,
            top_n=15,
        )
    finally:
        retriever.close()

    reranker = Reranker()

    top_chunks = reranker.rerank(
        query=query,
        candidates=candidates,
        top_n=3,
    )

    builder = ContextBuilder(
        max_total_chars=6000,
        max_source_chars=2500,
        expansion_radius=2,
    )

    context_payload = builder.build_context(
        top_chunks
    )

    print(
        "\nConstructed Context:\n"
    )

    print(
        context_payload["formatted_context"]
    )
