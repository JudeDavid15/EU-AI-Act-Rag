from typing import Optional
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from src.ingestion.schemas import ParentChunk, ChildChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_HTML_PATH = PROJECT_ROOT / "data" / "raw" / "eu_ai_act_2024_1689.html"
PROCESSED_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def clean_text(text: str) -> str:
    """Normalizes whitespace and converts non-breaking space unicode (\\xa0)."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

def _direct_grid_marker(container):
    """Return the marker belonging directly to this grid list."""
    for marker_node in container.select(".grid-list-column-1"):
        if (
            marker_node.find_parent(
                "div", class_="grid-container"
            )
            is not container
        ):
            continue

        marker = clean_text(
            marker_node.get_text(" ", strip=True)
        )

        match = re.fullmatch(r"\(\s*([^()]+?)\s*\)", marker)
        if match:
            return match.group(1).strip()

    return None


def _grid_marker_path(container):
    """Return list markers from outermost to innermost."""
    ancestors = [
        node
        for node in container.find_parents(
            "div", class_="grid-container"
        )
        if "grid-list" in (node.get("class") or [])
    ]

    markers = []

    for node in reversed(ancestors):
        marker = _direct_grid_marker(node)
        if marker:
            markers.append(marker)

    current_marker = _direct_grid_marker(container)
    if current_marker:
        markers.append(current_marker)

    return markers


def _grid_citation_label(
    container,
    article_number,
    paragraph_number=None,
):
    """Build the complete article, paragraph and list-item citation."""
    markers = _grid_marker_path(container)
    parts = []

    if paragraph_number is not None:
        paragraph = str(paragraph_number)
        parts.append(paragraph)

        # Avoid duplicating a paragraph marker already in the HTML path.
        if markers and markers[0] == paragraph:
            markers = markers[1:]

    parts.extend(markers)

    suffix = "".join(f"({part})" for part in parts)
    return f"{article_number}{suffix}"
_ROMAN_MARKERS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii",
    "viii", "ix", "x", "xi", "xii", "xiii",
    "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
}


# def _flat_marker_citation(article_number, text, state):
#     """Infer citations for list items rendered as standalone paragraphs."""
#     match = re.match(r"^\s*\(([^()]+)\)\s*", text)

#     if match is None:
#         state["last_text_ends_colon"] = (
#             bool(state["path"]) and text.rstrip().endswith(":")
#         )
#         return article_number

#     marker = clean_text(match.group(1)).lower()
#     is_number = bool(re.fullmatch(r"\d+[a-z]?", marker))
#     is_alpha = bool(re.fullmatch(r"[a-z]", marker))
#     is_roman = marker in _ROMAN_MARKERS
#     previous_ends_colon = state["last_text_ends_colon"]

#     if is_number:
#         state["path"] = [marker]
#         state["number"] = marker
#         state["mode"] = "number"
#         state["roman_parent"] = None

#     elif state["mode"] == "roman" and is_roman:
#         parent_path = state["roman_parent"] or state["path"][:-1]
#         state["path"] = parent_path + [marker]

#     elif previous_ends_colon and state["path"]:
#         if (
#             state["mode"] == "number"
#             and is_alpha
#             and not is_roman
#         ):
#             state["path"] = [state["number"], marker]
#             state["mode"] = "number"
#         elif is_roman:
#             parent_path = list(state["path"])
#             state["roman_parent"] = parent_path
#             state["path"] = parent_path + [marker]
#             state["mode"] = "roman"
#         else:
#             state["path"] = list(state["path"]) + [marker]
#             state["mode"] = "nested"

#     elif state["number"] is not None and is_alpha:
#         state["path"] = [state["number"], marker]
#         state["mode"] = "number"

#     elif is_alpha:
#         state["path"] = [marker]
#         state["number"] = None
#         state["mode"] = "alpha"
#         state["roman_parent"] = None

#     elif is_roman and state["path"]:
#         parent_path = list(state["path"])
#         state["roman_parent"] = parent_path
#         state["path"] = parent_path + [marker]
#         state["mode"] = "roman"

#     else:
#         state["path"] = [marker]
#         state["number"] = None
#         state["mode"] = "other"
#         state["roman_parent"] = None

#     state["last_text_ends_colon"] = text.rstrip().endswith(":")

#     suffix = "".join(f"({part})" for part in state["path"])
#     return f"{article_number}{suffix}"

def parse_eurlex_html(html_file_path: Path) -> list[ParentChunk]:
    if not html_file_path.exists():
        raise FileNotFoundError(f"HTML file not found at: {html_file_path}")

    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        article_divs = soup.find_all(
            "div",
            class_="eli-subdivision",
            id=re.compile(
                r"^art_\d+[a-z]?$",
                re.IGNORECASE,
            ),
        )

        if article_divs:
            parent_chunks: list[ParentChunk] = []

            for article_div in article_divs:
                heading = article_div.find(
                    "p",
                    class_="title-article-norm",
                )

                if heading is None:
                    continue

                heading_text = clean_text(
                    heading.get_text(" ", strip=True)
                )

                match = re.fullmatch(
                    r"Article\s+(\d+[a-z]?)",
                    heading_text,
                    re.IGNORECASE,
                )

                if match is None:
                    continue

                article_suffix = match.group(1).lower()
                article_number = f"Article {article_suffix}"
                parent_id = f"art_{article_suffix}"

                # The article title is separate from the formal heading.
                title_node = article_div.find(
                    "div",
                    class_="eli-title",
                )

                article_title = (
                    clean_text(title_node.get_text(" ", strip=True))
                    if title_node is not None
                    else ""
                )

                parent = ParentChunk(
                    parent_id=parent_id,
                    article_number=article_number,
                    title=article_number,
                    full_text="",
                    children=[],
                )

                full_text_parts = [article_number]

                if article_title:
                    full_text_parts.append(article_title)

                current_paragraph_number: Optional[str] = None
                child_index = 0
                # flat_citation_state = {
                #         "path": [],
                #         "number": None,
                #         "mode": None,
                #         "roman_parent": None,
                #         "last_text_ends_colon": False,
                #     }

                def add_child(
                        text,
                        paragraph_number=None,
                        subpoint=None,
                        citation_label=None,
                    ):
                    nonlocal child_index

                                   # Merge a continuation block when the preceding
                # chunk in the same paragraph ends mid-sentence.
                    if (
                        subpoint is None
                        and paragraph_number is not None
                        and parent.children
                    ):
                        previous = parent.children[-1]

                        if (
                            previous.paragraph_number == paragraph_number
                            and previous.citation_label
                            == f"{article_number}({paragraph_number})"
                            and not re.search(
                                r'[.!?;]["”’)\]]?$',
                                previous.text.rstrip(),
                            )
                        ):
                            previous.text = clean_text(
                                f"{previous.text} {text}"
                            )
                            return
                    if citation_label is None:
                        if paragraph_number is not None:
                            citation_label = (
                                f"{article_number}({paragraph_number})"
                            )

                            if subpoint is not None:
                                citation_label += f"({subpoint})"
                        else:
                            citation_label = article_number
                    child_index += 1

                    parent.children.append(
                        ChildChunk(
                            chunk_id=(
                                f"{parent_id}_p{child_index}"
                            ),
                            parent_id=parent_id,
                            article_number=article_number,
                            title=article_number,
                            text=text,
                            paragraph_number=paragraph_number,
                            citation_label=citation_label,
                        )
                    )

                body_nodes = article_div.find_all(
                    ["div", "p"],
                    class_="norm",
                )

                for node in body_nodes:
                    # Process only the outer normative block.
                    # Nested norm elements are handled through
                    # their containing block.
                    if node.find_parent(class_="norm") is not None:
                        continue

                    marker = node.find(
                        "span",
                        class_="no-parag",
                        recursive=False,
                    )

                    if marker is not None:
                        marker_match = re.match(
                            r"^\s*(\d+)\s*\.",
                            marker.get_text(" ", strip=True),
                        )

                        if marker_match:
                            current_paragraph_number = (
                                marker_match.group(1)
                            )

                    node_text = clean_text(
                        node.get_text(" ", strip=True)
                    )

                    if node_text:
                        full_text_parts.append(node_text)

                    # Find the direct content wrapper.
                    inline_container = None

                    for child in node.find_all(
                        "div",
                        recursive=False,
                    ):
                        classes = set(child.get("class") or [])

                        if {
                            "norm",
                            "inline-element",
                        }.issubset(classes):
                            inline_container = child
                            break

                    if inline_container is None:
    # Handle normative blocks without the inline wrapper.
                        fallback_text = node_text

                        if marker is not None:
                            fallback_text = re.sub(
                                r"^\s*\d+\.\s*",
                                "",
                                fallback_text,
                                count=1,
                            )

                        # List markers may be stored in ancestor grid containers.
                        grid_container = node.find_parent(
                            "div",
                            class_="grid-container",
                        )

                        if (
                            grid_container is not None
                            and "grid-list" in (grid_container.get("class") or [])
                        ):
                            fallback_citation = _grid_citation_label(
                                grid_container,
                                article_number,
                                current_paragraph_number,
                            )
                        elif current_paragraph_number is not None:
                            fallback_citation = (
                                f"{article_number}({current_paragraph_number})"
                            )
                        else:
                            fallback_citation = article_number

                        add_child(
                            fallback_text,
                            current_paragraph_number,
                            citation_label=fallback_citation,
                        )
                        continue
                    for part in inline_container.children:
                        if not getattr(part, "name", None):
                            text = clean_text(str(part))

                            if not text:
                                continue

                            # Preserve the paragraph number when this is the first
                            # text block belonging to a numbered paragraph.
                            if (
                                marker is not None
                                and current_paragraph_number is not None
                            ):
                                text = f"{current_paragraph_number}. {text}"
                                marker = None

                            add_child(
                                text,
                                current_paragraph_number,
                            )
                            continue

                        classes = set(part.get("class") or [])
                        # Handle lettered provisions such as
                        # (a), (b), and (c) separately.
                        if (
                            part.name == "div"
                            and "grid-container" in classes
                            and "grid-list" in classes
                        ):
                            label_column = part.find(
                                "div",
                                class_="grid-list-column-1",
                            )

                            body_column = part.find(
                                "div",
                                class_="grid-list-column-2",
                            )

                            label_text = (
                                clean_text(
                                    label_column.get_text(
                                        " ",
                                        strip=True,
                                    )
                                )
                                if label_column is not None
                                else ""
                            )
                            point_text = (
                                clean_text(
                                    body_column.get_text(" ", strip=True)
                                )
                                if body_column is not None
                                else ""
                            )
                            marker_match = re.fullmatch(
                                r"\(\s*([^()]+?)\s*\)",
                                label_text,
                            )

                            grid_citation = _grid_citation_label(
                                part,
                                article_number,
                                current_paragraph_number,
                            )

                            if not point_text:
                                continue

                            if marker_match is not None:
                                add_child(
                                    point_text,
                                    current_paragraph_number,
                                    marker_match.group(1).lower(),
                                    citation_label=grid_citation,
                                )
                            else:
                                add_child(
                                    f"{label_text} {point_text}".strip(),
                                    current_paragraph_number,
                                    citation_label=grid_citation,
                                )
                        elif "norm" in classes:
                            # Introductory paragraphs and ordinary
                            # text blocks remain paragraph-level.
                            text = clean_text(
                                part.get_text(" ", strip=True)
                            )

                            if not text:
                                continue

                            # Retain the original paragraph prefix
                            # when this is its first text block.
                            if (
                                marker is not None
                                and re.match(
                                    r"^\s*\d+\s*\.",
                                    marker.get_text(
                                        " ",
                                        strip=True,
                                    ),
                                )
                            ):
                                text = (
                                    f"{current_paragraph_number}. "
                                    f"{text}"
                                )
                                marker = None

                            add_child(
                                text,
                                current_paragraph_number,
                            )

                parent.full_text = "\n".join(full_text_parts)
                parent_chunks.append(parent)
            if not parent_chunks:
                raise ValueError(
                    "Consolidated HTML was detected, but no article "
                    "containers could be parsed."
                )

            zero_child_articles = [
                parent.article_number
                for parent in parent_chunks
                if not parent.children
            ]

            print(
                "Parsed consolidated article containers:",
                len(parent_chunks),
            )

            if zero_child_articles:
                print(
                    "WARNING: Articles with no child chunks:",
                    zero_child_articles,
                )

            return parent_chunks
    parent_chunks: list[ParentChunk] = []
    article_pattern = re.compile(
    r"^Article\s+(\d+[a-z]?)$",
    re.IGNORECASE,
    )

    elements = soup.find_all("p")

    current_parent: Optional[ParentChunk] = None
    current_paragraphs: list[str] = []
    art_count = 0

    for el in elements:
        text = clean_text(el.get_text())
        if not text:
            continue

        is_article_heading = (
            "title-article-norm" in (el.get("class") or [])
        )

        match = (
            article_pattern.fullmatch(text)
            if is_article_heading
            else None
        )

        if match:
            if current_parent and current_paragraphs:
                _finalize_parent(current_parent, current_paragraphs, parent_chunks)
                current_paragraphs = []

            art_count += 1
            article_id = match.group(1).lower()

            art_num = f"Article {article_id}"
            parent_id = f"art_{article_id}"
            current_parent = ParentChunk(
                parent_id=parent_id,
                article_number=art_num,
                title=text,
                full_text="",
                children=[]
            )
            continue

        if current_parent:
            current_paragraphs.append(text)


    if current_parent and current_paragraphs:
        _finalize_parent(current_parent, current_paragraphs, parent_chunks)

    print(f"Successfully parsed {len(parent_chunks)} Articles from HTML")
    return parent_chunks


def _finalize_parent(
    parent: ParentChunk,
    paragraphs: list[str],
    output_list: list[ParentChunk],
) -> None:
    parent.full_text = (
        f"{parent.title}\n" + "\n".join(paragraphs)
    )

    current_paragraph_number: Optional[str] = None

    for idx, p_text in enumerate(paragraphs, start=1):
        if len(p_text.split()) < 5:
            continue

        # Update the active paragraph when the text begins with
        # a numbered legal paragraph, e.g. "2. In order to..."
        paragraph_match = re.match(
            r"^\s*(\d+)\.\s+",
            p_text,
        )

        if paragraph_match:
            current_paragraph_number = paragraph_match.group(1)

        citation_label = (
            f"{parent.article_number}({current_paragraph_number})"
            if current_paragraph_number is not None
            else None
        )

        child = ChildChunk(
            chunk_id=f"{parent.parent_id}_p{idx}",
            parent_id=parent.parent_id,
            article_number=parent.article_number,
            title=parent.title,
            text=p_text,
            paragraph_number=current_paragraph_number,
            citation_label=citation_label,
        )

        parent.children.append(child)

    output_list.append(parent)

def main():
    PROCESSED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Starting EUR-Lex HTML parsing...")
    parents = parse_eurlex_html(RAW_HTML_PATH)

    data = [p.model_dump() for p in parents]
    with open(PROCESSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved processed chunks to {PROCESSED_JSON_PATH}")


if __name__ == "__main__":
    main()
