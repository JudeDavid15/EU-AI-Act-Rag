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


def parse_eurlex_html(html_file_path: Path) -> list[ParentChunk]:
    if not html_file_path.exists():
        raise FileNotFoundError(f"HTML file not found at: {html_file_path}")

    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    parent_chunks: list[ParentChunk] = []
    article_pattern = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)

    elements = soup.find_all("p")

    current_parent: Optional[ParentChunk] = None
    current_paragraphs: list[str] = []
    art_count = 0

    for el in elements:
        text = clean_text(el.get_text())
        if not text:
            continue

        match = article_pattern.search(text)
        if match:
            if current_parent and current_paragraphs:
                _finalize_parent(current_parent, current_paragraphs, parent_chunks)
                current_paragraphs = []

            art_count += 1
            art_num = f"Article {match.group(1)}"
            parent_id = f"art_{match.group(1)}"

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


def _finalize_parent(parent: ParentChunk, paragraphs: list[str], output_list: list[ParentChunk]):
    parent.full_text = f"{parent.title}\n" + "\n".join(paragraphs)

    for idx, p_text in enumerate(paragraphs, start=1):
        if len(p_text.split()) < 5:
            continue

        child = ChildChunk(
            chunk_id=f"{parent.parent_id}_p{idx}",
            parent_id=parent.parent_id,
            article_number=parent.article_number,
            title=parent.title,
            text=p_text
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
