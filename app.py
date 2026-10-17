import hashlib
import json
import os
import time

import requests
import streamlit as st


st.set_page_config(
    page_title="EU AI Act RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("EU AI Act Compliance & Query Assistant")
st.markdown(
    "Ask legal and technical compliance questions about the **EU AI Act**. "
    "The backend checks the completed answer with its **NLI Guardrail Verifier**."
)

CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 100

if "rag_verified_response_cache" not in st.session_state:
    st.session_state["rag_verified_response_cache"] = {}


def _cache_key(api_url: str, query: str, top_n_candidates: int, top_n_rerank: int, version: str) -> str:
    material = json.dumps(
        {
            "api_url": api_url,
            "query": query,
            "top_n_candidates": top_n_candidates,
            "top_n_rerank": top_n_rerank,
            "version": version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _get_cached_response(key: str):
    cache = st.session_state["rag_verified_response_cache"]
    item = cache.get(key)
    if not item:
        return None
    if time.time() - item["saved_at"] > CACHE_TTL_SECONDS:
        cache.pop(key, None)
        return None
    return item["data"]


def _save_cached_response(key: str, data: dict) -> None:
    cache = st.session_state["rag_verified_response_cache"]
    cache[key] = {"saved_at": time.time(), "data": data}
    for old_key, item in list(cache.items()):
        if time.time() - item.get("saved_at", 0) > CACHE_TTL_SECONDS:
            cache.pop(old_key, None)
    while len(cache) > CACHE_MAX_ENTRIES:
        oldest = min(cache, key=lambda k: cache[k].get("saved_at", 0))
        cache.pop(oldest, None)


def _parse_stream(response, progress, draft_placeholder):

    draft = ""
    final_data = None
    backend_error = None
    response.encoding = "utf-8"

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        raw_event = line[len("data:"):].strip()
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "status":
            progress.write(event.get("message", "Processing query..."))
        elif event_type == "token":
            token = event.get("text", "")
            if token:
                draft += token
                draft_placeholder.markdown(
                    "**Provisional draft — citations and claims have not yet been verified.**\n\n"
                    + draft
                )
        elif event_type == "complete":
            final_data = event.get("data")
        elif event_type == "error":
            backend_error = event

    return final_data, backend_error


st.sidebar.header("Pipeline Configuration")
default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
api_base_url = st.sidebar.text_input(
    "FastAPI Base URL", value=default_api_url
).strip().rstrip("/")
top_n_candidates = st.sidebar.slider(
    "Initial retrieval candidates (BM25 + vector)",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
)
top_n_rerank = st.sidebar.slider(
    "Reranked chunks in context",
    min_value=1,
    max_value=20,
    value=8,
    step=1,
)

st.sidebar.caption("Completed responses are cached per browser session for 5 minutes.")
if st.sidebar.button("Clear cached query results"):
    st.session_state["rag_verified_response_cache"].clear()
    st.sidebar.success("Cached query results cleared.")

st.sidebar.markdown("---")
if st.sidebar.button("Check Backend Health"):
    try:
        health_resp = requests.get(f"{api_base_url}/health", timeout=20)
        if health_resp.status_code == 200:
            st.sidebar.success("Backend is online and ready.")
        else:
            st.sidebar.warning(f"Backend status: HTTP {health_resp.status_code}")
    except requests.exceptions.RequestException as exc:
        st.sidebar.error(f"Backend unavailable: {exc}")

with st.form("query_form"):
    user_query = st.text_area(
        "Enter your question:",
        value="What are the transparency obligations for high-risk AI systems?",
        height=100,
    )
    submit_button = st.form_submit_button("Submit Query", type="primary")

if submit_button:
    normalized_query = user_query.strip()
    if not normalized_query:
        st.warning("Enter a question before submitting.")
    elif not api_base_url:
        st.error("Enter the FastAPI base URL in the sidebar.")
    else:
        cache_version = os.getenv("RAG_CACHE_VERSION", "dev-1")
        key = _cache_key(
            api_base_url,
            normalized_query,
            top_n_candidates,
            top_n_rerank,
            cache_version,
        )
        data = _get_cached_response(key)
        was_cached = data is not None

        if was_cached:
            st.info("Loaded a previously completed response from this session's cache.")
        else:
            progress_placeholder = st.empty()
            draft_placeholder = st.empty()
            try:
                with st.status("Processing your question…", expanded=True) as progress:
                    progress.write("Connecting to the streaming query endpoint.")
                    payload = {
                        "query": normalized_query,
                        "top_n_candidates": top_n_candidates,
                        "top_n_rerank": top_n_rerank,
                    }

                    with requests.post(
                        f"{api_base_url}/query/stream",
                        json=payload,
                        stream=True,
                        timeout=(10, 660),
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        if response.status_code != 200:
                            try:
                                error_body = response.json()
                            except ValueError:
                                error_body = {"detail": response.text or "Backend returned an empty response."}
                            detail = error_body.get("detail", "Unknown backend error")
                            if response.status_code == 404:
                                st.warning(str(detail))
                            else:
                                st.error(f"Backend error {response.status_code}: {detail}")
                            st.stop()

                        data, backend_error = _parse_stream(response, progress, draft_placeholder)

                    if backend_error:
                        draft_placeholder.empty()
                        if backend_error.get("status_code") == 404:
                            st.warning(backend_error.get("detail", "No relevant context found."))
                        else:
                            st.error(
                                f"Backend error {backend_error.get('status_code', 500)}: "
                                f"{backend_error.get('detail', 'Unknown error')}"
                            )
                        st.stop()

                    if not isinstance(data, dict):
                        st.error("The stream ended without a completed, verified result.")
                        st.stop()

                    progress.update(
                        label="Answer generated and verification completed",
                        state="complete",
                        expanded=False,
                    )

                _save_cached_response(key, data)
                draft_placeholder.empty()

            except requests.exceptions.Timeout:
                draft_placeholder.empty()
                st.error(
                    "The backend request timed out. Check FastAPI and Ollama logs before retrying."
                )
                st.stop()
            except requests.exceptions.ConnectionError:
                draft_placeholder.empty()
                st.error(
                    "Could not connect to FastAPI. Ensure the backend is running and the base URL is correct."
                )
                st.stop()
            except requests.exceptions.RequestException as exc:
                draft_placeholder.empty()
                st.error(f"Request failed: {exc}")
                st.stop()

        answer = data.get("answer", "")
        audit = data.get("verification_audit", {}) or {}
        sources = data.get("sources", []) or []
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Generated Answer — Final")
            if answer:
                st.markdown(answer)
            else:
                st.warning("The backend returned an empty answer.")

            with st.expander(f"Retrieved Context Sources ({len(sources)} items)"):
                if not sources:
                    st.warning("The backend returned no source documents.")

                for idx, src in enumerate(sources, 1):
                    art_id = (
                        src.get("article_number")
                        or src.get("article_id")
                        or f"Source {idx}"
                    )
                    title = src.get("title") or ""

                    source_text = (
                        src.get("full_text")
                        or src.get("text")
                        or src.get("content")
                        or ""
                    )

                    selected_chunks = (
                        src.get("selected_chunks")
                        or src.get("chunks")
                        or []
                    )

                    # Build readable text from selected chunks if necessary.
                    if not source_text and selected_chunks:
                        source_parts = []

                        for chunk in selected_chunks:
                            chunk_text = str(chunk.get("text") or "").strip()
                            citation = (
                                chunk.get("citation_label")
                                or chunk.get("chunk_id")
                                or ""
                            )

                            if chunk_text:
                                if citation:
                                    source_parts.append(
                                        f"[{citation}]\n{chunk_text}"
                                    )
                                else:
                                    source_parts.append(chunk_text)

                        source_text = "\n\n".join(source_parts)

                    heading = f"**{art_id}**"
                    if title:
                        heading += f" — {title}"

                    st.markdown(heading)

                    if selected_chunks:
                        for chunk in selected_chunks:
                            chunk_text = str(chunk.get("text") or "").strip()
                            citation_label = chunk.get("citation_label")
                            paragraph_number = chunk.get("paragraph_number")

                            if not chunk_text:
                                continue

                            if citation_label:
                                st.markdown(f"**{citation_label}**")
                            elif paragraph_number:
                                st.caption(f"Paragraph {paragraph_number}")

                            st.text(chunk_text)
                    elif source_text.strip():
                        excerpt_limit = 1500
                        excerpt = source_text[:excerpt_limit]

                        if len(source_text) > excerpt_limit:
                            excerpt += "\n\n[Excerpt truncated]"

                        st.text(excerpt)

                    else:
                        st.warning("The source contains no displayable text.")
                        st.caption(
                            "Available fields: " + ", ".join(sorted(src.keys()))
                        )
                    st.divider()
        with col2:
            st.subheader("NLI Guardrail Audit")
            is_faithful = audit.get("is_faithful", False)
            hallucination_rate = audit.get("hallucination_rate", 0.0)
            claims = audit.get("claims", []) or []

            has_uncited_claims = any(
                "UNCITED" in str(claim.get("status", "")).upper()
                for claim in claims
            )

            metric_col1, metric_col2 = st.columns(2)

            if not is_faithful:
                metric_col1.metric(
                    "Status", "FLAGGED", delta="Unfaithful", delta_color="inverse"
                )
            elif has_uncited_claims:
                metric_col1.metric(
                    "Status",
                    "CITATION GAP",
                    delta="Some claims lack citations",
                    delta_color="off",
                )
            else:
                metric_col1.metric("Status", "VERIFIED", delta="Faithful")

            metric_col2.metric(
                "Hallucination Rate",
                f"{float(hallucination_rate) * 100:.1f}%"
            )


            st.markdown("### Claim Breakdown")
            if not claims:
                st.caption("No claim-level audit details were returned.")
            for claim_item in claims:
                claim_text = claim_item.get("claim", "")
                status_text = claim_item.get("status", "UNKNOWN").upper()
                confidence = claim_item.get("confidence")
                flagged = claim_item.get("flagged", False)

                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                    confidence_label = f"(Confidence: {confidence:.2f})"
                else:
                    confidence_label = "(Confidence unavailable)"

                message = f"**[{status_text}]** {confidence_label}\n\n{claim_text}"
                if flagged:
                    st.error(message)
                else:
                    st.success(message)
