from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv

from services.embedding_service import compute_final_alignment_score, _try_precomputed
from services.explanation_service import generate_alignment_explanation
from services.wikipedia_service import (
    get_entity_text,
    get_entity_text_by_title,
    DisambiguationError,
)
from services.auto_ingest import process_pending_ingests
from utils.scoring_utils import relationship_label
from utils.styles import get_app_styles
from utils.ui_utils import (
    render_identity_profile,
    render_score_card,
    render_classification_banner,
    render_bullet,
)

EXPLANATION_FILE_NAME = './utils/explanation.txt'

with open(EXPLANATION_FILE_NAME, 'rt') as file:
    EXPLANATION_TEXT = file.read()

st.set_page_config(
    page_title="Celebrity\u2013Brand Alignment AI",
    page_icon="\U0001f50d",
    layout="centered",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    }
)

# Load environment variables from .env if present.
load_dotenv()

# Inject custom styles
st.markdown(get_app_styles(), unsafe_allow_html=True)

# ── App header ──
st.markdown("""
<div class="app-header">
    <h1>Celebrity\u2013Brand Alignment AI</h1>
    <p class="app-subtitle">Semantic analysis of entity relationships using AI embeddings</p>
    <div class="poc-badge">PROOF OF CONCEPT</div>
</div>
""", unsafe_allow_html=True)

# ── Entity input form ──
with st.form("alignment_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Celebrity / Public Figure")
        entity_a = st.text_input("Entity A", placeholder="e.g. Taylor Swift", label_visibility="collapsed")
        image_slot_a = st.empty()
        expander_slot_a = st.empty()
    with col2:
        st.markdown("##### Brand / Organization")
        entity_b = st.text_input("Entity B", placeholder="e.g. Nike", label_visibility="collapsed")
        image_slot_b = st.empty()
        expander_slot_b = st.empty()

    analyze = st.form_submit_button("Analyze Alignment")

# ── Disambiguation state ──
if "disambig" not in st.session_state:
    st.session_state.disambig = None  # None or dict with disambiguation info


def _resolve_entity(name, expected_type, resolved_title=None):
    """Fetch entity data, using a resolved title when provided."""
    if resolved_title:
        return get_entity_text_by_title(resolved_title, expected_type=expected_type)
    return get_entity_text(name, expected_type=expected_type)


def _run_analysis(entity_a, entity_b, resolved_a=None, resolved_b=None):
    """Execute the full analysis pipeline. Raises DisambiguationError on ambiguity.

    Stores results in session-state cache and records the cache key so the
    results can be re-displayed after a Streamlit rerun.

    After successful analysis, automatically ingests entities that were not
    found in the pre-computed database.
    """
    # Check session-state cache for repeated queries
    cache_key = (entity_a, entity_b, resolved_a, resolved_b)
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    cached = st.session_state.analysis_cache.get(cache_key)

    if not cached:
        # Check for pre-computed identity profiles in the entity database.
        # The lookup name is the resolved title if available, otherwise the input name.
        lookup_a = resolved_a or entity_a
        lookup_b = resolved_b or entity_b
        precomputed_a = _try_precomputed(lookup_a)
        precomputed_b = _try_precomputed(lookup_b)

        with st.spinner("Fetching and processing Wikipedia content..."):
            # Fetch both entities concurrently
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(_resolve_entity, entity_a, "person", resolved_a)
                future_b = executor.submit(_resolve_entity, entity_b, "organization", resolved_b)
                # Retrieve entity A first; if it raises DisambiguationError, let it propagate
                entity_a_data = future_a.result()
                entity_b_data = future_b.result()

        text_a = entity_a_data["text"]
        text_b = entity_b_data["text"]

        with st.spinner("Analyzing alignment..."):
            # Run scoring and explanation generation concurrently.
            # Pass pre-computed profiles to skip identity extraction + embedding for indexed entities.
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_scores = executor.submit(
                    compute_final_alignment_score, text_a, text_b,
                    precomputed_a=precomputed_a, precomputed_b=precomputed_b,
                )
                future_bullets = executor.submit(generate_alignment_explanation, text_a, text_b)
                domain_score, value_score, domain_score_dict, value_score_dict, identity_a, identity_b = future_scores.result()
                bullets = future_bullets.result()

        # Store in session cache
        st.session_state.analysis_cache[cache_key] = {
            "entity_a_data": entity_a_data,
            "entity_b_data": entity_b_data,
            "domain_score": domain_score,
            "value_score": value_score,
            "domain_score_dict": domain_score_dict,
            "value_score_dict": value_score_dict,
            "identity_a": identity_a,
            "identity_b": identity_b,
            "bullets": bullets,
            "display_entity_a": entity_a,
            "display_entity_b": entity_b,
        }

        # Queue entities for background auto-ingestion if they were not precomputed
        pending_ingest = []
        if precomputed_a is None:
            pending_ingest.append({
                "name": lookup_a,
                "type": "person",
                "identity_profile": identity_a,
            })
        if precomputed_b is None:
            pending_ingest.append({
                "name": lookup_b,
                "type": "organization",
                "identity_profile": identity_b,
            })

        if pending_ingest:
            process_pending_ingests(pending_ingest)

    # Remember which analysis to display (survives rerun)
    st.session_state.last_analysis_key = cache_key


def _display_results(cache_key):
    """Render analysis results from the session-state cache."""
    cached = st.session_state.analysis_cache[cache_key]
    entity_a_data = cached["entity_a_data"]
    entity_b_data = cached["entity_b_data"]
    domain_score = cached["domain_score"]
    value_score = cached["value_score"]
    domain_score_dict = cached["domain_score_dict"]
    value_score_dict = cached["value_score_dict"]
    identity_a = cached["identity_a"]
    identity_b = cached["identity_b"]
    bullets = cached["bullets"]
    display_a = cached.get("display_entity_a", cache_key[0])
    display_b = cached.get("display_entity_b", cache_key[1])

    # Show images (fixed-height container keeps cards aligned)
    if entity_a_data.get("image_url"):
        image_slot_a.markdown(
            f'<div class="entity-image-slot"><img src="{entity_a_data["image_url"]}" alt=""></div>',
            unsafe_allow_html=True,
        )
    else:
        image_slot_a.markdown(
            '<div class="entity-image-slot" style="color:#9ca3af;font-size:0.85rem;">No Wikipedia image available</div>',
            unsafe_allow_html=True,
        )

    if entity_b_data.get("image_url"):
        image_slot_b.markdown(
            f'<div class="entity-image-slot"><img src="{entity_b_data["image_url"]}" alt=""></div>',
            unsafe_allow_html=True,
        )
    else:
        image_slot_b.markdown(
            '<div class="entity-image-slot" style="color:#9ca3af;font-size:0.85rem;">No Wikipedia image available</div>',
            unsafe_allow_html=True,
        )

    render_identity_profile(display_a, identity_a, expander_slot_a)
    render_identity_profile(display_b, identity_b, expander_slot_b)

    # ── Classification banner ──
    label = relationship_label(domain_score, value_score)
    render_classification_banner(label)

    # ── Score cards side by side ──
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        render_score_card("Domain Relatedness", domain_score, domain_score_dict)
    with score_col2:
        render_score_card("Value Alignment", value_score, value_score_dict)

    # ── Alignment bullets ──
    positive_count = sum(1 for b in bullets if b.strip().startswith("\u2713"))
    negative_count = sum(1 for b in bullets if b.strip().startswith("\u2717"))

    if negative_count > positive_count:
        st.markdown("#### Relationship Type: Non-aligned")
    elif positive_count > 0:
        st.markdown("#### Alignment Assessment")

    for bullet in bullets:
        render_bullet(bullet)


if analyze:
    if not entity_a or not entity_b:
        st.error("Please provide both entities before analyzing.")
    else:
        # Clear any previous disambiguation state on new analysis
        st.session_state.disambig = None
        st.session_state.pop("last_analysis_key", None)
        try:
            _run_analysis(entity_a, entity_b)
            _display_results(st.session_state.last_analysis_key)
        except DisambiguationError as exc:
            st.session_state.disambig = {
                "entity_name": exc.entity_name,
                "candidates": exc.candidates,
                "entity_a": entity_a,
                "entity_b": entity_b,
                "field": "a" if exc.entity_name.strip().lower() == entity_a.strip().lower() else "b",
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to complete alignment: {exc}")

# ── Show any disambiguation error from a previous rerun ──
if st.session_state.get("disambig_error"):
    st.error(f"Unable to complete alignment: {st.session_state.disambig_error}")
    st.session_state.disambig_error = None

# ── Disambiguation picker ──
if st.session_state.disambig is not None and not st.session_state.get("disambig_confirmed"):
    disambig = st.session_state.disambig
    entity_label = "Celebrity / Public Figure" if disambig["field"] == "a" else "Brand / Organization"
    st.warning(
        f"**\"{disambig['entity_name']}\"** matches a Wikipedia disambiguation page. "
        f"Please select the correct {entity_label.lower()}:"
    )

    options = [f"{title} — {desc}" for title, desc in disambig["candidates"]]
    selected = st.radio(
        f"Select the correct article for \"{disambig['entity_name']}\":",
        options=options,
        key="disambig_radio",
    )

    if st.button("Confirm selection", key="disambig_confirm"):
        idx = options.index(selected)
        chosen_title = disambig["candidates"][idx][0]
        resolved_a = disambig.get("resolved_a")
        resolved_b = disambig.get("resolved_b")
        if disambig["field"] == "a":
            resolved_a = chosen_title
        else:
            resolved_b = chosen_title
        # Store the confirmed selection and rerun so the picker disappears immediately
        st.session_state.disambig_confirmed = {
            "entity_a": disambig["entity_a"],
            "entity_b": disambig["entity_b"],
            "resolved_a": resolved_a,
            "resolved_b": resolved_b,
        }
        st.rerun()

# ── Run analysis after disambiguation was confirmed (picker already hidden) ──
if st.session_state.get("disambig_confirmed"):
    confirmed = st.session_state.pop("disambig_confirmed")
    try:
        _run_analysis(
            confirmed["entity_a"],
            confirmed["entity_b"],
            resolved_a=confirmed["resolved_a"],
            resolved_b=confirmed["resolved_b"],
        )
        st.session_state.disambig = None
        st.session_state.disambig_displayed = True  # Mark that we will display results
        _display_results(st.session_state.last_analysis_key)
    except DisambiguationError as exc:
        # The other entity is also ambiguous — show a new picker
        st.session_state.disambig = {
            "entity_name": exc.entity_name,
            "candidates": exc.candidates,
            "entity_a": confirmed["entity_a"],
            "entity_b": confirmed["entity_b"],
            "field": "a" if exc.entity_name.strip().lower() == confirmed["entity_a"].strip().lower() else "b",
            "resolved_a": confirmed["resolved_a"],
            "resolved_b": confirmed["resolved_b"],
        }
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.session_state.disambig = None
        st.session_state.disambig_error = str(exc)
        st.rerun()

# ── Re-display cached results after a rerun (e.g. disambiguation resolved) ──
if (
    not analyze
    and st.session_state.disambig is None
    and not st.session_state.get("disambig_confirmed")
    and not st.session_state.get("disambig_displayed")
    and st.session_state.get("last_analysis_key") is not None
    and st.session_state.get("analysis_cache", {}).get(st.session_state.last_analysis_key)
):
    _display_results(st.session_state.last_analysis_key)

# Clear the display flag after use
if st.session_state.get("disambig_displayed"):
    st.session_state.disambig_displayed = False

# ── Methodology section ──
st.markdown('<div class="methodology-section"></div>', unsafe_allow_html=True)
with st.expander("How the Scores Work"):
    st.markdown(EXPLANATION_TEXT)

# ── Footer ──
st.markdown("""
<div class="app-footer">
    Data source: Wikipedia REST API &middot; Embeddings &amp; explanation via Gemini &middot; Proof of Concept
</div>
""", unsafe_allow_html=True)
