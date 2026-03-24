import streamlit as st
from dotenv import load_dotenv

from services.embedding_service import compute_final_alignment_score
from services.explanation_service import generate_alignment_explanation
from services.wikipedia_service import get_entity_text
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

if analyze:
    if not entity_a or not entity_b:
        st.error("Please provide both entities before analyzing.")
    else:
        try:
            with st.spinner("Fetching and processing Wikipedia content..."):
                entity_a_data = get_entity_text(
                    entity_a, expected_type="person"
                )
                entity_b_data = get_entity_text(
                    entity_b, expected_type="organization"
                )

            text_a = entity_a_data["text"]
            text_b = entity_b_data["text"]

            with st.spinner("Computing contextual relatedness..."):
                domain_score, value_score, domain_score_dict, value_score_dict, identity_a, identity_b = compute_final_alignment_score(text_a, text_b)

            with st.spinner("Determining alignment direction..."):
                bullets = generate_alignment_explanation(text_a, text_b)

            # Show images after successful resolution (fixed-height container keeps cards aligned)
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

            render_identity_profile(entity_a, identity_a, expander_slot_a)
            render_identity_profile(entity_b, identity_b, expander_slot_b)

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

        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to complete alignment: {exc}")

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
