import streamlit as st
from dotenv import load_dotenv

from services.embedding_service import compute_alignment_score
from services.explanation_service import generate_alignment_explanation
from services.wikipedia_service import get_entity_text


st.set_page_config(
    page_title="Celebrity–Brand Alignment AI (PoC)",
    page_icon="🔍",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    }
)

# Load environment variables from .env if present.
load_dotenv()

# Hide Streamlit header for client demo
st.markdown(
    "<style>header {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)

st.title("Celebrity–Brand Alignment AI (PoC)")

st.info("This is a Proof of Concept (PoC), not a production system.")

with st.form("alignment_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        entity_a = st.text_input("Entity A (celebrity/public figure)")
        image_slot_a = st.empty()
    with col2:
        entity_b = st.text_input("Entity B (brand/public figure)")
        image_slot_b = st.empty()

    analyze = st.form_submit_button("Analyze Alignment")

if analyze:
    if not entity_a or not entity_b:
        st.error("Please provide both entities before analyzing.")
    else:
        try:
            with st.spinner("Fetching and processing Wikipedia content..."):
                # Entity A: MUST be a person (celebrity/public figure)
                entity_a_data = get_entity_text(
                    entity_a, expected_type="person"
                )
                # Entity B: MUST be an organization (brand/company)
                entity_b_data = get_entity_text(
                    entity_b, expected_type="organization"
                )

            text_a = entity_a_data["text"]
            text_b = entity_b_data["text"]

            with st.spinner("Computing contextual relatedness..."):
                score = compute_alignment_score(text_a, text_b)

            with st.spinner("Determining alignment direction..."):
                bullets = generate_alignment_explanation(text_a, text_b, score)

            # Show images after successful resolution
            if entity_a_data.get("image_url"):
                image_slot_a.image(entity_a_data["image_url"], use_container_width=True)
            else:
                image_slot_a.caption("No Wikipedia image available")

            if entity_b_data.get("image_url"):
                image_slot_b.image(entity_b_data["image_url"], use_container_width=True)
            else:
                image_slot_b.caption("No Wikipedia image available")

            # Check if all bullets are negative (✗)
            all_negative = all(bullet.strip().startswith("✗") for bullet in bullets)
            
            if all_negative:
                # Reject nonsensical or adversarial-only comparisons
                st.warning("⚠️ **No meaningful alignment could be established between these entities.**")
                st.subheader("Relationship Type")
                st.markdown("**Adversarial / Non-aligned**")
            else:
                # Show normal results with framing
                st.subheader("Relatedness Score")
                st.metric(label="Contextual relatedness (0-100)", value=f"{score:.2f}")
                # st.caption("ℹ️ High relatedness can indicate cooperation OR conflict. See bullets below for direction.")
                
                # Determine if mostly positive or mixed
                positive_count = sum(1 for b in bullets if b.strip().startswith("✓"))
                negative_count = sum(1 for b in bullets if b.strip().startswith("✗"))
                
                if negative_count > positive_count:
                    st.subheader("Relationship Type: Adversarial / Non-aligned")
                elif positive_count > 0:
                    st.subheader("Alignment Assessment")

            # Always show bullets
            for bullet in bullets:
                st.markdown(bullet)

        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to complete alignment: {exc}")

st.caption("Data source: Wikipedia REST API. Embeddings and explanation via Gemini. This is a PoC, not production.")
