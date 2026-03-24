import html

import streamlit as st

from utils.styles import score_color_class, classification_class


def _esc(text):
    """HTML-escape a string for safe rendering."""
    return html.escape(str(text))


def _render_tags(items, css_class="tag"):
    """Render a list of items as styled tag pills."""
    tags = "".join(
        f'<span class="{css_class}">{_esc(item.strip())}</span>'
        for item in items
        if item.strip()
    )
    return f'<div class="tag-container">{tags}</div>'


def render_identity_profile(name, identity, expander_slot):
    """Render a sleek identity profile card for an entity."""
    with expander_slot.container():
        with st.expander("View Identity Profile", expanded=False):
            card_html = f"""
            <div class="identity-card">
                <div class="identity-card-header">
                    <h3>Identity Profile</h3>
                </div>
                <div class="profile-grid">
                    <div class="profile-field">
                        <div class="field-label">Primary Domain</div>
                        <div class="field-value">{_esc(identity['primary_domain'])}</div>
                    </div>
                    <div class="profile-field">
                        <div class="field-label">Economic Model</div>
                        <div class="field-value">{_esc(identity['economic_model'])}</div>
                    </div>
                    <div class="profile-field-full">
                        <div class="field-label">Sub Domains</div>
                        {_render_tags(identity['sub_domains'])}
                    </div>
                    <div class="profile-field-full">
                        <div class="field-label">Core Mission</div>
                        <div class="field-value"><em>{_esc(identity['core_mission'])}</em></div>
                    </div>
                    <div class="profile-field-full">
                        <div class="field-label">Value Signals</div>
                        {_render_tags(identity['value_signals'])}
                    </div>
                    <div class="profile-field">
                        <div class="field-label">Cultural Positioning</div>
                        <div class="field-value">{_esc(identity['cultural_positioning'])}</div>
                    </div>
                    <div class="profile-field">
                        <div class="field-label">Power Positioning</div>
                        <div class="field-value" style="font-weight:600;">{_esc(identity['power_positioning'])}</div>
                    </div>
                    <div class="profile-field-full">
                        <div class="field-label">Controversy Themes</div>
                        {_render_tags(identity['controversy_themes'], css_class="tag tag-warning")}
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


def render_score_card(title, score, detail_dict=None):
    """Render a styled score card with custom progress bar."""
    color = score_color_class(score)
    st.markdown(f"""
    <div class="score-section">
        <div class="score-header">
            <span class="score-title">{_esc(title)}</span>
            <span class="score-value score-value-{color}">{score:.1f}</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill progress-fill-{color}" style="width: {min(score, 100):.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_classification_banner(label):
    """Render the relationship classification as a styled banner."""
    css_class = classification_class(label)
    st.markdown(f"""
    <div class="classification-banner {css_class}">
        {_esc(label)}
    </div>
    """, unsafe_allow_html=True)


def render_bullet(bullet_text):
    """Render a single alignment bullet with styling."""
    text = bullet_text.strip()
    if text.startswith("\u2713"):
        content = _esc(text[1:].strip())
        st.markdown(f"""
        <div class="bullet-positive">
            <span>\u2713</span><span>{content}</span>
        </div>
        """, unsafe_allow_html=True)
    elif text.startswith("\u2717"):
        content = _esc(text[1:].strip())
        st.markdown(f"""
        <div class="bullet-negative">
            <span>\u2717</span><span>{content}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(text)
