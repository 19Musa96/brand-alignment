"""Custom CSS styles for the Celebrity–Brand Alignment AI app."""


def get_app_styles() -> str:
    """Return the full custom CSS for the application."""
    return """
<style>
    /* ── Hide default Streamlit chrome ── */
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ── Typography & base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        font-size: 2rem !important;
    }

    /* ── App title area ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* ── PoC badge ── */
    .poc-badge {
        display: inline-block;
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        color: #4338ca;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin: 0.5rem auto 1.2rem auto;
    }

    /* ── Entity image slots – fixed height so profile cards align ── */
    .entity-image-slot img {
        height: 220px;
        width: 100%;
        object-fit: contain;
        border-radius: 8px;
    }
    .entity-image-slot {
        height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }

    /* ── Entity input cards ── */
    .entity-column {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s ease;
    }
    .entity-column:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* ── Identity Profile Card ── */
    .identity-card {
        background: linear-gradient(135deg, #fafbff 0%, #f5f7ff 100%);
        border: 1px solid #e0e4ef;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 0.75rem;
    }
    .identity-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e0e4ef;
    }
    .identity-card-header h3 {
        margin: 0 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    .profile-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
    }
    .profile-field {
        background: #ffffff;
        border: 1px solid #e8ecf4;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .profile-field-full {
        grid-column: 1 / -1;
        background: #ffffff;
        border: 1px solid #e8ecf4;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .field-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6366f1;
        margin-bottom: 0.3rem;
    }
    .field-value {
        font-size: 0.9rem;
        color: #1e293b;
        line-height: 1.45;
    }
    .field-value em {
        color: #475569;
        font-style: italic;
    }

    /* ── Tag pills for lists ── */
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.2rem;
    }
    .tag {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .tag-warning {
        background: #fef3c7;
        color: #92400e;
    }

    /* ── Score cards ── */
    .score-section {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .score-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .score-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #374151;
    }
    .score-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-value-high { color: #059669; }
    .score-value-mid { color: #d97706; }
    .score-value-low { color: #dc2626; }

    /* ── Custom progress bar ── */
    .progress-track {
        width: 100%;
        height: 8px;
        background: #f1f5f9;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    .progress-fill-high { background: linear-gradient(90deg, #34d399, #059669); }
    .progress-fill-mid { background: linear-gradient(90deg, #fbbf24, #d97706); }
    .progress-fill-low { background: linear-gradient(90deg, #f87171, #dc2626); }

    /* ── Classification banner ── */
    .classification-banner {
        text-align: center;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .classification-strong {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #6ee7b7;
        color: #065f46;
    }
    .classification-competitive {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1px solid #fca5a5;
        color: #991b1b;
    }
    .classification-shared {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
        color: #1e40af;
    }
    .classification-low {
        background: linear-gradient(135deg, #f9fafb, #f3f4f6);
        border: 1px solid #d1d5db;
        color: #374151;
    }

    /* ── Alignment bullets ── */
    .bullet-positive {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.6rem 0.85rem;
        background: #f0fdf4;
        border-left: 3px solid #22c55e;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        color: #15803d;
    }
    .bullet-negative {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.6rem 0.85rem;
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        color: #991b1b;
    }

    /* ── Methodology section ── */
    .methodology-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        padding: 1.5rem 0 1rem 0;
        border-top: 1px solid #f3f4f6;
        margin-top: 2rem;
    }

    /* ── Streamlit overrides ── */
    .stForm {
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        padding: 0.6rem 0.85rem !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }
    button[kind="primaryFormSubmit"],
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
    }

    /* ── Hide default metric label styling ── */
    div[data-testid="stMetricValue"] {
        display: none;
    }
</style>
"""


def score_color_class(score: float) -> str:
    """Return the CSS class suffix based on score value."""
    if score >= 65:
        return "high"
    elif score >= 40:
        return "mid"
    return "low"


def classification_class(label: str) -> str:
    """Return the CSS class for a classification label."""
    mapping = {
        "Strong Strategic Alignment": "classification-strong",
        "Competitive or Adversarial Relationship": "classification-competitive",
        "Shared Values but Different Domains": "classification-shared",
        "Low Strategic Connection": "classification-low",
    }
    return mapping.get(label, "classification-low")
