"""Generate concise alignment explanations for two entities."""

import os
from typing import List

from google import genai

_MODEL_NAME = "gemini-3-flash-preview"
_configured: bool = False
_client: genai.Client | None = None


def _configure_client() -> None:
    """Configure the client once using the API key from the environment."""
    global _configured, _client
    if _configured:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is required for explanations."
        )
    _client = genai.Client(api_key=api_key)
    _configured = True


def generate_alignment_explanation(
    summary_a: str, summary_b: str, alignment_score: float
) -> List[str]:
    """
    Produce 3-5 factual bullet points explaining alignment direction.

    Each bullet is prefixed with:
    - ✓ ONLY if: shared goals, mutual benefit, collaboration, value alignment
    - ✗ if: conflict, adversarial history, opposing roles, different domains with no cooperation

    Constraints: bullets must rely only on the provided Wikipedia content and
    avoid speculation or new facts. Output is returned as a list of strings.
    """

    if not summary_a or not summary_a.strip():
        raise ValueError("summary_a must be a non-empty string.")
    if not summary_b or not summary_b.strip():
        raise ValueError("summary_b must be a non-empty string.")

    _configure_client()
    if _client is None:
        raise RuntimeError("Gemini client not configured for explanations.")

    prompt = f"""
You are analyzing BRAND ALIGNMENT between two entities (Entity A and Entity B).

Brand alignment means similarity in:
- Values
- Public persona
- Tone and positioning
- Audience and cultural association
- Thematic focus (e.g. performance, innovation, luxury, family, rebellion)

IMPORTANT DISTINCTIONS:
- Brand alignment is NOT the same as endorsement, sponsorship, or collaboration.
- Lack of partnership does NOT imply misalignment.
- Existing endorsements with competitors should NOT reduce alignment, but may be mentioned as a separate commercial conflict note.

STRICT RULES FOR ✓ (positive alignment):
Use ✓ when there is clear similarity in:
- Values or messaging
- Public image or persona
- Target audience or cultural positioning
- Thematic focus or reputation

STRICT RULES FOR ✗ (misalignment):
Use ✗ ONLY when there is:
- Clear contradiction in values or persona
- Opposing brand positioning (e.g. family-friendly vs provocative)
- Reputational incompatibility

DO NOT classify as ✗ merely because:
- There is no partnership
- They operate independently in the same industry

Produce 3–5 concise bullet points:
- Focus primarily on brand/persona similarity or mismatch
- If relevant, include at most ONE bullet noting commercial conflict (clearly labelled)

Entity A content:
{summary_a}

Entity B content:
{summary_b}

Return ONLY the bullets.
Each line MUST start with exactly '✓' or '✗' followed by a space.
Do not number the bullets.

"""

    response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
    text = getattr(response, "text", None) or ""

    bullets: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Ensure each bullet has the ✓/✗ prefix
        if not stripped.startswith(("✓", "✗")):
            if stripped.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
                # Remove common bullet/list markers
                stripped = stripped.lstrip("-*•0123456789. ").strip()
            # Default to ✗ if no indicator provided (conservative default)
            if stripped:
                stripped = f"✗ {stripped}"
        bullets.append(stripped)

    bullets = [b for b in bullets if b and any(c in b for c in ("✓", "✗"))]
    if not bullets:
        raise ValueError("Gemini explanation did not return any bullet points.")
    if len(bullets) > 5:
        bullets = bullets[:5]
    return bullets


# TODO: Consider lightweight validation of claims if stricter sourcing is needed later.
