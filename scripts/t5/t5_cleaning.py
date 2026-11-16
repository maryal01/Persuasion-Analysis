"""Text clean-up for the T5 posts: the post as it was before the discussion started.

The stored posts contain things that do not belong to the original argument and can give the outcome away:
- the moderator footnote that is appended to every thread,
- paragraphs the OP added later that START with an edit or update marker ("Edit:", "Update -", "EDIT 2:", "**Edit**", ...),
- delta symbols.
Only paragraphs that start with such a marker are removed; the word "edit" elsewhere in a paragraph is left alone.
"""
import re

_FOOTNOTE = re.compile(
    r"(?is)\n\s*(?:[_\-*]{3,}\s*\n\s*)?(?:&gt;|>)?\s*\*?\s*(?:hello, users of cmv! )?"
    r"this is a footnote from (?:the cmv moderators|your moderators).*$")

_EDIT_MARKER = (
    r"^[\s>*_#]*(?:edit|update)s?"                     # "Edit", "UPDATE", "**Edit"
    r"\s*(?:#?\d+|[ivx]{1,4}\b|\([^)\n]{0,40}\)|\[[^\]\n]{0,30}\])?"  # optional number, "II" or "(date)"
    r"(?:\s+(?:for|to)\s+\w+)?"                        # optional "for clarity", "to clarify"
    r"\s*(?:[*_]+|[:\-–—~/]|$)"              # closing "**", ":", "-", ... or the end of the line
)
_EDIT_PARAGRAPH = re.compile(r"(?im)" + _EDIT_MARKER + r"(?s:.*?)(?:\n\s*\n|\Z)")   # the marker line up to the end of its paragraph

_DELTA_SYMBOL = re.compile(r"[∆Δ]|&amp;#8710;|!delta", re.IGNORECASE)


def clean_text(text: str) -> str:
    """The post text without the moderator footnote, the OP's later edit/update paragraphs and delta symbols."""
    text = _FOOTNOTE.sub("", text)
    text = _EDIT_PARAGRAPH.sub("\n\n", text)  # keep a paragraph break where a paragraph was removed
    text = _DELTA_SYMBOL.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
