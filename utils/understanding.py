import re

SOLVING_WORDS = ["calculat", "determin", "comput", "solv", "perform", "set up", "formulat"]
STEP_TITLE_PATTERN = re.compile(r"^[ \t]*\d+\.\s+\*\*(.+?)\*\*", re.M)


def extract_understanding(
    reasoning: str,
) -> str | None:
    for match in STEP_TITLE_PATTERN.finditer(reasoning):
        if any(word in match.group(1).lower() for word in SOLVING_WORDS):
            return reasoning[: match.start()]
    return None
