import re

SOLVING_WORDS = ["calculat", "determin", "comput", "solv", "perform", "set up", "formulat"]
LENGTH_RATIOS = [0.5, 0.75, 1, 1.5, 2]
STEP_TITLE_PATTERN = re.compile(r"^[ \t]*\d+\.\s+\*\*(.+?)\*\*", re.M)


def extract_understanding(
    reasoning: str,
) -> str | None:
    for match in STEP_TITLE_PATTERN.finditer(reasoning):
        step_title = match.group(1).lower()
        for word in SOLVING_WORDS:
            if word in step_title:
                return reasoning[: match.start()]
    return None
