SUMMARIZE_PROMPT = """
You are generating a traceable video knowledge report. Every claim must cite segment_id.
If evidence is insufficient, output insufficient_evidence rather than inventing content.
"""

STORYLINE_PROMPT = """
Build a query-guided storyline, not a generic chapter summary. Each node must show how the
video's information advances over time and must cite evidence_segment_ids.
"""

DEEP_ANALYSIS_PROMPT = """
Analyze claims, evidence, caveats, and counterpoints. Do not add insights without segment_id.
"""

ACTION_ITEMS_PROMPT = """
Extract actionable steps only when the transcript supports them. Cite segment_id for each item.
"""

QA_PROMPT = """
Answer the question using only the supplied transcript segments. Cite segment_id and quote evidence.
If the video does not contain enough evidence, say insufficient_evidence.
"""

EVIDENCE_CHECK_PROMPT = """
Check whether the candidate answer is supported by the supplied evidence. Return segment_ids and
insufficient_evidence when needed. This is not a truth proof outside the video.
"""


def summarize_prompt() -> str:
    return SUMMARIZE_PROMPT


def storyline_prompt() -> str:
    return STORYLINE_PROMPT


def deep_analysis_prompt() -> str:
    return DEEP_ANALYSIS_PROMPT


def action_items_prompt() -> str:
    return ACTION_ITEMS_PROMPT


def qa_prompt() -> str:
    return QA_PROMPT


def evidence_check_prompt() -> str:
    return EVIDENCE_CHECK_PROMPT
