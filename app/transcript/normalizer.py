from __future__ import annotations


ASR_REPLACEMENTS = {
    "晴天珠": "擎天柱",
    "晴天柱": "擎天柱",
    "买个床": "威震天",
    "卖个床": "威震天",
    "麦个床": "威震天",
    "居住反派": "霸天虎反派",
    "便衣中": "变一中",
    "服衣中": "变一中",
    "复火": "复活",
    "火众员": "火种源",
    "做了情章": "堕落金刚",
    "情章": "金刚",
}


def normalize_asr_text(text: str) -> str:
    normalized = text
    for source, target in ASR_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())
