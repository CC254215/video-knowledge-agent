from app.retrieval.evidence import compute_evidence_agreement


def evidence_agreement_metric(query_ids: list[str], evidence_ids: list[str]) -> float:
    return compute_evidence_agreement(query_ids, evidence_ids)
