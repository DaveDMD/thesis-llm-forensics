"""Incident-response playbook (orchestration, non-destructive).

Implements an IR playbook: triage, snapshot (model+index+logs), preservation
(WORM + timestamp signature), escalation and reporting, by orchestrating the
building blocks already in the package — the timeline
reconstruction, the attribution heuristics, the hash-chain verifier and the
file hashing — into a reproducible procedure that PRODUCES evidence and a
structured report. It never performs destructive or irreversible actions
(no deletion, no mutation of logs): forensic preservation requires that the
tool only reads, hashes and attests.

Phases
------
1. triage         — summarise suspicious sessions, reconstructed phases, and
                    attribution links; assign a coarse severity.
2. snapshot       — hash the model + index + logs file set into an evidence
                    bundle with an aggregate digest (chain of custody).
3. preservation   — verify the hash-chain (integrity) and record the presence
                    of the OpenTimestamps sidecar (WORM/time-stamping attest).
4. report         — emit a structured IR report combining the above.

The organisational steps (who escalates to whom, SLAs, external
reporting duties) are procedural and belong to the thesis text; this module
produces the technical evidence and the report skeleton those steps consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attribution import correlate_sessions
from .hashing import file_sha256, sha256_hex
from .timeline import (
    PHASE_BENIGN,
    PHASE_EXTRACTION,
    PHASE_INJECTION,
    PHASE_MEMBERSHIP,
    PHASE_RECON,
    reconstruct_timelines,
)
from .verifier import EvidenceVerifier

# Coarse severity thresholds (thesis-defined, documented).
SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

# Phases that constitute a confirmed-attack signal. Reconnaissance alone (benign
# RAG retrieval shares this surface) is a weak signal and does not, by itself,
# raise severity — a deliberate choice to avoid flagging legitimate retrieval.
_STRONG_ATTACK_PHASES = frozenset({PHASE_MEMBERSHIP, PHASE_EXTRACTION, PHASE_INJECTION})


@dataclass
class TriageResult:
    n_sessions: int
    suspicious_sessions: list[str]
    recon_only_sessions: list[str]
    multi_phase_sessions: list[str]
    attribution_links: list[dict[str, Any]]
    severity: str
    phase_summary: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_sessions": self.n_sessions,
            "suspicious_sessions": self.suspicious_sessions,
            "recon_only_sessions": self.recon_only_sessions,
            "multi_phase_sessions": self.multi_phase_sessions,
            "n_attribution_links": len(self.attribution_links),
            "attribution_links": self.attribution_links,
            "severity": self.severity,
            "phase_summary": self.phase_summary,
        }


def _severity_from(n_suspicious: int, n_multi_phase: int, n_links: int) -> str:
    if n_suspicious == 0:
        return SEVERITY_NONE
    if n_multi_phase > 0 or n_links > 0:
        return SEVERITY_HIGH
    if n_suspicious >= 3:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def triage(
    forensic_records: Sequence[Mapping[str, Any]],
    *,
    feature_rows: Sequence[Mapping[str, Any]] | None = None,
    attribution_min_confidence: float = 0.5,
) -> TriageResult:
    """Summarise the incident: suspicious sessions, phases, attribution, severity."""
    timelines = reconstruct_timelines(forensic_records, feature_rows=feature_rows)

    phase_summary: dict[str, int] = {}
    suspicious: list[str] = []
    recon_only: list[str] = []
    multi_phase: list[str] = []
    for sid, tl in timelines.items():
        attack_phases = [e.phase for e in tl.events if e.phase != PHASE_BENIGN]
        for ph in attack_phases:
            phase_summary[ph] = phase_summary.get(ph, 0) + 1
        strong = [ph for ph in attack_phases if ph in _STRONG_ATTACK_PHASES]
        if strong:
            suspicious.append(sid)
        elif any(ph == PHASE_RECON for ph in attack_phases):
            # reconnaissance-only: weak signal, not a confirmed incident
            recon_only.append(sid)
        if len(set(strong)) >= 2:
            multi_phase.append(sid)

    links = [
        l.as_dict()
        for l in correlate_sessions(
            forensic_records, min_confidence=attribution_min_confidence
        )
    ]
    severity = _severity_from(len(suspicious), len(multi_phase), len(links))
    return TriageResult(
        n_sessions=len(timelines),
        suspicious_sessions=sorted(suspicious),
        recon_only_sessions=sorted(recon_only),
        multi_phase_sessions=sorted(multi_phase),
        attribution_links=links,
        severity=severity,
        phase_summary=phase_summary,
    )


@dataclass
class SnapshotResult:
    components: dict[str, dict[str, Any]]
    aggregate_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {"components": self.components, "aggregate_digest": self.aggregate_digest}


def snapshot(
    *,
    log_path: str | Path,
    model_artifacts: Mapping[str, str | Path] | None = None,
    index_paths: Mapping[str, str | Path] | None = None,
) -> SnapshotResult:
    """Hash the model + index + logs file set into an evidence bundle.

    Each present file contributes its sha256; the aggregate digest is the hash
    of the sorted per-component digests, giving a single chain-of-custody value
    for the whole snapshot. Missing files are recorded as ``present: false`` so
    the snapshot is honest about what was available.
    """
    components: dict[str, dict[str, Any]] = {}

    def _add(kind: str, name: str, path: str | Path) -> None:
        p = Path(path)
        if p.exists() and p.is_file():
            components[f"{kind}:{name}"] = {
                "path": str(p),
                "present": True,
                "sha256": file_sha256(p),
                "size_bytes": p.stat().st_size,
            }
        else:
            components[f"{kind}:{name}"] = {"path": str(p), "present": False}

    _add("logs", "forensic_stream", log_path)
    for name, path in (model_artifacts or {}).items():
        _add("model", name, path)
    for name, path in (index_paths or {}).items():
        _add("index", name, path)

    digests = sorted(
        c["sha256"] for c in components.values() if c.get("present") and "sha256" in c
    )
    aggregate = sha256_hex("|".join(digests).encode("utf-8"))
    return SnapshotResult(components=components, aggregate_digest=aggregate)


@dataclass
class PreservationResult:
    chain_verified: bool
    total_records: int
    manifest_first: bool
    ots_sidecar_present: bool
    issues: list[dict[str, Any]]
    # Offline OpenTimestamps binding (True/False/None) and attestation state
    # ("confirmed"/"pending"/"none"/None). See verifier.verify_ots_commitment.
    ots_commitment_verified: bool | None = None
    ots_attestation_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_verified": self.chain_verified,
            "total_records": self.total_records,
            "manifest_first": self.manifest_first,
            "ots_sidecar_present": self.ots_sidecar_present,
            "ots_commitment_verified": self.ots_commitment_verified,
            "ots_attestation_status": self.ots_attestation_status,
            "n_issues": len(self.issues),
            "issues": self.issues,
        }


def preservation(*, log_path: str | Path) -> PreservationResult:
    """Verify integrity (hash-chain) and attest WORM/time-stamping presence."""
    report = EvidenceVerifier(str(log_path)).verify()
    d = report.to_dict()
    issues = d.get("issues", [])
    # chain verified iff there are no error-severity issues
    chain_ok = not any(i.get("severity") == "error" for i in issues)
    return PreservationResult(
        chain_verified=chain_ok,
        total_records=d.get("total_records", 0),
        manifest_first=d.get("has_manifest_first", False),
        ots_sidecar_present=d.get("ots_sidecar_present", False),
        ots_commitment_verified=d.get("ots_commitment_verified"),
        ots_attestation_status=d.get("ots_attestation_status"),
        issues=issues,
    )


def run_playbook(
    *,
    log_path: str | Path,
    forensic_records: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]] | None = None,
    model_artifacts: Mapping[str, str | Path] | None = None,
    index_paths: Mapping[str, str | Path] | None = None,
    attribution_min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Run the full IR playbook and return a structured report.

    Non-destructive: reads records, hashes files, verifies the chain, and
    produces a report. Performs no deletion or mutation.
    """
    tri = triage(
        forensic_records,
        feature_rows=feature_rows,
        attribution_min_confidence=attribution_min_confidence,
    )
    snap = snapshot(log_path=log_path, model_artifacts=model_artifacts, index_paths=index_paths)
    pres = preservation(log_path=log_path)

    # Escalation recommendation derived from severity + integrity.
    if not pres.chain_verified:
        escalation = "INTEGRITY_FAILURE: escalate immediately; evidence chain broken"
    elif tri.severity == SEVERITY_HIGH:
        escalation = "escalate to incident lead; preserve snapshot; consider GDPR breach assessment"
    elif tri.severity in (SEVERITY_MEDIUM, SEVERITY_LOW):
        escalation = "open incident ticket; monitor; retain snapshot"
    else:
        escalation = "no escalation; routine retention"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "triage": tri.as_dict(),
        "snapshot": snap.as_dict(),
        "preservation": pres.as_dict(),
        "escalation_recommendation": escalation,
        "report_note": (
            "Technical IR evidence and report skeleton. Organisational steps "
            "(escalation chain, SLAs, external reporting duties) are defined in "
            "the thesis text. This tool performs no destructive action."
        ),
    }


__all__ = [
    "TriageResult",
    "triage",
    "SnapshotResult",
    "snapshot",
    "PreservationResult",
    "preservation",
    "run_playbook",
    "SEVERITY_NONE",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_HIGH",
]
