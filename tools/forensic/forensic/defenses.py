"""Level-2 environment defences (E1): rate limiting, query filtering, anomaly detection.

These are PREVENTIVE mechanisms of the target system, active
only in environment E1. They LOG-NOT-FILTER: when a request trips a defence, the
request is still logged (the outcome is an observable residue), the client gets a
4xx, but the dataset keeps the record — never a hole.

Anti-circularity — the design rules this module obeys:
* Criteria are GENERIC, BEHAVIOURAL and DECLARED — per-session request rate,
  prompt length, special-character density, single-character repetition, and
  sequential near-duplication. There is NOT a single attack keyword/phrase in
  this module: defences never match the lexical content of the attacks.
* Each defence CAN trip on plausible benign traffic (documented per defence
  below as the "benign trigger"), so the outcome is not a proxy for the
  attack label. The point is structural: an honest rate limiter also throttles a
  legitimate power user.
* The defence outcome is a residue the detector MAY observe, NEVER a ground-truth
  label of attack/benign. Defences and detector are independent.

Thresholds are PLACEHOLDERS: reasonable and configurable, to be
calibrated empirically later on trial traffic. How much they block is an outcome
to MEASURE (A-axis, descriptive), not a parameter to pre-set. They are NOT tuned
to let attacks through (the attack residuals come from E0).
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

from .schema import DefenseOutcome


@dataclass(frozen=True)
class DefenseConfig:
    """Parameterised, declared thresholds for the level-2 defences (placeholders)."""

    # Rate limiting (per session): at most ``rate_max_requests`` in a sliding
    # window of ``rate_window_seconds``. Benign trigger: a legitimate power
    # user or an authorised batch/compliance scan exceeding the burst rate.
    rate_max_requests: int = 20
    rate_window_seconds: float = 60.0

    # Query filtering (structural, per request). Benign trigger: a user
    # pasting a long log file, a base64 blob, or a table for analysis.
    filter_max_prompt_chars: int = 4000
    filter_max_special_ratio: float = 0.30
    filter_min_len_for_special: int = 40  # ignore special-ratio on very short prompts
    filter_max_char_run: int = 40

    # Anomaly detection (behavioural, per-session sequence). Benign trigger:
    # a user retrying near-identical queries, or a polling/monitoring script.
    anomaly_history_window: int = 5
    anomaly_sim_threshold: float = 0.85
    anomaly_min_similar: int = 3


@dataclass(frozen=True)
class DefenseDecision:
    """Outcome of evaluating the level-2 defences for one request."""

    outcome: str  # a DefenseOutcome value
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.outcome != DefenseOutcome.ACCEPTED.value


# ── structural / behavioural primitives (generic; no attack content) ─────────

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def special_char_ratio(text: str) -> float:
    """Fraction of characters that are neither alphanumeric nor whitespace."""
    if not text:
        return 0.0
    special = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    return special / len(text)


def longest_char_run(text: str) -> int:
    """Length of the longest run of a single repeated character."""
    best = run = 0
    prev = None
    for c in text:
        run = run + 1 if c == prev else 1
        prev = c
        best = max(best, run)
    return best


def _trigrams(text: str) -> set[str]:
    t = _normalize(text)
    if len(t) < 3:
        return {t} if t else set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


def sequence_similarity(a: str, b: str) -> float:
    """Character-trigram Jaccard similarity in [0, 1] (generic, content-agnostic)."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def query_filter_reason(prompt: str, config: DefenseConfig) -> str | None:
    """Return a structural reason to filter the prompt, or ``None`` to accept.

    Purely structural; reasons are declared tokens, never attack patterns.
    """
    if len(prompt) > config.filter_max_prompt_chars:
        return f"length>{config.filter_max_prompt_chars}"
    if (
        len(prompt) >= config.filter_min_len_for_special
        and special_char_ratio(prompt) > config.filter_max_special_ratio
    ):
        return f"special_char_ratio>{config.filter_max_special_ratio}"
    if longest_char_run(prompt) > config.filter_max_char_run:
        return f"char_run>{config.filter_max_char_run}"
    return None


# ── the level-2 defence pipeline (stateful per server instance) ──────────────


@dataclass
class Level2Defenses:
    """The three level-2 defences, evaluated in order; first trip wins.

    Stateful per server lifecycle (rate window + per-session prompt history). Used
    only in E1; in E0 the server never constructs/consults it (outcome stays
    ``accepted``). ``evaluate`` is deterministic given ``now`` (injected in tests).
    """

    config: DefenseConfig = field(default_factory=DefenseConfig)
    _rate: dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque), init=False)
    _history: dict[str, Deque[str]] = field(default_factory=lambda: defaultdict(deque), init=False)
    _similar_count: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)

    def evaluate(
        self,
        *,
        session_id: str,
        user_id: str,
        prompt: str,
        now: float,
    ) -> DefenseDecision:
        cfg = self.config
        key = session_id or user_id

        # 1) Rate limiting (per session). The request WAS received, so it counts
        #    toward the window even when it ends up rate-limited.
        dq = self._rate[key]
        dq.append(now)
        while dq and now - dq[0] > cfg.rate_window_seconds:
            dq.popleft()
        if len(dq) > cfg.rate_max_requests:
            return DefenseDecision(
                DefenseOutcome.RATE_LIMITED.value,
                f"rate>{cfg.rate_max_requests}/{cfg.rate_window_seconds:g}s",
            )

        # 2) Query filtering (structural).
        reason = query_filter_reason(prompt, cfg)
        if reason is not None:
            return DefenseDecision(DefenseOutcome.FILTERED.value, reason)

        # 3) Anomaly detection (sequential near-duplication within the session).
        hist = self._history[key]
        sim = max((sequence_similarity(prompt, p) for p in hist), default=0.0)
        if sim >= cfg.anomaly_sim_threshold:
            self._similar_count[key] += 1
        hist.append(prompt)
        while len(hist) > cfg.anomaly_history_window:
            hist.popleft()
        if self._similar_count[key] >= cfg.anomaly_min_similar:
            return DefenseDecision(
                DefenseOutcome.ANOMALY.value, "repeated_similar_requests"
            )

        return DefenseDecision(DefenseOutcome.ACCEPTED.value, None)


__all__ = [
    "DefenseConfig",
    "DefenseDecision",
    "Level2Defenses",
    "special_char_ratio",
    "longest_char_run",
    "sequence_similarity",
    "query_filter_reason",
]
