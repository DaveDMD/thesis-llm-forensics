"""Anti-circular textual/behavioural session features.

The detector should recognise the PROMPT patterns that characterise extraction /
probing campaigns — but WITHOUT keyword matching (the prompt-side keyword features
would be circular and are not used). These are STRUCTURAL/behavioural
observables of the session's prompts, computable by anyone from the logged text,
with no ground-truth label and no planted phrase:

* **degeneracy** — divergence attacks repeat a token (low-entropy prompt);
* **self-similarity** — systematic enumeration reuses near-duplicate prompts;
* **incompleteness** — prefix-continuation submits truncated text (a prefix ends
  mid-sentence, without terminal punctuation).

These are session-level features derived from the session's prompts. They are merged
into the per-session aggregate (``pile_detector.aggregate_sessions``) so both the
offline and the online detector can key on the extraction/probing prompt pattern,
anti-circularly.
"""
from __future__ import annotations

import re
from collections import Counter
from statistics import mean

_WS = re.compile(r"\s+")
_TERMINAL = ".!?\"')]}"


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def prompt_degeneracy(prompt: str) -> float:
    """Token-repetition degeneracy in [0, 1]: 0 when all tokens are distinct, 1
    when a single token repeats throughout (the divergence signature). Uses
    ``(max_freq - 1) / (n - 1)`` so it measures REPETITION, not prompt shortness."""
    toks = _norm(prompt).split()
    n = len(toks)
    if n <= 1:
        return 0.0
    return (Counter(toks).most_common(1)[0][1] - 1) / (n - 1)


def prompt_incomplete(prompt: str) -> bool:
    """True when the prompt ends WITHOUT terminal punctuation — a truncated prefix
    (the prefix-continuation signature), as opposed to a full sentence/question."""
    t = _norm(prompt)
    return bool(t) and t[-1] not in _TERMINAL


def char_ngrams(text: str, n: int = 4) -> set[str]:
    t = _norm(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mean_pairwise_similarity(prompts: list[str], *, cap: int = 30) -> float:
    """Mean pairwise char-ngram Jaccard over the session's prompts (enumeration /
    near-duplicate probing). Capped to the first ``cap`` prompts to bound O(n^2)."""
    grams = [char_ngrams(p) for p in prompts[:cap]]
    sims = [jaccard(grams[i], grams[j]) for i in range(len(grams)) for j in range(i + 1, len(grams))]
    return mean(sims) if sims else 0.0


def session_text_features(prompts: list[str]) -> dict[str, float]:
    """Session-level anti-circular textual features from the session's prompts."""
    degen = [prompt_degeneracy(p) for p in prompts]
    incompl = [1.0 if prompt_incomplete(p) else 0.0 for p in prompts]
    return {
        "feature_session_prompt_degeneracy_max": max(degen) if degen else 0.0,
        "feature_session_prompt_degeneracy_mean": mean(degen) if degen else 0.0,
        "feature_session_prompt_incomplete_rate": mean(incompl) if incompl else 0.0,
        "feature_session_prompt_self_similarity": _mean_pairwise_similarity(prompts),
    }


def response_novelty(text: str, n: int = 3) -> float:
    """Distinct word-n-gram ratio of a response in (0, 1]: unique n-grams / total.

    LOW = internally repetitive / looping output — the degenerate-extraction /
    sampling-regurgitation fingerprint; 1.0 = all n-grams distinct. Observable from
    the response alone (no label). Too-short responses → 1.0 (no repetition signal)."""
    toks = _norm(text).split()
    if len(toks) < n:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def session_chaining_features(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Multi-turn fingerprints from the session's ORDERED (prompt, response) pairs.

    Candidate anti-circular features for the stealth (adaptive) blind spot, to be
    validated OOD before being trusted:
    * response novelty (mean / min) — low = repetitive/regurgitated output;
    * chaining rate — fraction of response[N] char-ngrams that reappear in
      prompt[N+1] (the attacker feeds the model's own output back to refine the
      extraction — the adaptive-multi-turn signature);
    * prompt growth — fraction of consecutive turns whose prompt grows (monotone
      context accretion across the conversation).
    All computed from the logged text only; no ground-truth label."""
    prompts = [p or "" for p, _ in pairs]
    responses = [r or "" for _, r in pairs]
    nov = [response_novelty(r) for r in responses if r.strip()]
    chain = []
    for i in range(len(pairs) - 1):
        rg = char_ngrams(responses[i])
        pg = char_ngrams(prompts[i + 1])
        chain.append(len(rg & pg) / len(rg) if rg else 0.0)
    plens = [len(p) for p in prompts]
    growth = (sum(1 for i in range(len(plens) - 1) if plens[i + 1] > plens[i]) / (len(plens) - 1)
              if len(plens) > 1 else 0.0)
    return {
        "feature_session_response_novelty_mean": mean(nov) if nov else 1.0,
        "feature_session_response_novelty_min": min(nov) if nov else 1.0,
        "feature_session_chaining_rate": mean(chain) if chain else 0.0,
        "feature_session_prompt_growth": growth,
    }


__all__ = [
    "prompt_degeneracy",
    "prompt_incomplete",
    "char_ngrams",
    "jaccard",
    "session_text_features",
    "response_novelty",
    "session_chaining_features",
]
