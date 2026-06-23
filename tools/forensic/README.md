# forensic — Tier-1 forensic logging for the thesis experiments

Append-only JSON Lines with a SHA-256 hash chain, a Tier-2 SQLite derivation,
end-to-end verification, and optional anchoring via OpenTimestamps.

## Installation

```bash
pip install -e tools/forensic
```

The package has no mandatory runtime dependencies.

## Minimal usage (Python)

```python
import secrets
from forensic import ForensicLogger, EventType, build_manifest_payload, pseudonymize

salt = secrets.token_bytes(32)         # kept in a separate vault; never logged in clear

with ForensicLogger("./logs/run-001/forensic.jsonl", run_id="run-001") as fl:
    fl.append(EventType.MANIFEST, build_manifest_payload(
        run_id="run-001",
        experiment_phase="track_a_baseline",
        model_id="EleutherAI/pythia-1.4b-deduped",
        experiment_config={"seed": 42, "attacks": ["min_k", "loss"]},
        salt=salt,
        repo_path=".",
    ))
    user = pseudonymize("alice@example.org", salt)
    fl.append(EventType.PROMPT, {"text": "..."}, session_id="s-1", user_pseudonym=user)
```

## CLI

```bash
forensic verify  ./logs/run-001/forensic.jsonl
forensic index   ./logs/run-001/forensic.jsonl --db ./logs/tier2.sqlite
forensic anchor  ./logs/run-001/forensic.jsonl     # requires `ots` installed
```

`verify` exits with code `0` if the chain is intact, `1` if it contains errors.
JSON output makes integration into CI / pre-commit pipelines straightforward.

## Record schema (envelope)

```jsonc
{
  "schema_version": "1.1",
  "record_id": "uuid4",
  "ts_iso": "2026-05-11T14:32:01.123456+00:00",
  "ts_monotonic_ns": 1234567890,
  "stream": "forensic",                   // or "groundtruth"
  "run_id": "run-001",
  "session_id": "s-1",                    // null if delegated to the indexer
  "user_pseudonym": "hmac-sha256-hex",    // null for non-user events
  "event_type": "manifest|prompt|completion|mia_score|...",
  "payload": { /* event-specific */ },
  "prev_hash": "sha256-hex",              // GENESIS = "0"*64
  "record_hash": "sha256-hex"             // SHA256(canonical(record \ {record_hash}))
}
```

See `forensic/schema.py` for the closed vocabulary of `event_type`.

## Two-tier architecture

```
                  ┌──────────────────────┐
   experiment ──→ │ forensic.jsonl       │ ◀── primary evidence (append-only, hashed)
                  │ groundtruth.jsonl    │ ◀── blind eval ground truth (separate)
                  └─────────┬────────────┘
                            │  evidence_indexer
                            ▼
                  ┌──────────────────────┐
                  │ tier2.sqlite         │ ◀── derived, query-friendly, reproducible
                  └──────────────────────┘
                            │
                            ▼  detector / analysis
```

The JSONL file is the **primary evidence**. The SQLite store is derived: if lost,
it is regenerated from `forensic index`.

## Forensic guarantees

| Property | Mechanism |
|---|---|
| Per-record integrity | `record_hash` = SHA-256 over canonical JSON |
| Cross-record integrity | `prev_hash[N] == record_hash[N-1]` |
| Deletion detection | chain break at the following line |
| Payload tampering | recomputed `record_hash` no longer matches |
| Line reordering | broken chain + decreasing monotonic clock |
| Environment reproducibility | manifest with git SHA, dataset/model/config hashes |
| External dating | `.ots` sidecar (Bitcoin anchoring via OpenTimestamps) |
| Identity confidentiality | HMAC-SHA256 with a salt never logged in clear |

Stated limitations:
- the chain does not protect against an attacker with **write access** to the
  file before the daily anchoring: such an attacker can rewrite the entire
  chain. OpenTimestamps anchoring is the mechanism that prevents **retroactive**
  rewrites.
- the `record_hash` does not sign the record: anyone who can read the file can
  produce new valid records. Producer authentication would require an asymmetric
  signature (out of scope for the thesis).

## References

- Record schema aligned with **ISO/IEC 27037** (identification, collection,
  preservation) and **NIST SP 800-86** for digital evidence preservation.
- HMAC pseudonymization with a secret salt, in line with GDPR Art. 4(5)
  (pseudonymisation).
- Hash chain inspired by *Schneier & Kelsey (1999), "Secure Audit Logs to
  Support Computer Forensics"*; SHA-256 replaces the older MAC constructions.
- Temporal anchoring via **OpenTimestamps** (Todd, 2016) — RFC 3161 compatible.

## Pseudonymization salt policy

The application-forensic pipeline uses a persistent salt for HMAC-based
pseudonymization. The salt is treated as a secret and must be stored outside the
Git repository, for example in a gitignored `.env` file or in a local secret file.

The salt is not written to the Tier-1 forensic stream. Manifest records include
only a `salt_fingerprint`, which allows runs to be compared for pseudonymization
policy consistency without exposing the underlying secret.

This design intentionally favours cross-session and cross-run correlation of
pseudonymous users, which is required for timeline reconstruction and forensic
analysis. If the salt is compromised, pseudonyms may become linkable to their
source identifiers; therefore, the salt must be handled as sensitive material.
