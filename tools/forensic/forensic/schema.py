"""Schema constants for the Tier-1 forensic log.

The schema is intentionally versioned: every change to the canonical record
shape MUST bump SCHEMA_VERSION and the verifier must learn to read both
versions for backward compatibility on existing logs.
"""
from enum import Enum

# Bump on any breaking change to the record envelope.
SCHEMA_VERSION = "1.1"

# Genesis sentinel for the first record's prev_hash field.
GENESIS_HASH = "0" * 64


class EventType(str, Enum):
    """Closed vocabulary of event types written to the Tier-1 stream.

    Keep this list deliberately small. Sub-typing happens inside ``payload``.
    """

    # Lifecycle
    MANIFEST = "manifest"            # first record of every session/run
    SESSION_OPEN = "session_open"    # explicit session start (post-manifest)
    SESSION_CLOSE = "session_close"  # explicit session end
    HEARTBEAT = "heartbeat"          # periodic liveness marker

    # Model interaction
    PROMPT = "prompt"                # user/attacker input received
    COMPLETION = "completion"        # model output produced
    LOGPROBS = "logprobs"            # top-k logprobs (controlled experimental mode)

    # Attack steps (emitted by the applicative traffic simulators)
    ATTACK_STEP = "attack_step"      # iteration step of an attack simulator
    MIA_SCORE = "mia_score"          # per-sample MIA scorer output
    CANARY_PROBE = "canary_probe"    # canary extraction attempt (auxiliary scenario)
    RAG_RETRIEVAL = "rag_retrieval"  # RAG retriever invocation

    # Detector
    DETECTION_EVENT = "detection_event"  # detector output

    # Operational
    ERROR = "error"
    NOTE = "note"                    # free-form annotation (human-readable)


# Streams: physical separation of forensic evidence from experimental ground truth.
class Stream(str, Enum):
    FORENSIC = "forensic"
    GROUNDTRUTH = "groundtruth"


# Payload-level vocabularies. These are sub-type fields written
# INSIDE ``payload`` (the documented extension point), not new envelope keys, so
# they do NOT change the canonical record shape and do NOT require a
# SCHEMA_VERSION bump or verifier changes.
class Environment(str, Enum):
    """Which of the two environments produced a record (B-axis comparison).

    The log schema is IDENTICAL across the two; ``environment`` is the metadatum
    that distinguishes them. E0 = zero level-2 defences; E1 = level-2 defences
    active. The level-1 defensive system prompt is present in both.
    """

    E0 = "E0"
    E1 = "E1"


class DefenseOutcome(str, Enum):
    """Outcome of the level-2 environment defences for one request (a residue).

    Written in the FORENSIC stream as an observable residue the detector MAY use,
    NOT a ground-truth label of attack/benign (defences and detector are
    independent). In E0 the value is always ``accepted`` (no level-2 defence);
    the field exists in both environments so the schema stays identical.
    """

    ACCEPTED = "accepted"
    RATE_LIMITED = "rate_limited"
    FILTERED = "filtered"
    ANOMALY = "anomaly"


# Reserved keys at the top level of every record. Payload-specific keys go inside ``payload``.
ENVELOPE_KEYS = frozenset({
    "schema_version",
    "record_id",
    "ts_iso",
    "ts_monotonic_ns",
    "stream",
    "run_id",
    "session_id",
    "user_pseudonym",
    "event_type",
    "payload",
    "prev_hash",
    "record_hash",
})
