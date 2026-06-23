"""
forensic — Tier-1 forensic logging, evidence indexing and chain verification
for the thesis "From Attack to Evidence: An Application-Forensic Approach to
Membership Inference and Secret Extraction in LLMs" (Politecnico di Torino).

Public API:
    ForensicLogger      — append-only JSON Lines logger with SHA-256 hash chain
    SessionManifest     — helper for the first record of every run
    EvidenceIndexer     — derives a queryable SQLite Tier-2 store from Tier-1
    EvidenceVerifier    — verifies hash-chain integrity end-to-end
    pseudonymize        — keyed pseudonymisation of user identifiers
    canonical_json      — deterministic serialisation used for hashing
"""
from .logger import ForensicLogger
from .manifest import SessionManifest, build_manifest_payload
from .indexer import EvidenceIndexer
from .verifier import EvidenceVerifier, VerificationReport
from .hashing import canonical_json, sha256_hex, pseudonymize
from .schema import SCHEMA_VERSION, GENESIS_HASH, EventType, Environment, DefenseOutcome

__all__ = [
    "ForensicLogger",
    "SessionManifest",
    "build_manifest_payload",
    "EvidenceIndexer",
    "ApplicationForensicLogger",
    "sha256_text",
    "EvidenceVerifier",
    "VerificationReport",
    "canonical_json",
    "sha256_hex",
    "pseudonymize",
    "SCHEMA_VERSION",
    "GENESIS_HASH",
    "EventType",
    # environments E0/E1 + level-2 defences
    "Environment",
    "DefenseOutcome",
    "DefenseConfig",
    "DefenseDecision",
    "Level2Defenses",
    # backend abstraction (server is backend-agnostic)
    "CompletionResult",
    "RetrievedHit",
    "ModelBackend",
    "Retriever",
    "DeterministicMockBackend",
    "DeterministicMockRetriever",
    # level-1 defensive system prompt + secret-leak oracle
    "DEFENSE_SYSTEM_PROMPT",
    "build_secret_groundtruth",
    "evaluate_secret_leak",
    "SecretLeakResult",
    # F-MT multi-turn adaptive campaigns
    "BranchConfig",
    "StructuralOutcome",
    "BranchSignal",
    "observe_outcome",
    "classify_outcome",
    "detect_refusal_opening",
    "choose_next_move",
    "Move",
    "CampaignPlan",
    "build_campaign_plans",
    "TurnRecord",
    "CampaignState",
    "attacker_recognizes_success",
    "CampaignLabel",
    "label_campaign",
    "run_campaign",
    "run_campaigns",
    "CampaignRunResult",
]

from .application import ApplicationForensicLogger, sha256_text
from .backends import (
    CompletionResult,
    DeterministicMockBackend,
    DeterministicMockRetriever,
    ModelBackend,
    Retriever,
    RetrievedHit,
)
from .secret_oracle import (
    DEFENSE_SYSTEM_PROMPT,
    SecretLeakResult,
    build_secret_groundtruth,
    evaluate_secret_leak,
)
from .defenses import DefenseConfig, DefenseDecision, Level2Defenses
from .campaign import (
    BranchConfig,
    BranchSignal,
    CampaignLabel,
    CampaignPlan,
    CampaignState,
    Move,
    StructuralOutcome,
    TurnRecord,
    attacker_recognizes_success,
    build_campaign_plans,
    choose_next_move,
    classify_outcome,
    detect_refusal_opening,
    label_campaign,
    observe_outcome,
)
from .campaign_runner import CampaignRunResult, run_campaign, run_campaigns
