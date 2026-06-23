"""Session manifest construction.

The manifest is ALWAYS the first record of a Tier-1 log file. It anchors the
run to a reproducible execution context: model artefacts, dataset hashes,
git commit, library versions, and a (hashed) global salt.

The salt itself is never written to disk by this module — only its SHA-256
fingerprint, so that two manifests can be compared without revealing the
secret.
"""
from __future__ import annotations

import dataclasses
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from .hashing import file_sha256, sha256_hex


def _git_info(repo: Path) -> dict:
    """Return commit SHA + dirty flag, or {'unavailable': reason} on failure."""
    repo_arg = str(Path(repo).resolve())
    # ``-c safe.directory`` keeps the capture working when the repository is
    # owned by a different uid than the running process. This is the common
    # case inside the container, where the bind-mounted workspace belongs to
    # the host user and Git would otherwise refuse with a "dubious ownership"
    # error and exit non-zero (recorded as CalledProcessError).
    git_prefix = ["git", "-c", f"safe.directory={repo_arg}", "-C", repo_arg]
    try:
        sha = subprocess.check_output(
            [*git_prefix, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        status = subprocess.check_output(
            [*git_prefix, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        return {"commit_sha": sha, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": type(exc).__name__}


def _library_versions() -> dict:
    """Snapshot of key library versions; failures degrade silently."""
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for lib in ("torch", "transformers", "datasets", "numpy", "scipy"):
        try:
            mod = __import__(lib)
            out[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    # CUDA / GPU info via torch if available.
    try:
        import torch  # type: ignore
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_version"] = torch.version.cuda
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — best-effort snapshot
        pass
    return out


@dataclasses.dataclass
class SessionManifest:
    """Builder for the manifest payload.

    Use ``build_manifest_payload`` for one-shot construction from primitives;
    use this dataclass when the caller already has the components in hand.
    """

    run_id: str
    experiment_phase: str            # free-form snake_case label, e.g. "track_a_baseline", "pipeline_mia_probing", "pipeline_rag_leakage"
    model_id: str                    # human-readable, e.g. "EleutherAI/pythia-1.4b-deduped"
    model_artifact_hashes: Mapping[str, str]  # {filename: sha256}
    dataset_hashes: Mapping[str, str]
    config_hash: str                 # SHA-256 of the experiment-config blob
    salt_fingerprint: str            # SHA-256 of the global pseudonymisation salt
    repo_path: str
    notes: str = ""

    def to_payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "experiment_phase": self.experiment_phase,
            "model": {
                "id": self.model_id,
                "artifact_hashes": dict(self.model_artifact_hashes),
            },
            "datasets": dict(self.dataset_hashes),
            "experiment_config_hash": self.config_hash,
            "salt_fingerprint": self.salt_fingerprint,
            "git": _git_info(Path(self.repo_path)),
            "libraries": _library_versions(),
            "pid": os.getpid(),
            "notes": self.notes,
        }


def build_manifest_payload(
    *,
    run_id: str,
    experiment_phase: str,
    model_id: str,
    model_artifacts: Mapping[str, Path] | None = None,
    dataset_paths: Mapping[str, Path] | None = None,
    experiment_config: Mapping | None = None,
    salt: bytes,
    repo_path: str | Path = ".",
    notes: str = "",
) -> dict:
    """Compute every fingerprint and return a ready-to-log manifest payload.

    *model_artifacts* maps short names (e.g. ``"config.json"``) to filesystem
    paths; the function streams each file through SHA-256. Mapping ``None``
    leaves the section empty (e.g. when the model is loaded from an
    ephemeral HuggingFace cache that the user does not want to re-hash).
    """
    from .hashing import canonical_json

    model_hashes = {
        name: file_sha256(path) for name, path in (model_artifacts or {}).items()
    }
    dataset_hashes = {
        name: file_sha256(path) for name, path in (dataset_paths or {}).items()
    }
    config_hash = sha256_hex(canonical_json(experiment_config or {}))
    salt_fingerprint = sha256_hex(salt)

    manifest = SessionManifest(
        run_id=run_id,
        experiment_phase=experiment_phase,
        model_id=model_id,
        model_artifact_hashes=model_hashes,
        dataset_hashes=dataset_hashes,
        config_hash=config_hash,
        salt_fingerprint=salt_fingerprint,
        repo_path=str(repo_path),
        notes=notes,
    )
    return manifest.to_payload()
