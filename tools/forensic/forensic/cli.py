"""Command-line entry points exposed via the ``forensic`` package.

Three subcommands:
    verify    Run EvidenceVerifier on a JSONL log and print a JSON report.
    index     Build/refresh the Tier-2 SQLite database.
    anchor    Invoke ``ots stamp`` on a JSONL log to produce a daily anchor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .indexer import EvidenceIndexer
from .logger import ForensicLogger
from .verifier import EvidenceVerifier


def _cmd_verify(args: argparse.Namespace) -> int:
    verifier = EvidenceVerifier(args.log)
    report = verifier.verify()
    json.dump(report.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if report.ok else 1


def _cmd_index(args: argparse.Namespace) -> int:
    with EvidenceIndexer(args.db) as idx:
        stats = idx.ingest_log(args.log, gap_seconds=args.gap_seconds)
    json.dump(stats, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_anchor(args: argparse.Namespace) -> int:
    # Construct a minimal logger handle that knows the file path; we don't
    # write a record, we only invoke the anchoring helper.
    fl = ForensicLogger(args.log, run_id="anchor-only", create_parents=False)
    try:
        sidecar = fl.anchor_with_opentimestamps(ots_binary=args.ots_binary)
    finally:
        fl.close()
    print(str(sidecar))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forensic", description="Tier-1 forensic log tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify hash chain and structure")
    v.add_argument("log", type=Path, help="path to the JSONL log file")
    v.set_defaults(func=_cmd_verify)

    i = sub.add_parser("index", help="ingest JSONL into the Tier-2 SQLite store")
    i.add_argument("log", type=Path)
    i.add_argument("--db", type=Path, required=True, help="path to SQLite DB")
    i.add_argument("--gap-seconds", type=int, default=1800,
                   help="implicit session gap (default 1800s = 30 min)")
    i.set_defaults(func=_cmd_index)

    a = sub.add_parser("anchor", help="OpenTimestamps anchor a JSONL log")
    a.add_argument("log", type=Path)
    a.add_argument("--ots-binary", default="ots")
    a.set_defaults(func=_cmd_anchor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
