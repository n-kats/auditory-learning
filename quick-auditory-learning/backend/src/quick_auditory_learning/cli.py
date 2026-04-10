from __future__ import annotations

import argparse
from pathlib import Path

from quick_auditory_learning.importer import import_jsonl, sync_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quick-auditory-learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-jsonl", help="Import arXiv JSONL records")
    import_parser.add_argument("path", type=Path)

    sync_parser = subparsers.add_parser("sync-jsonl", help="Import JSONL when the file changed")
    sync_parser.add_argument("path", type=Path)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "import-jsonl":
        result = import_jsonl(args.path)
        print(f"imported={result.imported} updated={result.updated}")
        return 0
    if args.command == "sync-jsonl":
        result = sync_jsonl(args.path)
        if result is None:
            print("imported=0 updated=0")
        else:
            print(f"imported={result.imported} updated={result.updated}")
        return 0
    parser.error("unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
