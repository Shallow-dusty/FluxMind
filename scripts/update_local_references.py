#!/usr/bin/env python3
"""Update local Codex/Claude references from FluxMind's old index to the new one.

This script is intentionally conservative:

- dry-run by default;
- only touches known local config files;
- writes a `.bak-fluxmind-11` backup before modifying each file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    "/home/shallow/04.AI-Prism/80.FluxMind": "/home/shallow/04.AI-Prism/11.FluxMind",
    "/home/shallow/04.AI-Prism/90.Archive/80-FluxMind": "/home/shallow/04.AI-Prism/90.Archive/11-FluxMind-PreFormal",
}

TARGETS = [
    Path("/home/shallow/.codex/config.toml"),
    Path("/home/shallow/.claude.json"),
]


def rewrite_text(text: str) -> tuple[str, int]:
    count = 0
    for old, new in REPLACEMENTS.items():
        hits = text.count(old)
        if hits:
            text = text.replace(old, new)
            count += hits
    return text, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes")
    args = parser.parse_args()

    total = 0
    for path in TARGETS:
        if not path.exists():
            print(f"skip missing {path}")
            continue

        original = path.read_text(encoding="utf-8")
        updated, count = rewrite_text(original)
        total += count
        mode = "update" if args.apply else "would update"
        print(f"{mode:12} {path} replacements={count}")

        if args.apply and count:
            backup = path.with_name(f"{path.name}.bak-fluxmind-11")
            backup.write_text(original, encoding="utf-8")
            path.write_text(updated, encoding="utf-8")

    if not args.apply:
        print("dry-run only; rerun with --apply from a writable shell")

    return 0 if total or not args.apply else 1


if __name__ == "__main__":
    raise SystemExit(main())
