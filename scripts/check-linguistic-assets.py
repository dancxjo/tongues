#!/usr/bin/env python3
"""Verify pinned linguistic inputs and deterministic generated artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "crates/speaking/assets/linguistic-assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"{label} checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    print(f"{label}: {actual}  {path}")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = {asset["id"]: asset for asset in manifest["assets"]}

    cargo_home = Path(
        os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))
    )
    cmudict_roots = sorted(
        cargo_home.glob("registry/src/*/arpabet_cmudict-2.0.0")
    )
    if not cmudict_roots:
        raise SystemExit("arpabet_cmudict 2.0.0 is missing; run `cargo fetch` first")
    cmudict_root = cmudict_roots[0]
    cmudict = assets["cmudict-0.7b"]
    require_hash(
        cmudict_root / "cmudict/cmudict-0.7b",
        cmudict["source_sha256"],
        "cmudict source",
    )
    package_archives = sorted(
        cargo_home.glob("registry/cache/*/arpabet_cmudict-2.0.0.crate")
    )
    if not package_archives:
        raise SystemExit("arpabet_cmudict package archive is missing; run `cargo fetch` first")
    require_hash(
        package_archives[0],
        cmudict["package_sha256"],
        "cmudict package",
    )

    target_directory = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
    generated = sorted(
        target_directory.glob("debug/build/arpabet_cmudict-*/out/codegen.rs")
    )
    if not generated:
        raise SystemExit(
            "CMUdict generated artifact is missing; run "
            "`CARGO_NET_OFFLINE=true cargo check -p speaking` first"
        )
    for path in generated:
        require_hash(path, cmudict["generated_sha256"], "cmudict generated")

    lexique = assets["lexique383-seed"]
    lexique_path = ROOT / "crates/speaking/src/data/lexicons/Lexique383.tsv"
    require_hash(lexique_path, lexique["source_sha256"], "lexique source")
    require_hash(lexique_path, lexique["generated_sha256"], "lexique generated")


if __name__ == "__main__":
    main()
