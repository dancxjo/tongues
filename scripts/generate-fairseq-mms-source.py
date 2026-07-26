#!/usr/bin/env python3
"""Build the checksum-complete Fairseq MMS catalog source snapshot.

Runtime inference remains Rust-only. This maintainer tool reads Hugging Face's
Git/LFS metadata for the immutable upstream revision and hashes the small
config/vocabulary files without downloading every 145 MB checkpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPOSITORY = "facebook/mms-tts"
DEFAULT_LANGUAGE_INDEX = (
    "https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html"
)
LICENSE_EVIDENCE = (
    "https://huggingface.co/facebook/mms-tts/blob/"
    "44cc7fb408064ef9ea6e7c59130d88cac1274671/README.md"
)


def get_bytes(url: str) -> bytes:
    for attempt in range(8):
        request = urllib.request.Request(
            url, headers={"User-Agent": "tongues-fairseq-catalog/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 7:
                raise
            delay = int(error.headers.get("Retry-After", min(60, 2 ** attempt)))
            print(
                f"fairseq-source: rate limited; retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def get_json(url: str) -> dict | list:
    return json.loads(get_bytes(url))


def language_rows(source: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    pattern = re.compile(r"<p>\s*(.*?)\s*&emsp;\s*(.*?)\s*</p>")
    for model_id, name in pattern.findall(source):
        model_id = html.unescape(model_id).strip()
        name = html.unescape(name).strip()
        if model_id.lower() == "iso code":
            continue
        if model_id in rows:
            raise RuntimeError(f"language index repeats {model_id!r}")
        rows[model_id] = name
    if not rows:
        raise RuntimeError("language index contains no model rows")
    return rows


def file_metadata(sibling: dict) -> dict:
    lfs = sibling.get("lfs")
    if not lfs:
        raise RuntimeError(f"{sibling['rfilename']} is missing LFS SHA-256 metadata")
    return {"sha256": lfs["sha256"], "size_bytes": int(lfs["size"])}


def metadata_by_model(repository: str, revision: str) -> dict[str, dict]:
    encoded = urllib.parse.quote(repository, safe="/")
    info = get_json(
        f"https://huggingface.co/api/models/{encoded}?blobs=true&revision={revision}"
    )
    files: dict[str, dict] = {}
    for sibling in info["siblings"]:
        match = re.fullmatch(r"models/([^/]+)/(G_100000\.pth|config\.json|vocab\.txt)", sibling["rfilename"])
        if match:
            files.setdefault(match.group(1), {})[match.group(2)] = sibling
    return files


def missing_model_metadata(
    repository: str, revision: str, model_id: str
) -> tuple[str, dict]:
    encoded_repository = urllib.parse.quote(repository, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(f"models/{model_id}", safe="/")
    rows = get_json(
        f"https://huggingface.co/api/models/{encoded_repository}/tree/"
        f"{encoded_revision}/{encoded_path}?expand=true"
    )
    return model_id, {
        pathlib.PurePosixPath(row["path"]).name: {
            "rfilename": row["path"],
            "size": row["size"],
            "lfs": row.get("lfs"),
        }
        for row in rows
    }


def fetch_small_model_files(
    repository: str, revision: str, model_id: str
) -> tuple[str, bytes, bytes]:
    base = (
        f"https://huggingface.co/{repository}/resolve/{revision}/models/"
        f"{urllib.parse.quote(model_id, safe='')}"
    )
    return model_id, get_bytes(f"{base}/config.json"), get_bytes(f"{base}/vocab.txt")


def qualifier_metadata(model_id: str) -> tuple[str, str | None]:
    language = model_id.split("-", 1)[0]
    script_match = re.search(r"(?:^|-)script_([^-]+)", model_id)
    return language, script_match.group(1) if script_match else None


def lfs_pointer_metadata(source: bytes, path: pathlib.Path) -> dict:
    text = source.decode("ascii")
    sha = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if not sha or not size:
        raise RuntimeError(
            f"{path} is not a Git LFS pointer; clone with GIT_LFS_SKIP_SMUDGE=1"
        )
    return {"sha256": sha.group(1), "size_bytes": int(size.group(1))}


def source_entry(
    model_id: str,
    language_name: str,
    checkpoint: dict,
    config_bytes: bytes,
    vocab_bytes: bytes,
) -> dict:
    config = json.loads(config_bytes)
    language, script = qualifier_metadata(model_id)
    preprocessing = (
        ["uroman"]
        if config["data"]["training_files"].rsplit(".", 1)[-1].lower() == "uroman"
        else ["lowercase-and-filter-vocab"]
    )
    return {
        "model_id": model_id,
        "language_name": language_name,
        "language": language,
        "script": script,
        # A language id is not evidence that a voice covers one of Tongues'
        # finer pronunciation varieties.
        "varieties": [],
        "preprocessing": preprocessing,
        "sample_rate_hz": int(config["data"]["sampling_rate"]),
        "license": {
            "expression": "CC-BY-NC-4.0",
            "evidence": LICENSE_EVIDENCE,
        },
        "checkpoint": checkpoint,
        "config": {
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "size_bytes": len(config_bytes),
        },
        "vocab": {
            "sha256": hashlib.sha256(vocab_bytes).hexdigest(),
            "size_bytes": len(vocab_bytes),
        },
    }


def write_json_part(path: pathlib.Path, value: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return part


def atomic_json(path: pathlib.Path, value: dict) -> None:
    part = write_json_part(path, value)
    os.replace(part, path)


def source_document(revision: str, entries: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "id": f"fairseq-mms-vits-{revision[:8]}",
        "revision": revision,
        "entries": entries,
    }


def write_progress(path: pathlib.Path, revision: str, entries: list[dict], total: int) -> None:
    progress = source_document(revision, entries)
    progress["progress"] = {
        "complete": False,
        "indexed_entries": len(entries),
        "total_entries": total,
    }
    write_json_part(path, progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision")
    parser.add_argument(
        "--checkout",
        type=pathlib.Path,
        help="Git LFS checkout made with GIT_LFS_SKIP_SMUDGE=1",
    )
    parser.add_argument("--language-index", default=DEFAULT_LANGUAGE_INDEX)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 64:
        raise RuntimeError("--jobs must be in 1..=64")
    repository_url = urllib.parse.quote(args.repository, safe="/")
    if args.revision:
        revision = args.revision
    elif args.checkout:
        revision = subprocess.check_output(
            ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        revision = get_json(
            f"https://huggingface.co/api/models/{repository_url}"
        )["sha"]
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError("revision must resolve to a 40-character Git commit")
    print(f"fairseq-source: revision {revision}", file=sys.stderr)

    if re.match(r"https?://", args.language_index):
        language_index = get_bytes(args.language_index).decode("utf-8")
    else:
        language_index = pathlib.Path(args.language_index).read_text(encoding="utf-8")
    languages = language_rows(language_index)
    print(
        f"fairseq-source: language index contains {len(languages)} models",
        file=sys.stderr,
    )

    if args.checkout:
        entries = []
        model_root = args.checkout / "models"
        checkout_ids = {
            path.name
            for path in model_root.iterdir()
            if path.is_dir()
            and (path / "G_100000.pth").is_file()
            and (path / "config.json").is_file()
            and (path / "vocab.txt").is_file()
        }
        additions = sorted(set(languages) - checkout_ids)
        removals = sorted(checkout_ids - set(languages))
        if additions or removals:
            raise RuntimeError(
                f"repository/index drift: {len(additions)} additions and "
                f"{len(removals)} removals; additions={additions[:10]!r}, "
                f"removals={removals[:10]!r}"
            )
        for index, model_id in enumerate(sorted(languages), 1):
            directory = model_root / model_id
            checkpoint_path = directory / "G_100000.pth"
            config_bytes = (directory / "config.json").read_bytes()
            vocab_bytes = (directory / "vocab.txt").read_bytes()
            entries.append(
                source_entry(
                    model_id,
                    languages[model_id],
                    lfs_pointer_metadata(
                        checkpoint_path.read_bytes(), checkpoint_path
                    ),
                    config_bytes,
                    vocab_bytes,
                )
            )
            if index <= 3 or index % 100 == 0 or index == len(languages):
                print(
                    f"fairseq-source: indexed {index}/{len(languages)} {model_id}",
                    file=sys.stderr,
                )
                write_progress(args.out, revision, entries, len(languages))
        atomic_json(args.out, source_document(revision, entries))
        print(f"fairseq-source: complete {args.out}", file=sys.stderr)
        return 0

    model_files = metadata_by_model(args.repository, revision)
    missing = sorted(set(languages) - set(model_files))
    if missing:
        print(
            f"fairseq-source: loading metadata for {len(missing)} entries beyond the API summary limit",
            file=sys.stderr,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for index, (model_id, files) in enumerate(
                pool.map(
                    lambda model_id: missing_model_metadata(
                        args.repository, revision, model_id
                    ),
                    missing,
                ),
                1,
            ):
                model_files[model_id] = files
                if index <= 3 or index % 100 == 0 or index == len(missing):
                    print(
                        f"fairseq-source: metadata {index}/{len(missing)} {model_id}",
                        file=sys.stderr,
                    )

    absent = sorted(set(languages) - set(model_files))
    removed = sorted(set(model_files) - set(languages))
    if absent or removed:
        raise RuntimeError(
            f"repository/index drift: {len(absent)} additions and "
            f"{len(removed)} removals; additions={absent[:10]!r}, "
            f"removals={removed[:10]!r}"
        )

    print(
        f"fairseq-source: hashing config/vocab for {len(languages)} models",
        file=sys.stderr,
    )
    entries = []
    model_ids = sorted(languages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (model_id, config, vocab) in enumerate(
            pool.map(
                lambda model_id: fetch_small_model_files(
                    args.repository, revision, model_id
                ),
                model_ids,
            ),
            1,
        ):
            files = model_files[model_id]
            for required in ("G_100000.pth", "config.json", "vocab.txt"):
                if required not in files:
                    raise RuntimeError(f"{model_id} is missing {required}")
            entries.append(
                source_entry(
                    model_id,
                    languages[model_id],
                    file_metadata(files["G_100000.pth"]),
                    config,
                    vocab,
                )
            )
            if index <= 3 or index % 100 == 0 or index == len(model_ids):
                print(
                    f"fairseq-source: hashed {index}/{len(model_ids)} {model_id}",
                    file=sys.stderr,
                )
                write_progress(args.out, revision, entries, len(model_ids))

    result = source_document(revision, entries)
    print(
        f"fairseq-source: writing {len(entries)} entries to {args.out}",
        file=sys.stderr,
    )
    atomic_json(args.out, result)
    print(f"fairseq-source: complete {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fairseq-source: error: {error}", file=sys.stderr)
        raise SystemExit(1)
