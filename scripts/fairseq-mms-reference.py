#!/usr/bin/env python3
"""Pinned upstream Fairseq MMS VITS token/waveform reference probe.

This script is only for conformance evidence. Tongues runtime inference does
not import Python, Torch, Fairseq, or Coqui.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--text", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def intersperse(values: list[int], item: int) -> list[int]:
    output = [item] * (len(values) * 2 + 1)
    output[1::2] = values
    return output


def main() -> int:
    args = parse_args()
    vits_source = args.reference_source / "vits"
    sys.path.insert(0, str(vits_source))

    import numpy as np
    import torch

    import utils
    from models import SynthesizerTrn

    config = utils.get_hparams_from_file(str(args.model_dir / "config.json"))
    symbols = (args.model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
    if config.data.training_files.rsplit(".", 1)[-1].lower() == "uroman":
        raise RuntimeError(
            "reference probe requires already-Uromanized --text for this model"
        )
    normalized = args.text.strip().lower()
    if args.language.split("-script_", 1)[0] == "ron":
        normalized = normalized.replace("ț", "ţ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    filtered = "".join(symbol for symbol in normalized if symbol in symbol_to_id)
    token_ids = [symbol_to_id[symbol] for symbol in filtered]
    if config.data.add_blank:
        token_ids = intersperse(token_ids, 0)

    torch.manual_seed(args.seed)
    model = SynthesizerTrn(
        len(symbols),
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
    )
    model.eval()
    utils.load_checkpoint(str(args.model_dir / "G_100000.pth"), model, None)
    tokens = torch.LongTensor(token_ids).unsqueeze(0)
    lengths = torch.LongTensor([len(token_ids)])
    with torch.no_grad():
        waveform = (
            model.infer(
                tokens,
                lengths,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )[0][0, 0]
            .cpu()
            .float()
            .numpy()
        )
    waveform_bytes = b"".join(struct.pack("<f", float(sample)) for sample in waveform)
    result = {
        "schema": "tongues-fairseq-mms-reference-v1",
        "reference_runtime": {
            "source_revision": "65d863f41196654aa8b8f3dc586474a4c8f30934",
            "torch": torch.__version__,
        },
        "language": args.language,
        "text": args.text,
        "normalized_text": normalized,
        "filtered_text": filtered,
        "token_ids": token_ids,
        "seed": args.seed,
        "sample_rate_hz": int(config.data.sampling_rate),
        "sample_count": int(waveform.size),
        "minimum": float(np.min(waveform)),
        "maximum": float(np.max(waveform)),
        "rms": float(np.sqrt(np.mean(np.square(waveform)))),
        "pcm_f32le_sha256": hashlib.sha256(waveform_bytes).hexdigest(),
        "first_samples": [float(sample) for sample in waveform[:64]],
        "last_samples": [float(sample) for sample in waveform[-64:]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part = args.output.with_suffix(args.output.suffix + ".part")
    part.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    part.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
