#!/usr/bin/env python3
"""Regenerate the small MPL-2.0 Align-TTS checkpoint/parity fixture."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.models.align_tts import AlignTTS, AlignTTSArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer


TEXT = "Morning light rested on the cedar trees while the kettle began to sing."
SOURCE_REVISION = "0cf3265a4686d7e856bd472cdaf1572d61cab2b8"
SEED = 210021


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    model_args = AlignTTSArgs(
        num_chars=None,
        out_channels=80,
        hidden_channels=8,
        hidden_channels_dp=8,
        encoder_type="fftransformer",
        encoder_params={
            "hidden_channels_ffn": 16,
            "num_heads": 2,
            "num_layers": 1,
            "dropout_p": 0.1,
        },
        decoder_type="fftransformer",
        decoder_params={
            "hidden_channels_ffn": 16,
            "num_heads": 2,
            "num_layers": 1,
            "dropout_p": 0.1,
        },
        length_scale=1.0,
    )
    config = AlignTTSConfig(
        model_args=model_args,
        use_phonemes=True,
        phoneme_language="en-us",
        add_blank=False,
        enable_eos_bos_chars=False,
        phase_start_steps=[10, 20, 30, 40],
    )
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = AlignTTS(config, tokenizer=tokenizer)
    model.eval()

    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        temporary = Path(temporary)
        checkpoint = temporary / "model_file.pth"
        torch.save({"model": model.state_dict(), "step": 0}, checkpoint)

        root = config.to_dict()
        root["audio"].update(
            {
                "num_mels": 80,
                "sample_rate": 22050,
                "fft_size": 1024,
                "win_length": 1024,
                "hop_length": 256,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
                "signal_norm": False,
                "log_func": "np.log",
                "spec_gain": 1.0,
                "do_amp_to_db_mel": True,
                "stft_pad_mode": "reflect",
            }
        )
        config_path = temporary / "config.json"
        config_path.write_text(
            json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        token_ids = tokenizer.text_to_ids(TEXT)
        tokens = torch.tensor([token_ids], dtype=torch.long)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long)
        with torch.no_grad():
            encoded, encoded_dp, mask, speaker = model._forward_encoder(tokens, lengths)
            duration_log = model.duration_predictor(encoded_dp, mask)
            durations = model.format_durations(duration_log, mask).squeeze(1)
            frame_lengths = durations.sum(1)
            mel_channels, alignment = model._forward_decoder(
                encoded,
                encoded_dp,
                durations,
                mask,
                frame_lengths,
                speaker,
            )
            mel = mel_channels.transpose(1, 2)

        flat_mel = mel.flatten()
        last = flat_mel.numel() - 1
        probe_indexes = [0, 79, 80, flat_mel.numel() // 2, last - 79, last]
        reference = {
            "schema": "tongues-align-tts-conformance-v1",
            "license": "MPL-2.0",
            "source_revision": SOURCE_REVISION,
            "seed": SEED,
            "text": TEXT,
            "checkpoint_symbols": tokenizer.ids_to_text(token_ids),
            "token_ids": token_ids,
            "duration_log": duration_log[0, 0].tolist(),
            "durations": durations[0].tolist(),
            "alignment_shape": list(alignment.shape),
            "mel_shape": list(mel.shape),
            "mel_probes": [
                [index, float(flat_mel[index])] for index in probe_indexes
            ],
            "checkpoint_sha256": sha256(checkpoint),
        }
        reference_path = temporary / "reference.json"
        reference_path.write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        license_path = temporary / "LICENSE.txt"
        upstream_license = Path("/opt/coqui-tts/LICENSE.txt").read_text(encoding="utf-8")
        license_path.write_text(
            "\n".join(line.rstrip() for line in upstream_license.splitlines()) + "\n",
            encoding="utf-8",
        )

        for source in [config_path, checkpoint, reference_path, license_path]:
            os.replace(source, args.out / source.name)


if __name__ == "__main__":
    main()
