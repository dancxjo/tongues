#!/usr/bin/env python3
"""Generate deterministic Coqui 0.6.1 speech conformance evidence.

This runs only inside the pinned container built by speech-conformance.sh.
The large model files are mounted read-only and are never copied into the
repository.
"""

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import torch

from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.tts.utils.helpers import generate_path, sequence_mask
from TTS.vocoder.models import setup_generator


TEXT = "Morning light rested on the cedar trees while the kettle began to sing."
COQUI_TTS_REVISION = "0cf3265a4686d7e856bd472cdaf1572d61cab2b8"
DESCRIPT_MELGAN_REVISION = "6488045bfba1975602288de07a58570c7b4d66ea"
EXPECTED_SHA256 = {
    "ljspeech/melgan/linda_johnson.pt": "d9f8a9934a162128a276b49122733a315eb434261e9c06162e11e0c5fa7a59e1",
    "ljspeech/speedy-speech/config.json": "40c571c8561ab20bb92f5c3b86a6dbe78812c8a0453aea0166d111132ea4ca02",
    "ljspeech/speedy-speech/model_file.pth": "9088f3352731e93e3ef2436f2fd4f8b116e3a7cfbd69f96140cd2da127f84ae1",
    "ljspeech/fast-pitch/config.json": "857510cdf1d33aa3b622d5f1178794cbc3842891917ebcac2d6660c3e91410d8",
    "ljspeech/fast-pitch/model_file.pth": "1779ef4ef9f9f3c016efee5925c0742393eb7c7183f6daae1928b88cbef294b8",
    "ljspeech/hifigan-v2/config.json": "12450ab044715d37dad3f472627862aed507d8bacc9d347c90a8388841ff8615",
    "ljspeech/hifigan-v2/model_file.pth": "4047e93886faa1aba11948efa71f59dcb0ec9117e286660e59b91892ef98d129",
    "ljspeech/multiband-melgan/config.json": "d4c0301bf658fc1dafdd2559dd10b13bd5a083a47e041d7917cc4c287332cd24",
    "ljspeech/multiband-melgan/model_file.pth": "56f16cee42bef70a2d75b08f9b9ea952c9ee0ccf76dd88a91d51e3ca4c11b449",
    "ljspeech/multiband-melgan/scale_stats.npy": "8c4a45b935563157509ddbff09f59e4ffea35e1d07f3bbf87ec21484cb275c4a",
    "vctk/vits/config.json": "b0ec9a22153002cb5fdadb270f6c1363460c720560c13a356d018f24a7f6cca6",
    "vctk/vits/model_file.pth": "cbec6b420abcc677fe4a357994ee68f8f3b6fa84502e7accad42b11a79f6ad0d",
    "vctk/vits/speaker_ids.json": "da4a2ecf091625a5e061e8e87c5e6032cc26f40ea7fa1981085a830a924d2887",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Directory containing ljspeech/ and vctk/ model directories",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fastpitch-only",
        action="store_true",
        help="Generate only FastPitch evidence (useful while developing the import)",
    )
    parser.add_argument(
        "--melgan-only",
        action="store_true",
        help="Generate only Descript MelGAN evidence",
    )
    parser.add_argument(
        "--multiband-melgan-only",
        action="store_true",
        help="Generate only MultiBand-MelGAN evidence",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_artifacts(model_root: Path, required=None) -> dict[str, Path]:
    artifacts = {}
    for relative, expected in EXPECTED_SHA256.items():
        if required is not None and relative not in required:
            continue
        path = model_root / relative
        if not path.is_file():
            raise SystemExit(f"required Coqui artifact is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"Coqui artifact checksum mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
        artifacts[relative] = path
    return artifacts


def fast_pitch_reference(artifacts: dict[str, Path]) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        config = load_config(str(artifacts["ljspeech/fast-pitch/config.json"]))
        acoustic = setup_model(config)
        acoustic.load_checkpoint(
            config,
            str(artifacts["ljspeech/fast-pitch/model_file.pth"]),
            eval=True,
        )

    token_ids = acoustic.tokenizer.text_to_ids(TEXT)
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    token_mask = torch.ones((1, 1, len(token_ids)), dtype=torch.float32)
    embedded = acoustic.emb(token_tensor).transpose(1, -1)
    encoded = acoustic.encoder(embedded, token_mask)
    duration_log = acoustic.duration_predictor(encoded, token_mask)
    durations = acoustic.format_durations(duration_log, token_mask).squeeze(1)
    pitch = acoustic.pitch_predictor(encoded, token_mask)
    pitch_conditioned = encoded + acoustic.pitch_emb(pitch)
    output_lengths = durations.sum(1)
    output_mask = sequence_mask(output_lengths, None).unsqueeze(1).to(encoded.dtype)
    expanded, _ = acoustic.expand_encoder_outputs(
        pitch_conditioned, durations, token_mask, output_mask
    )
    positioned = acoustic.pos_encoder(expanded, output_mask)
    mel = acoustic.decoder(positioned, output_mask).transpose(1, 2)[0]

    if not all(
        torch.isfinite(value).all()
        for value in [encoded, duration_log, durations, pitch, positioned, mel]
    ):
        raise SystemExit("Coqui FastPitch reference produced non-finite output")

    token_count = len(token_ids)
    frame_count = mel.shape[0]
    return {
        "text": TEXT,
        "checkpoint_symbols": acoustic.tokenizer.ids_to_text(token_ids),
        "token_ids": token_ids,
        "durations": [float(value) for value in durations[0]],
        "pitch": [float(value) for value in pitch[0, 0]],
        "stages": {
            "encoded": probes(
                encoded[0],
                [(0, 0), (7, 1), (64, token_count // 2), (383, token_count - 1)],
            ),
            "duration_log": probes(
                duration_log[0],
                [(0, 0), (0, 1), (0, token_count // 2), (0, token_count - 1)],
            ),
            "pitch": probes(
                pitch[0],
                [(0, 0), (0, 1), (0, token_count // 2), (0, token_count - 1)],
            ),
            "positioned": probes(
                positioned[0],
                [(0, 0), (7, 1), (64, frame_count // 2), (383, frame_count - 1)],
            ),
            "mel": probes(
                mel,
                [
                    (0, 0),
                    (0, 79),
                    (1, 7),
                    (5, 23),
                    (20, 40),
                    (frame_count // 2, 40),
                    (frame_count - 1, 79),
                ],
            ),
        },
        "mel_shape": list(mel.shape),
    }


def probes(tensor: torch.Tensor, positions: list[tuple[int, int]]) -> list[list[float]]:
    return [[first, second, float(tensor[first, second])] for first, second in positions]


def speedy_speech_reference(artifacts: dict[str, Path]) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        tts_config = load_config(
            str(artifacts["ljspeech/speedy-speech/config.json"])
        )
        acoustic = setup_model(tts_config)
        acoustic.load_checkpoint(
            tts_config,
            str(artifacts["ljspeech/speedy-speech/model_file.pth"]),
            eval=True,
        )

    token_ids = acoustic.tokenizer.text_to_ids(TEXT)
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    token_mask = torch.ones((1, 1, len(token_ids)), dtype=torch.float32)
    embedded = acoustic.emb(token_tensor).transpose(1, -1)
    encoded = acoustic.encoder(embedded, token_mask)
    duration_log = acoustic.duration_predictor(encoded, token_mask)
    durations = acoustic.format_durations(duration_log, token_mask).squeeze(1)
    output_lengths = durations.sum(1)
    output_mask = sequence_mask(output_lengths, None).unsqueeze(1).to(encoded.dtype)
    expanded, _ = acoustic.expand_encoder_outputs(
        encoded, durations, token_mask, output_mask
    )
    positioned = acoustic.pos_encoder(expanded, output_mask)
    mel = acoustic.decoder(positioned, output_mask).transpose(1, 2)[0]

    with contextlib.redirect_stdout(sys.stderr):
        vocoder_config = load_config(
            str(artifacts["ljspeech/hifigan-v2/config.json"])
        )
        vocoder = setup_generator(vocoder_config)
        vocoder.load_checkpoint(
            vocoder_config,
            str(artifacts["ljspeech/hifigan-v2/model_file.pth"]),
            eval=True,
        )
    waveform = vocoder.inference(mel.transpose(0, 1).unsqueeze(0)).flatten()

    if not torch.isfinite(mel).all() or not torch.isfinite(waveform).all():
        raise SystemExit("Coqui reference produced non-finite output")

    return {
        "text": TEXT,
        "checkpoint_symbols": acoustic.tokenizer.ids_to_text(token_ids),
        "token_ids": token_ids,
        "durations": [float(value) for value in durations[0]],
        "stages": {
            "encoded": probes(
                encoded[0], [(0, 0), (7, 1), (64, 20), (127, 61)]
            ),
            "expanded": probes(
                expanded[0], [(0, 0), (7, 1), (64, 168), (127, 338)]
            ),
            "positioned": probes(
                positioned[0], [(0, 0), (7, 1), (64, 168), (127, 338)]
            ),
            "mel": probes(
                mel,
                [
                    (0, 0),
                    (0, 79),
                    (1, 7),
                    (5, 23),
                    (20, 40),
                    (50, 79),
                    (100, 23),
                    (168, 40),
                    (250, 7),
                    (338, 79),
                ],
            ),
        },
        "mel_shape": list(mel.shape),
        "waveform": {
            "sample_rate_hz": 22050,
            "channels": 1,
            "samples": int(waveform.numel()),
            "duration_seconds": waveform.numel() / 22050,
            "minimum": float(waveform.min()),
            "maximum": float(waveform.max()),
            "rms": float(waveform.square().mean().sqrt()),
            "non_finite_samples": int((~torch.isfinite(waveform)).sum()),
            "probes": [
                [index, float(waveform[index])]
                for index in [
                    0,
                    1,
                    255,
                    256,
                    1000,
                    waveform.numel() // 2,
                    waveform.numel() - 2,
                    waveform.numel() - 1,
                ]
            ],
        },
    }


def multiband_melgan_reference(artifacts: dict[str, Path]) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        config = load_config(
            str(artifacts["ljspeech/multiband-melgan/config.json"])
        )
        vocoder = setup_generator(config)
        vocoder.load_checkpoint(
            config,
            str(artifacts["ljspeech/multiband-melgan/model_file.pth"]),
            eval=True,
        )

    frames = 8
    mel = torch.linspace(-1.0, 1.0, 80 * frames, dtype=torch.float32).reshape(
        1, 80, frames
    )
    waveform = vocoder.inference(mel).flatten()
    if not torch.isfinite(waveform).all():
        raise SystemExit("Coqui MultiBand-MelGAN reference produced non-finite output")

    return {
        "input_shape": list(mel.shape),
        "input_pattern": "channel-major-linspace-negative-one-to-one",
        "waveform": {
            "sample_rate_hz": int(config.audio.sample_rate),
            "channels": 1,
            "samples": int(waveform.numel()),
            "minimum": float(waveform.min()),
            "maximum": float(waveform.max()),
            "rms": float(waveform.square().mean().sqrt()),
            "non_finite_samples": int((~torch.isfinite(waveform)).sum()),
            "probes": [
                [index, float(waveform[index])]
                for index in [
                    0,
                    1,
                    255,
                    256,
                    1000,
                    waveform.numel() // 2,
                    waveform.numel() - 2,
                    waveform.numel() - 1,
                ]
            ],
        },
    }


def melgan_reference(artifacts: dict[str, Path]) -> dict:
    from mel2wav.modules import Generator

    generator = Generator(80, 32, 3)
    generator.load_state_dict(
        torch.load(
            artifacts["ljspeech/melgan/linda_johnson.pt"],
            map_location=torch.device("cpu"),
        )
    )
    generator.eval()
    frames = 8
    mel = torch.linspace(-1.0, 1.0, 80 * frames, dtype=torch.float32).reshape(
        1, 80, frames
    )
    with torch.no_grad():
        waveform = generator(mel).flatten()
    if not torch.isfinite(waveform).all():
        raise SystemExit("Descript MelGAN reference produced non-finite output")

    return {
        "input_shape": list(mel.shape),
        "input_pattern": "channel-major-linspace-negative-one-to-one",
        "waveform": {
            "sample_rate_hz": 22050,
            "channels": 1,
            "samples": int(waveform.numel()),
            "minimum": float(waveform.min()),
            "maximum": float(waveform.max()),
            "rms": float(waveform.square().mean().sqrt()),
            "non_finite_samples": int((~torch.isfinite(waveform)).sum()),
            "probes": [
                [index, float(waveform[index])]
                for index in [
                    0,
                    1,
                    255,
                    256,
                    1000,
                    waveform.numel() // 2,
                    waveform.numel() - 2,
                    waveform.numel() - 1,
                ]
            ],
        },
    }


def vits_speaker_reference(model, token_tensor: torch.Tensor, speaker: str) -> dict:
    speaker_id = model.speaker_manager.speaker_ids[speaker]
    speaker_tensor = torch.tensor([speaker_id], dtype=torch.long)
    token_lengths = torch.tensor([token_tensor.shape[1]], dtype=torch.long)
    conditioning = model.emb_g(speaker_tensor).unsqueeze(-1)
    encoded, prior_mean, prior_log_scale, token_mask = model.text_encoder(
        token_tensor, token_lengths
    )
    log_durations = model.duration_predictor(
        encoded,
        token_mask,
        g=conditioning if model.args.condition_dp_on_speaker else None,
        reverse=True,
        noise_scale=0.0,
    )
    durations = torch.ceil(torch.exp(log_durations) * token_mask * model.length_scale)
    output_lengths = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
    output_mask = sequence_mask(output_lengths, None).to(token_mask.dtype).unsqueeze(1)
    attention_mask = token_mask * output_mask.transpose(1, 2)
    attention = generate_path(
        durations.squeeze(1), attention_mask.squeeze(1).transpose(1, 2)
    )
    expanded_mean = torch.matmul(
        attention.transpose(1, 2), prior_mean.transpose(1, 2)
    ).transpose(1, 2)
    expanded_log_scale = torch.matmul(
        attention.transpose(1, 2), prior_log_scale.transpose(1, 2)
    ).transpose(1, 2)
    latent_prior = expanded_mean
    latent = model.flow(
        latent_prior, output_mask, g=conditioning, reverse=True
    ) * output_mask
    waveform = model.waveform_decoder(latent, g=conditioning).flatten()

    if not all(
        torch.isfinite(value).all()
        for value in [
            encoded,
            prior_mean,
            prior_log_scale,
            log_durations,
            latent,
            waveform,
        ]
    ):
        raise SystemExit(f"Coqui VITS reference produced non-finite output for {speaker}")

    token_count = token_tensor.shape[1]
    output_frames = int(output_lengths[0])
    return {
        "speaker": speaker,
        "speaker_id": speaker_id,
        "durations": [float(value) for value in durations.flatten()],
        "output_frames": output_frames,
        "stages": {
            "speaker_embedding": probes(
                conditioning[0], [(0, 0), (17, 0), (127, 0), (255, 0)]
            ),
            "encoded": probes(
                encoded[0],
                [(0, 0), (7, 1), (64, token_count // 2), (191, token_count - 1)],
            ),
            "prior_mean": probes(
                prior_mean[0],
                [(0, 0), (7, 1), (64, token_count // 2), (191, token_count - 1)],
            ),
            "log_durations": probes(
                log_durations[0],
                [(0, 0), (0, 1), (0, token_count // 2), (0, token_count - 1)],
            ),
            "expanded_mean": probes(
                expanded_mean[0],
                [(0, 0), (7, 1), (64, output_frames // 2), (191, output_frames - 1)],
            ),
            "latent": probes(
                latent[0],
                [(0, 0), (7, 1), (64, output_frames // 2), (191, output_frames - 1)],
            ),
        },
        "waveform": {
            "sample_rate_hz": 22050,
            "channels": 1,
            "samples": int(waveform.numel()),
            "duration_seconds": waveform.numel() / 22050,
            "minimum": float(waveform.min()),
            "maximum": float(waveform.max()),
            "rms": float(waveform.square().mean().sqrt()),
            "non_finite_samples": int((~torch.isfinite(waveform)).sum()),
            "probes": [
                [index, float(waveform[index])]
                for index in [
                    0,
                    1,
                    255,
                    256,
                    1000,
                    waveform.numel() // 2,
                    waveform.numel() - 2,
                    waveform.numel() - 1,
                ]
            ],
        },
    }


def vits_reference(artifacts: dict[str, Path]) -> dict:
    with contextlib.redirect_stdout(sys.stderr):
        config = load_config(str(artifacts["vctk/vits/config.json"]))
        speaker_map = str(artifacts["vctk/vits/speaker_ids.json"])
        config.model_args.speakers_file = speaker_map
        config.speakers_file = speaker_map
        model = setup_model(config)
        model.load_checkpoint(
            config, str(artifacts["vctk/vits/model_file.pth"]), eval=True
        )

    token_ids = model.tokenizer.text_to_ids(TEXT)
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    return {
        "text": TEXT,
        "checkpoint_symbols": model.tokenizer.ids_to_text(token_ids),
        "token_ids": token_ids,
        "noise_scale": 0.0,
        "duration_noise_scale": 0.0,
        "length_scale": float(model.length_scale),
        "speakers": [
            vits_speaker_reference(model, token_tensor, speaker)
            for speaker in ["p225", "p330", "p376"]
        ],
    }


def main() -> None:
    args = parse_args()
    if sum(
        [
            args.fastpitch_only,
            args.melgan_only,
            args.multiband_melgan_only,
        ]
    ) > 1:
        raise SystemExit("only one model-specific evidence mode may be selected")
    if args.fastpitch_only:
        required = {
            "ljspeech/fast-pitch/config.json",
            "ljspeech/fast-pitch/model_file.pth",
        }
        artifacts = require_artifacts(args.model_root, required)
        evidence = {
            "schema": "tongues-fastpitch-conformance-v1",
            "reference_runtime": {
                "name": "Coqui TTS",
                "revision": COQUI_TTS_REVISION,
                "torch": torch.__version__,
            },
            "artifacts_sha256": {
                key: EXPECTED_SHA256[key] for key in sorted(required)
            },
            "fast_pitch": fast_pitch_reference(artifacts),
        }
        serialized = json.dumps(
            evidence, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
        return

    if args.melgan_only:
        required = {"ljspeech/melgan/linda_johnson.pt"}
        artifacts = require_artifacts(args.model_root, required)
        evidence = {
            "schema": "tongues-melgan-conformance-v1",
            "reference_runtime": {
                "name": "descriptinc/melgan-neurips",
                "revision": DESCRIPT_MELGAN_REVISION,
                "torch": torch.__version__,
            },
            "artifacts_sha256": {
                key: EXPECTED_SHA256[key] for key in sorted(required)
            },
            "melgan": melgan_reference(artifacts),
        }
        serialized = json.dumps(
            evidence, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
        return

    if args.multiband_melgan_only:
        required = {
            "ljspeech/multiband-melgan/config.json",
            "ljspeech/multiband-melgan/model_file.pth",
            "ljspeech/multiband-melgan/scale_stats.npy",
        }
        artifacts = require_artifacts(args.model_root, required)
        evidence = {
            "schema": "tongues-multiband-melgan-conformance-v1",
            "reference_runtime": {
                "name": "Coqui TTS",
                "revision": COQUI_TTS_REVISION,
                "torch": torch.__version__,
            },
            "artifacts_sha256": {
                key: EXPECTED_SHA256[key] for key in sorted(required)
            },
            "multiband_melgan": multiband_melgan_reference(artifacts),
        }
        serialized = json.dumps(
            evidence, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
        return

    artifacts = require_artifacts(args.model_root)
    torch.manual_seed(27)
    evidence = {
        "schema": "tongues-speech-conformance-v1",
        "reference_runtime": {
            "name": "Coqui TTS",
            "revision": COQUI_TTS_REVISION,
            "torch": torch.__version__,
        },
        "artifacts_sha256": EXPECTED_SHA256,
        "fast_pitch": fast_pitch_reference(artifacts),
        "melgan": melgan_reference(artifacts),
        "multiband_melgan": multiband_melgan_reference(artifacts),
        "speedy_speech_hifigan": speedy_speech_reference(artifacts),
        "vits": vits_reference(artifacts),
    }
    serialized = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
