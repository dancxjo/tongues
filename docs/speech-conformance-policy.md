# Speech Conformance Policy

Artifact-backed conformance testing runs separately from the fast PR CI lane.
Every pinned external model is verified through the licensed catalog and
downloader before tests execute.  Missing or legally unavailable artifacts
produce an explicit SKIPPED entry rather than silent success.

## Running Locally

```sh
scripts/speech-conformance.sh
```

The script checks every artifact family that is locally available, runs its
tests, and writes a durable JSON summary to
`target/speech-conformance/conformance-summary.json`.  A human-readable report
table is printed at the end.

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `TONGUES_COQUI_MODEL_ROOT` | `$XDG_DATA_HOME/mortar-sea/models/speech/coqui/en` | Root of the Coqui LJSpeech/VCTK model tree |
| `TONGUES_YOURTTS_MODEL_ROOT` | `$XDG_DATA_HOME/mortar-sea/models/speech/coqui/multilingual/yourtts` | YourTTS model directory (CC-BY-NC-ND 4.0) |
| `TONGUES_FAIRSEQ_MODEL_ROOT` | *(unset)* | Fairseq MMS model directory (CC-BY-NC-4.0) |
| `TONGUES_FREEVC_MODEL_ROOT` | *(unset)* | FreeVC24 model directory (MIT, auxiliary artifacts required) |
| `TONGUES_FREEVC_SOURCE_WAV` | *(unset)* | Source WAV for FreeVC voice-conversion test |
| `TONGUES_FREEVC_TARGET_WAV` | *(unset)* | Target WAV for FreeVC voice-conversion test |
| `TONGUES_FREEVC_SPEAKER_WAV` | *(unset)* | Same-speaker WAV for the speaker-encoder test |
| `TONGUES_FREEVC_ALT_WAV` | *(unset)* | Different-speaker WAV for the speaker-encoder test |
| `TONGUES_CONFORMANCE_OUTPUT` | `target/speech-conformance` | Output directory for evidence and summary |
| `TONGUES_COQUI_REFERENCE_IMAGE` | `tongues-coqui-reference` | Docker image tag for the pinned Coqui runtime |

Install Coqui model families through the verified catalog before running:

```sh
cargo run --bin tongues -- models install \
  tts_models/en/ljspeech/speedy-speech-hifigan \
  tts_models/en/ljspeech/fast_pitch-HifiGAN \
  tts_models/en/ljspeech/glow-tts \
  tts_models/en/ljspeech/multiband-melgan \
  tts_models/en/vctk/vits
```

Alternatively use the `Justfile` target:

```sh
just speech-conformance
```

## Scheduled CI

The `.github/workflows/speech-conformance.yml` workflow runs automatically on
a weekly schedule and can be triggered manually via `workflow_dispatch`.  The
summary is published both as a GitHub Actions artifact (retained 90 days) and
as a job summary so drift is visible without downloading the artifact.

## Artifact Families

### Required for Release

These families must pass before a release is tagged.  A missing artifact is
reported as SKIPPED and must be resolved before the release proceeds.

| Family | Models | License |
|---|---|---|
| `align-tts` | Committed `align-tts-mpl-fixture` (tiny synthetic checkpoint) + Docker runtime | MPL-2.0 (Coqui TTS source) |
| `coqui-speedy-speech` | LJSpeech SpeedySpeech + HiFi-GAN v2 | NOASSERTION (no upstream license evidence; do not redistribute) |
| `coqui-vits` | VCTK VITS (speakers p225, p330, p376) | NOASSERTION (no upstream license evidence; do not redistribute) |
| `coqui-fastpitch` | LJSpeech FastPitch + HiFi-GAN v2 | NOASSERTION (no upstream license evidence; do not redistribute) |
| `glow-tts` | LJSpeech Glow-TTS + MultiBand-MelGAN | MPL (Coqui registry label) |
| `multiband-melgan` | LJSpeech MultiBand-MelGAN + PQMF | MPL (Coqui registry label) |
| `melgan` | Descript `linda_johnson.pt` MelGAN | MIT (Descript/melgan-neurips repo) |

A family is required if its implementation ticket is closed and the backend is
exposed in the published CLI.  Closed issues #4, #7, #8, #13, #20, #26, #38,
and #59 established these backends.

### Informational (not required for release)

These families produce useful evidence but are not required to pass before
tagging a release due to artifact licensing restrictions, hardware
requirements, or optional infrastructure.

| Family | Models | Classification reason |
|---|---|---|
| `yourtts` | Multilingual YourTTS + speaker encoder | **CC-BY-NC-ND 4.0** — restricts commercial use and redistribution of weights |
| `fairseq-mms` | Fairseq MMS VITS (1,143 languages) | **CC-BY-NC-4.0** — restricts commercial use; install via `models install fairseq-mms-vits-eng` |
| `freevc` | FreeVC24 voice conversion + speaker encoder | MIT weights, but all three auxiliary artifacts must be installed separately |
| `onnx` | ONNX Piper voices | Smoke only; specific voice availability varies by installation |

Informational families that fail (not skip) are surfaced in the summary and
should be investigated.  They do not block a release tag.

## Summary Format

The JSON summary written to `target/speech-conformance/conformance-summary.json`
follows the `tongues-speech-conformance-summary-v1` schema:

```jsonc
{
  "schema": "tongues-speech-conformance-summary-v1",
  "git_revision": "<SHA>",
  "timestamp": "<ISO8601>",
  "totals": {
    "families_run": true,
    "tests_run": 14,
    "tests_passed": 12,
    "tests_failed": 0,
    "families_skipped": 2
  },
  "families": {
    "align-tts": {
      "status": "passed",          // "passed" | "failed" | "skipped"
      "release_gate": "required",  // "required" | "informational"
      "tests_run": 2,
      "tests_passed": 2,
      "tests_failed": 0,
      "skip_reason": null
    },
    "yourtts": {
      "status": "skipped",
      "release_gate": "informational",
      "tests_run": 0,
      "tests_passed": 0,
      "tests_failed": 0,
      "skip_reason": "YourTTS artifacts not found under ... (CC-BY-NC-ND 4.0 — install separately)"
    }
    // ... one entry per family
  }
}
```

## Annotated Test Inventory

Every artifact-backed test carries an `#[ignore]` annotation that names the
conformance lane that owns it.  The annotation format is:

```
#[ignore = "requires <artifact set>; run scripts/speech-conformance.sh"]
```

This ensures the default `cargo test` run never silently passes without the
artifacts, and the lane reference tells contributors where to run the test.

| Test | File | Family |
|---|---|---|
| `published_fairseq_mms_checkpoint_loads_and_synthesizes_without_python` | `burn_vits.rs` | `fairseq-mms` |
| `published_checkpoint_stage_parity` (VITS) | `burn_vits.rs` | `coqui-vits` |
| `published_your_tts_checkpoints_load_when_available` | `burn_vits.rs` | `yourtts` |
| `published_your_tts_named_enrollment_synthesizes_when_available` | `burn_vits.rs` | `yourtts` |
| `published_yourtts_conformance` | `burn_vits.rs` | `yourtts` |
| `published_vits_stage_token_parity` | `burn_vits.rs` | `coqui-vits` |
| `published_checkpoint_stage_parity` (SpeedySpeech) | `burn_speedy_speech.rs` | `coqui-speedy-speech` |
| `published_checkpoint_stage_parity` (FastPitch) | `burn_fast_pitch.rs` | `coqui-fastpitch` |
| `published_glow_checkpoint_synthesizes` | `burn_glow_tts.rs` | `glow-tts` |
| `published_glow_checkpoint_stage_parity` | `burn_glow_tts.rs` | `glow-tts` |
| `external_sc_glow_checkpoint_synthesizes` | `burn_glow_tts.rs` | user-supplied (license evidence required) |
| `published_acoustic_backend_emits_neutral_mel` | `burn_glow_tts_acoustic.rs` | `glow-tts` |
| `published_acoustic_backend_covers_input_matrix` | `burn_glow_tts_acoustic.rs` | `glow-tts` |
| `published_multiband_melgan_checkpoint_parity` | `burn_vocoder.rs` | `multiband-melgan` |
| `published_melgan_checkpoint_parity` | `burn_vocoder.rs` | `melgan` |
| `published_resnet_encodes_reference_audio_when_available` | `speaker_encoder.rs` | `yourtts` |
| `published_speaker_encoder_loads_and_separates_fixtures` | `freevc.rs` | `freevc` |
| `published_artifacts_convert_without_python` | `freevc.rs` | `freevc` |
| `published_glow_tts_fixture_uses_common_importer` | `model_package.rs` | `glow-tts` |

## Release Checklist

Before tagging a release:

1. Run `scripts/speech-conformance.sh` with all required-family artifacts
   installed.
2. Confirm `conformance-summary.json` shows `"status": "passed"` for every
   required family.
3. Review any informational family failures for unexpected regressions.
4. Attach the summary (or its `git_revision` and `timestamp`) to the release
   notes.
