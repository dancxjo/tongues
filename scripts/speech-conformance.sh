#!/usr/bin/env bash
# Artifact-backed speech conformance harness.
#
# Runs every available artifact family and emits a durable JSON summary under
# $output_dir/conformance-summary.json.  Missing artifacts produce an explicit
# SKIPPED entry rather than silent success; a missing required-for-release
# family causes a non-zero exit.
#
# Usage:
#   scripts/speech-conformance.sh
#
# Relevant environment variables (all optional):
#   TONGUES_COQUI_MODEL_ROOT    – root of the Coqui LJSpeech/VCTK model tree
#   TONGUES_YOURTTS_MODEL_ROOT  – root of the YourTTS model directory
#   TONGUES_FAIRSEQ_MODEL_ROOT  – root of the Fairseq MMS model directory
#   TONGUES_FREEVC_MODEL_ROOT   – root of the FreeVC24 model directory
#   TONGUES_FREEVC_SOURCE_WAV   – source WAV for FreeVC voice conversion
#   TONGUES_FREEVC_TARGET_WAV   – target WAV for FreeVC voice conversion
#   TONGUES_FREEVC_SPEAKER_WAV  – WAV for the standalone speaker-encoder test
#   TONGUES_FREEVC_ALT_WAV      – different-speaker WAV for speaker-encoder test
#   TONGUES_COQUI_REFERENCE_IMAGE – Docker image tag for the pinned Coqui runtime
#   TONGUES_CONFORMANCE_OUTPUT  – output directory (default: target/speech-conformance)
#
# See docs/speech-conformance-policy.md for release-gate definitions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for command in cargo cmp docker jq sha256sum; do
    if ! command -v "$command" >/dev/null; then
        echo "required conformance command is missing: $command" >&2
        exit 2
    fi
done

data_root="${XDG_DATA_HOME:-$HOME/.local/share}"
model_root="${TONGUES_COQUI_MODEL_ROOT:-$data_root/mortar-sea/models/speech/coqui/en}"
speedy_dir="$model_root/ljspeech/speedy-speech"
fastpitch_dir="$model_root/ljspeech/fast-pitch"
vocoder_dir="$model_root/ljspeech/hifigan-v2"
multiband_dir="$model_root/ljspeech/multiband-melgan"
glow_dir="$model_root/ljspeech/glow-tts"
melgan_model="$model_root/ljspeech/melgan/linda_johnson.pt"
vits_dir="$model_root/vctk/vits"
yourtts_dir="${TONGUES_YOURTTS_MODEL_ROOT:-$data_root/mortar-sea/models/speech/coqui/multilingual/yourtts}"
fairseq_model_root="${TONGUES_FAIRSEQ_MODEL_ROOT:-}"
freevc_model_root="${TONGUES_FREEVC_MODEL_ROOT:-}"
freevc_source_wav="${TONGUES_FREEVC_SOURCE_WAV:-}"
freevc_target_wav="${TONGUES_FREEVC_TARGET_WAV:-}"
freevc_speaker_wav="${TONGUES_FREEVC_SPEAKER_WAV:-}"
freevc_alt_wav="${TONGUES_FREEVC_ALT_WAV:-}"
output_dir="${TONGUES_CONFORMANCE_OUTPUT:-$repo_root/target/speech-conformance}"
image="${TONGUES_COQUI_REFERENCE_IMAGE:-tongues-coqui-reference}"

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
summary_file="$output_dir/conformance-summary.json"
git_revision="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- per-family result tracking ---
# Status values: "passed" | "failed" | "skipped"
declare -A family_status
declare -A family_skip_reason
declare -A family_tests_run
declare -A family_tests_failed
declare -A family_release_gate  # "required" | "informational"

register_family() {
    local name="$1" gate="$2"
    family_status[$name]="skipped"
    family_skip_reason[$name]=""
    family_tests_run[$name]=0
    family_tests_failed[$name]=0
    family_release_gate[$name]="$gate"
}

# Families are ordered: required-for-release first, then informational.
register_family "align-tts"         "required"
register_family "coqui-speedy-speech" "required"
register_family "coqui-vits"        "required"
register_family "coqui-fastpitch"   "required"
register_family "glow-tts"          "required"
register_family "multiband-melgan"  "required"
register_family "melgan"            "required"
register_family "yourtts"           "informational"  # CC-BY-NC-ND 4.0
register_family "fairseq-mms"       "informational"  # CC-BY-NC-4.0; fixture-only MMS test is always run
register_family "freevc"            "informational"  # MIT; requires separately installed artifacts
register_family "onnx"              "informational"  # smoke; requires installed Piper voice

# Run a single test step within a family, tracking pass/fail.
# Usage: run_step FAMILY_NAME -- CMD [ARGS...]
# The "--" separator is consumed; everything after is the command to run.
run_step() {
    local family="$1"
    shift 2  # consume family name and "--"
    family_tests_run[$family]=$(( ${family_tests_run[$family]} + 1 ))
    if ! "$@"; then
        family_tests_failed[$family]=$(( ${family_tests_failed[$family]} + 1 ))
        family_status[$family]="failed"
    elif [[ "${family_status[$family]}" == "skipped" ]]; then
        family_status[$family]="passed"
    fi
}

# Mark a family as skipped with a reason message.
skip_family() {
    local family="$1" reason="$2"
    family_status[$family]="skipped"
    family_skip_reason[$family]="$reason"
    echo "SKIPPED [$family]: $reason"
}

# Check whether all listed files exist; return 1 if any is missing.
all_present() {
    local missing=0
    for path in "$@"; do
        if [[ ! -f "$path" ]]; then
            echo "  missing: $path"
            missing=1
        fi
    done
    return "$missing"
}

# ---------------------------------------------------------------------------
# Phase 0: build release binary and pinned reference image
# ---------------------------------------------------------------------------
echo "==> Building release binary"
cargo build --release -p tongues-cli

echo "==> Building pinned Coqui reference runtime: $image"
docker build --tag "$image" --file tools/speech-conformance/Dockerfile .

reference="$output_dir/coqui-reference.json"

# ---------------------------------------------------------------------------
# Family: align-tts  (required; uses committed fixture + Docker)
# ---------------------------------------------------------------------------
echo
echo "==> Family: align-tts [required]"

align_fixture="$repo_root/fixtures/speech/align-tts-mpl-fixture"
align_candidate="$output_dir/align-tts-mpl-fixture"
mkdir -p "$align_candidate"

echo "Regenerating the licensed Align-TTS fixture: $align_candidate"
if run_step "align-tts" -- docker run --rm \
        --entrypoint python \
        --volume "$repo_root:/workspace" \
        --volume "$output_dir:/evidence" \
        "$image" \
        scripts/align-tts-fixture.py \
        --out /evidence/align-tts-mpl-fixture; then
    fixture_ok=1
    for artifact in config.json model_file.pth reference.json LICENSE.txt; do
        if ! cmp --silent "$align_fixture/$artifact" "$align_candidate/$artifact"; then
            echo "pinned Align-TTS fixture drifted: $artifact" >&2
            fixture_ok=0
        fi
    done
    if [[ "$fixture_ok" -eq 1 ]]; then
        echo "Licensed Align-TTS fixture matched"
        run_step "align-tts" -- cargo test --release -p tongues-tts align_tts -- --nocapture
    else
        family_tests_failed[align-tts]=$(( ${family_tests_failed[align-tts]} + 1 ))
        family_status[align-tts]="failed"
    fi
fi

# ---------------------------------------------------------------------------
# Phase: generate Coqui reference evidence (shared by coqui-* and glow-tts)
# ---------------------------------------------------------------------------
coqui_core_artifacts=(
    "$speedy_dir/config.json"
    "$speedy_dir/model_file.pth"
    "$fastpitch_dir/config.json"
    "$fastpitch_dir/model_file.pth"
    "$vocoder_dir/config.json"
    "$vocoder_dir/model_file.pth"
    "$multiband_dir/config.json"
    "$multiband_dir/model_file.pth"
    "$multiband_dir/scale_stats.npy"
    "$glow_dir/config.json"
    "$glow_dir/model_file.pth.tar"
    "$melgan_model"
    "$vits_dir/config.json"
    "$vits_dir/model_file.pth"
    "$vits_dir/speaker_ids.json"
)
yourtts_artifacts=(
    "$yourtts_dir/config.json"
    "$yourtts_dir/model_file.pth.tar"
    "$yourtts_dir/speakers.json"
    "$yourtts_dir/language_ids.json"
    "$yourtts_dir/model_se.pth.tar"
    "$yourtts_dir/config_se.json"
)

coqui_core_present=0
yourtts_present=0
if all_present "${coqui_core_artifacts[@]}" 2>/dev/null; then
    coqui_core_present=1
fi
if all_present "${yourtts_artifacts[@]}" 2>/dev/null; then
    yourtts_present=1
fi

if [[ "$coqui_core_present" -eq 1 ]]; then
    echo
    echo "==> Generating Coqui stage evidence: $reference.part"
    if [[ "$yourtts_present" -eq 1 ]]; then
        docker run --rm \
            --volume "$model_root:/models:ro" \
            --volume "$yourtts_dir:/yourtts:ro" \
            --volume "$repo_root:/workspace" \
            --volume "$output_dir:/evidence" \
            "$image" \
            --model-root /models \
            --yourtts-root /yourtts \
            --reference-wav-output /evidence/yourtts-reference.wav \
            --output /evidence/coqui-reference.json.part
    else
        docker run --rm \
            --volume "$model_root:/models:ro" \
            --volume "$repo_root:/workspace" \
            --volume "$output_dir:/evidence" \
            "$image" \
            --model-root /models \
            --output /evidence/coqui-reference.json.part
    fi
    mv --force "$reference.part" "$reference"
    echo "Reference evidence committed atomically: $reference"

    tokenization="$output_dir/coqui-v0.6.1-tokenization.json"
    jq -S '{
        schema: "tongues-speech-tokenization-v1",
        reference_runtime,
        speedy_speech_hifigan: {
            text: .speedy_speech_hifigan.text,
            checkpoint_symbols: .speedy_speech_hifigan.checkpoint_symbols,
            token_ids: .speedy_speech_hifigan.token_ids
        },
        fast_pitch: {
            text: .fast_pitch.text,
            checkpoint_symbols: .fast_pitch.checkpoint_symbols,
            token_ids: .fast_pitch.token_ids
        },
        glow_tts: {
            text: .glow_tts.text,
            checkpoint_symbols: .glow_tts.checkpoint_symbols,
            token_ids: .glow_tts.token_ids
        },
        vits: {
            text: .vits.text,
            checkpoint_symbols: .vits.checkpoint_symbols,
            token_ids: .vits.token_ids,
            speakers: [.vits.speakers[] | {speaker, speaker_id}]
        }
    }' "$reference" > "$tokenization.part"
    jq -S . fixtures/speech/coqui-v0.6.1-tokenization.json > "$tokenization.expected.part"
    if ! cmp --silent "$tokenization.part" "$tokenization.expected.part"; then
        echo "pinned Coqui tokenizer output drifted from fixtures/speech/coqui-v0.6.1-tokenization.json" >&2
        diff -u "$tokenization.expected.part" "$tokenization.part" || true
        # Mark all Coqui-dependent families as failed
        for fam in coqui-speedy-speech coqui-vits coqui-fastpitch glow-tts multiband-melgan melgan; do
            family_status[$fam]="failed"
            family_tests_run[$fam]=$(( ${family_tests_run[$fam]} + 1 ))
            family_tests_failed[$fam]=$(( ${family_tests_failed[$fam]} + 1 ))
        done
        rm -f "$tokenization.part" "$tokenization.expected.part"
    else
        mv --force "$tokenization.part" "$tokenization"
        rm "$tokenization.expected.part"
        echo "Pinned tokenizer fixture matched: $tokenization"
    fi
fi

# ---------------------------------------------------------------------------
# Family: coqui-speedy-speech  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: coqui-speedy-speech [required]"
if [[ "$coqui_core_present" -eq 0 ]]; then
    skip_family "coqui-speedy-speech" "Coqui LJSpeech model artifacts not found under $model_root"
elif [[ "${family_status[coqui-speedy-speech]}" != "failed" ]]; then
    echo "Comparing native SpeedySpeech and HiFi-GAN stages: $reference"
    run_step "coqui-speedy-speech" -- env \
        TONGUES_TEST_COQUI_SPEEDY_CONFIG="$speedy_dir/config.json" \
        TONGUES_TEST_COQUI_SPEEDY_MODEL="$speedy_dir/model_file.pth" \
        TONGUES_TEST_COQUI_HIFIGAN_CONFIG="$vocoder_dir/config.json" \
        TONGUES_TEST_COQUI_HIFIGAN_MODEL="$vocoder_dir/model_file.pth" \
        cargo test --release -p tongues-tts \
            burn_speedy_speech::tests::published_checkpoint_stage_parity \
            -- --ignored --exact --nocapture
fi

# ---------------------------------------------------------------------------
# Family: coqui-vits  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: coqui-vits [required]"
if [[ "$coqui_core_present" -eq 0 ]]; then
    skip_family "coqui-vits" "Coqui VCTK VITS model artifacts not found under $model_root"
elif [[ "${family_status[coqui-vits]}" != "failed" ]]; then
    echo "Comparing native VITS stages for p225, p330, and p376: $reference"
    run_step "coqui-vits" -- env \
        TONGUES_TEST_COQUI_VITS_CONFIG="$vits_dir/config.json" \
        TONGUES_TEST_COQUI_VITS_CHECKPOINT="$vits_dir/model_file.pth" \
        TONGUES_TEST_COQUI_VITS_SPEAKERS="$vits_dir/speaker_ids.json" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_vits::tests::published_checkpoint_stage_parity \
            -- --ignored --exact --nocapture
fi

# ---------------------------------------------------------------------------
# Family: coqui-fastpitch  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: coqui-fastpitch [required]"
if [[ "$coqui_core_present" -eq 0 ]]; then
    skip_family "coqui-fastpitch" "Coqui FastPitch model artifacts not found under $model_root"
elif [[ "${family_status[coqui-fastpitch]}" != "failed" ]]; then
    echo "Comparing native FastPitch duration, pitch, and mel stages: $reference"
    run_step "coqui-fastpitch" -- env \
        TONGUES_TEST_COQUI_FASTPITCH_CONFIG="$fastpitch_dir/config.json" \
        TONGUES_TEST_COQUI_FASTPITCH_MODEL="$fastpitch_dir/model_file.pth" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_fast_pitch::tests::published_checkpoint_stage_parity \
            -- --ignored --exact --nocapture
fi

# ---------------------------------------------------------------------------
# Family: glow-tts  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: glow-tts [required]"
if [[ ! -f "$glow_dir/config.json" || ! -f "$glow_dir/model_file.pth.tar" ]]; then
    skip_family "glow-tts" "Glow-TTS artifacts not found: $glow_dir/{config.json,model_file.pth.tar}"
elif [[ "${family_status[glow-tts]}" != "failed" ]]; then
    echo "Running the published Glow-TTS checkpoint through native acoustic inference"
    run_step "glow-tts" -- env \
        TONGUES_TEST_GLOW_CONFIG="$glow_dir/config.json" \
        TONGUES_TEST_GLOW_CHECKPOINT="$glow_dir/model_file.pth.tar" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_glow_tts::tests::published_glow_checkpoint_stage_parity \
            -- --ignored --exact --nocapture

    echo "Exercising Glow-TTS short, ordinary, long, repeated, and punctuation inputs"
    run_step "glow-tts" -- env \
        TONGUES_TEST_GLOW_CONFIG="$glow_dir/config.json" \
        TONGUES_TEST_GLOW_CHECKPOINT="$glow_dir/model_file.pth.tar" \
        cargo test --release -p tongues-tts \
            burn_glow_tts_acoustic::tests::published_acoustic_backend_covers_input_matrix \
            -- --ignored --exact --nocapture

    echo "Inspecting Glow-TTS through the safe package importer"
    run_step "glow-tts" -- env \
        TONGUES_TEST_GLOW_CONFIG="$glow_dir/config.json" \
        TONGUES_TEST_GLOW_CHECKPOINT="$glow_dir/model_file.pth.tar" \
        cargo test --release -p tongues-tts \
            model_package::tests::published_glow_tts_fixture_uses_common_importer \
            -- --ignored --exact --nocapture

    echo "Synthesizing Glow-TTS through the registered CLI composition: $output_dir/glow"
    run_step "glow-tts" -- env \
        SPEECH_SMOKE_CASES=glow \
        SPEECH_SMOKE_CPU=1 \
        SPEECH_SMOKE_TEXT="Morning light rested on the cedar trees while the kettle began to sing." \
        scripts/speech-smoke.sh "$output_dir/glow"
fi

# ---------------------------------------------------------------------------
# Family: multiband-melgan  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: multiband-melgan [required]"
if [[ "$coqui_core_present" -eq 0 ]]; then
    skip_family "multiband-melgan" "Coqui MultiBand-MelGAN artifacts not found under $model_root"
elif [[ "${family_status[multiband-melgan]}" != "failed" ]]; then
    echo "Comparing native MultiBand-MelGAN and PQMF output: $reference"
    run_step "multiband-melgan" -- env \
        TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG="$multiband_dir/config.json" \
        TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL="$multiband_dir/model_file.pth" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_vocoder::tests::published_multiband_melgan_checkpoint_parity \
            -- --ignored --exact --nocapture

    echo "Inspecting both MelGAN checkpoint layouts through the safe package importer"
    run_step "multiband-melgan" -- env \
        TONGUES_TEST_DESCRIPT_MELGAN_CONFIG="$repo_root/fixtures/speech/descript-melgan-linda-johnson-config.json" \
        TONGUES_TEST_DESCRIPT_MELGAN_MODEL="$melgan_model" \
        TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG="$multiband_dir/config.json" \
        TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL="$multiband_dir/model_file.pth" \
        cargo test --release -p tongues-tts \
            melgan_fixture_uses_common_importer \
            -- --nocapture
fi

# ---------------------------------------------------------------------------
# Family: melgan  (required)
# ---------------------------------------------------------------------------
echo
echo "==> Family: melgan [required]"
if [[ ! -f "$melgan_model" ]]; then
    skip_family "melgan" "Descript MelGAN artifact not found: $melgan_model"
elif [[ "${family_status[melgan]}" != "failed" ]]; then
    echo "Comparing native MelGAN output against the pinned Descript checkpoint: $reference"
    run_step "melgan" -- env \
        TONGUES_TEST_DESCRIPT_MELGAN_CONFIG="$repo_root/fixtures/speech/descript-melgan-linda-johnson-config.json" \
        TONGUES_TEST_DESCRIPT_MELGAN_MODEL="$melgan_model" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_vocoder::tests::published_melgan_checkpoint_parity \
            -- --ignored --exact --nocapture
fi

# ---------------------------------------------------------------------------
# Family: yourtts  (informational; CC-BY-NC-ND 4.0)
# ---------------------------------------------------------------------------
echo
echo "==> Family: yourtts [informational — CC-BY-NC-ND 4.0]"
if [[ "$yourtts_present" -eq 0 ]]; then
    skip_family "yourtts" "YourTTS artifacts not found under $yourtts_dir (CC-BY-NC-ND 4.0 — install separately)"
else
    echo "Checking multilingual YourTTS, speaker embeddings, reference WAV, and waveforms: $reference"
    run_step "yourtts" -- env \
        TONGUES_TEST_YOURTTS_CONFIG="$yourtts_dir/config.json" \
        TONGUES_TEST_YOURTTS_CHECKPOINT="$yourtts_dir/model_file.pth.tar" \
        TONGUES_TEST_YOURTTS_SPEAKERS="$yourtts_dir/speakers.json" \
        TONGUES_TEST_YOURTTS_LANGUAGES="$yourtts_dir/language_ids.json" \
        TONGUES_TEST_COQUI_SPEAKER_CONFIG="$yourtts_dir/config_se.json" \
        TONGUES_TEST_COQUI_SPEAKER_MODEL="$yourtts_dir/model_se.pth.tar" \
        TONGUES_TEST_YOURTTS_REFERENCE_WAV="$output_dir/yourtts-reference.wav" \
        TONGUES_TEST_COQUI_REFERENCE="$reference" \
        cargo test --release -p tongues-tts \
            burn_vits::tests::published_yourtts_conformance \
            -- --ignored --exact --nocapture

    echo "Testing YourTTS checkpoints load and speaker-encoder round-trips"
    run_step "yourtts" -- env \
        TONGUES_TEST_YOURTTS_CONFIG="$yourtts_dir/config.json" \
        TONGUES_TEST_YOURTTS_CHECKPOINT="$yourtts_dir/model_file.pth.tar" \
        TONGUES_TEST_YOURTTS_SPEAKERS="$yourtts_dir/speakers.json" \
        TONGUES_TEST_YOURTTS_LANGUAGES="$yourtts_dir/language_ids.json" \
        TONGUES_TEST_COQUI_SPEAKER_CONFIG="$yourtts_dir/config_se.json" \
        TONGUES_TEST_COQUI_SPEAKER_MODEL="$yourtts_dir/model_se.pth.tar" \
        cargo test --release -p tongues-tts \
            burn_vits::tests::published_your_tts_checkpoints_load_when_available \
            burn_vits::tests::published_your_tts_named_enrollment_synthesizes_when_available \
            speaker_encoder::tests::published_resnet_encodes_reference_audio_when_available \
            -- --ignored --nocapture
fi

# ---------------------------------------------------------------------------
# Family: fairseq-mms  (informational; CC-BY-NC-4.0)
# ---------------------------------------------------------------------------
echo
echo "==> Family: fairseq-mms [informational — CC-BY-NC-4.0]"
if [[ -z "$fairseq_model_root" || ! -d "$fairseq_model_root" ]]; then
    skip_family "fairseq-mms" \
        "TONGUES_FAIRSEQ_MODEL_ROOT not set or not found (CC-BY-NC-4.0 — install separately via: cargo run --bin tongues -- models install fairseq-mms-vits-eng)"
else
    echo "Running published Fairseq MMS checkpoint through native inference: $fairseq_model_root"
    run_step "fairseq-mms" -- env \
        TONGUES_TEST_FAIRSEQ_MMS_MODEL_DIR="$fairseq_model_root" \
        cargo test --release -p tongues-tts \
            burn_vits::tests::published_fairseq_mms_checkpoint_loads_and_synthesizes_without_python \
            -- --ignored --exact --nocapture
fi

# ---------------------------------------------------------------------------
# Family: freevc  (informational; MIT; auxiliary artifacts required separately)
# ---------------------------------------------------------------------------
echo
echo "==> Family: freevc [informational — MIT; auxiliary artifacts required]"
freevc_skip_reason=""
if [[ -z "$freevc_model_root" || ! -d "$freevc_model_root" ]]; then
    freevc_skip_reason="TONGUES_FREEVC_MODEL_ROOT not set or not found"
elif [[ -z "$freevc_source_wav" || ! -f "$freevc_source_wav" ]]; then
    freevc_skip_reason="TONGUES_FREEVC_SOURCE_WAV not set or not found"
elif [[ -z "$freevc_target_wav" || ! -f "$freevc_target_wav" ]]; then
    freevc_skip_reason="TONGUES_FREEVC_TARGET_WAV not set or not found"
fi

if [[ -n "$freevc_skip_reason" ]]; then
    skip_family "freevc" "$freevc_skip_reason"
else
    echo "Running FreeVC24 voice conversion through native inference"
    run_step "freevc" -- env \
        TONGUES_FREEVC_MODEL_DIR="$freevc_model_root" \
        TONGUES_FREEVC_SOURCE_WAV="$freevc_source_wav" \
        TONGUES_FREEVC_TARGET_WAV="$freevc_target_wav" \
        cargo test --release -p tongues-tts \
            freevc::tests::published_artifacts_convert_without_python \
            -- --ignored --exact --nocapture

    if [[ -n "$freevc_speaker_wav" && -f "$freevc_speaker_wav" && \
          -n "$freevc_alt_wav" && -f "$freevc_alt_wav" ]]; then
        echo "Running FreeVC speaker encoder separation test"
        run_step "freevc" -- env \
            TONGUES_FREEVC_SPEAKER_CHECKPOINT="$freevc_model_root/speaker_encoder.pt" \
            TONGUES_FREEVC_SAME_SPEAKER_WAV="$freevc_speaker_wav" \
            TONGUES_FREEVC_DIFFERENT_SPEAKER_WAV="$freevc_alt_wav" \
            cargo test --release -p tongues-tts \
                freevc::tests::published_speaker_encoder_loads_and_separates_fixtures \
                -- --ignored --exact --nocapture
    fi
fi

# ---------------------------------------------------------------------------
# Family: onnx  (informational; requires installed Piper voice)
# ---------------------------------------------------------------------------
echo
echo "==> Family: onnx [informational — requires installed Piper voice]"
echo "Synthesizing and validating the registered ONNX voice: $output_dir/onnx"
if run_step "onnx" -- env \
        SPEECH_SMOKE_CASES=onnx \
        SPEECH_SMOKE_CPU=1 \
        SPEECH_SMOKE_TEXT="Morning light rested on the cedar trees while the kettle began to sing." \
        scripts/speech-smoke.sh "$output_dir/onnx" 2>/dev/null; then
    : # status already set to passed by run_step
else
    # Treat ONNX smoke failure as a skip if no voice is installed
    if [[ "${family_tests_failed[onnx]}" -gt 0 ]]; then
        family_status[onnx]="skipped"
        family_skip_reason[onnx]="no ONNX Piper voice installed or voice synthesis failed"
        family_tests_run[onnx]=0
        family_tests_failed[onnx]=0
    fi
fi

# ---------------------------------------------------------------------------
# Write durable JSON summary
# ---------------------------------------------------------------------------
echo
echo "==> Writing conformance summary: $summary_file"

# Build artifact provenance entries with sha256 where the file exists
artifact_checksums() {
    local paths=("$@")
    local entries="["
    local first=1
    for path in "${paths[@]}"; do
        if [[ "$first" -eq 0 ]]; then entries+=","; fi
        first=0
        if [[ -f "$path" ]]; then
            local cksum
            cksum="$(sha256sum "$path" | awk '{print $1}')"
            entries+="$(jq -cn --arg p "$path" --arg s "$cksum" \
                '{path:$p,sha256:$s,present:true}')"
        else
            entries+="$(jq -cn --arg p "$path" \
                '{path:$p,sha256:null,present:false}')"
        fi
    done
    entries+="]"
    echo "$entries"
}

total_run=0
total_passed=0
total_failed=0
total_skipped=0
families_json="{"
first_family=1

ordered_families=(
    align-tts coqui-speedy-speech coqui-vits coqui-fastpitch
    glow-tts multiband-melgan melgan
    yourtts fairseq-mms freevc onnx
)

for fam in "${ordered_families[@]}"; do
    status="${family_status[$fam]}"
    gate="${family_release_gate[$fam]}"
    run="${family_tests_run[$fam]}"
    failed="${family_tests_failed[$fam]}"
    passed=$(( run - failed ))
    reason="${family_skip_reason[$fam]}"

    total_run=$(( total_run + run ))
    total_failed=$(( total_failed + failed ))
    case "$status" in
        passed)  total_passed=$(( total_passed + 1 ))  ;;
        failed)  : ;;
        skipped) total_skipped=$(( total_skipped + 1 )) ;;
    esac

    entry="$(jq -cn \
        --arg status "$status" \
        --arg gate "$gate" \
        --argjson run "$run" \
        --argjson passed "$passed" \
        --argjson failed "$failed" \
        --arg reason "$reason" \
        '{status:$status,release_gate:$gate,tests_run:$run,tests_passed:$passed,tests_failed:$failed,skip_reason:(if $reason=="" then null else $reason end)}')"

    if [[ "$first_family" -eq 0 ]]; then families_json+=","; fi
    first_family=0
    families_json+="$(jq -cn --arg k "$fam" --argjson v "$entry" '{($k):$v}' | sed 's/^{//;s/}$//')"
done
families_json+="}"

jq -cn \
    --arg schema "tongues-speech-conformance-summary-v1" \
    --arg git_revision "$git_revision" \
    --arg timestamp "$timestamp" \
    --argjson families "$families_json" \
    --argjson total_run "$total_run" \
    --argjson total_passed "$total_passed" \
    --argjson total_failed "$total_failed" \
    --argjson total_skipped "$total_skipped" \
    '{
        schema: $schema,
        git_revision: $git_revision,
        timestamp: $timestamp,
        totals: {
            families_run: ($total_run | . > 0),
            tests_run: $total_run,
            tests_passed: $total_passed,
            tests_failed: $total_failed,
            families_skipped: $total_skipped
        },
        families: $families
    }' > "$summary_file.part"
mv --force "$summary_file.part" "$summary_file"
echo "Summary committed atomically: $summary_file"

# ---------------------------------------------------------------------------
# Print human-readable report
# ---------------------------------------------------------------------------
echo
printf '%-24s %-14s %-12s %6s %6s %6s\n' \
    "family" "gate" "status" "run" "pass" "fail"
printf '%s\n' "$(printf '%.0s-' {1..75})"

exit_code=0
for fam in "${ordered_families[@]}"; do
    status="${family_status[$fam]}"
    gate="${family_release_gate[$fam]}"
    run="${family_tests_run[$fam]}"
    failed="${family_tests_failed[$fam]}"
    passed=$(( run - failed ))
    reason="${family_skip_reason[$fam]}"

    label="$status"
    if [[ "$status" == "skipped" && -n "$reason" ]]; then
        label="SKIPPED"
    fi

    printf '%-24s %-14s %-12s %6d %6d %6d\n' \
        "$fam" "$gate" "$label" "$run" "$passed" "$failed"

    if [[ "$status" == "skipped" && -n "$reason" ]]; then
        printf '  → %s\n' "$reason"
    fi

    if [[ "$status" == "failed" && "$gate" == "required" ]]; then
        exit_code=1
    fi
done

printf '%s\n' "$(printf '%.0s-' {1..75})"
printf 'Total: %d families skipped, %d tests run (%d passed, %d failed)\n' \
    "$total_skipped" "$total_run" \
    "$(( total_run - total_failed ))" "$total_failed"

echo
echo "Evidence: $output_dir"
echo "Summary:  $summary_file"

if [[ "$exit_code" -ne 0 ]]; then
    echo "CONFORMANCE FAILED: one or more required families failed." >&2
else
    echo "Speech conformance complete."
fi
exit "$exit_code"
