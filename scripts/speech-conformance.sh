#!/usr/bin/env bash
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
melgan_model="$model_root/ljspeech/melgan/linda_johnson.pt"
vits_dir="$model_root/vctk/vits"
output_dir="${TONGUES_CONFORMANCE_OUTPUT:-$repo_root/target/speech-conformance}"
image="${TONGUES_COQUI_REFERENCE_IMAGE:-tongues-coqui-reference}"

required_artifacts=(
    "$speedy_dir/config.json"
    "$speedy_dir/model_file.pth"
    "$fastpitch_dir/config.json"
    "$fastpitch_dir/model_file.pth"
    "$vocoder_dir/config.json"
    "$vocoder_dir/model_file.pth"
    "$multiband_dir/config.json"
    "$multiband_dir/model_file.pth"
    "$multiband_dir/scale_stats.npy"
    "$melgan_model"
    "$vits_dir/config.json"
    "$vits_dir/model_file.pth"
    "$vits_dir/speaker_ids.json"
)
for artifact in "${required_artifacts[@]}"; do
    if [[ ! -f "$artifact" ]]; then
        echo "required full-model conformance artifact is missing: $artifact" >&2
        echo "Full conformance cannot be reported as passing without every pinned artifact." >&2
        exit 2
    fi
done

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
reference="$output_dir/coqui-reference.json"
echo "Building pinned Coqui reference runtime: $image"
docker build --tag "$image" --file tools/speech-conformance/Dockerfile .

echo "Generating Coqui stage evidence: $reference.part"
docker run --rm \
    --volume "$model_root:/models:ro" \
    --volume "$repo_root:/workspace" \
    --volume "$output_dir:/evidence" \
    "$image" \
    --model-root /models \
    --output /evidence/coqui-reference.json.part
mv "$reference.part" "$reference"
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
    exit 1
fi
mv "$tokenization.part" "$tokenization"
rm "$tokenization.expected.part"
echo "Pinned tokenizer fixture matched: $tokenization"

echo "Comparing native SpeedySpeech and HiFi-GAN stages: $reference"
env \
    TONGUES_TEST_COQUI_SPEEDY_CONFIG="$speedy_dir/config.json" \
    TONGUES_TEST_COQUI_SPEEDY_MODEL="$speedy_dir/model_file.pth" \
    TONGUES_TEST_COQUI_HIFIGAN_CONFIG="$vocoder_dir/config.json" \
    TONGUES_TEST_COQUI_HIFIGAN_MODEL="$vocoder_dir/model_file.pth" \
    cargo test --release -p tongues-tts \
        burn_speedy_speech::tests::published_checkpoint_stage_parity \
        -- --ignored --exact --nocapture

echo "Comparing native VITS stages for p225, p330, and p376: $reference"
env \
    TONGUES_TEST_COQUI_VITS_CONFIG="$vits_dir/config.json" \
    TONGUES_TEST_COQUI_VITS_CHECKPOINT="$vits_dir/model_file.pth" \
    TONGUES_TEST_COQUI_VITS_SPEAKERS="$vits_dir/speaker_ids.json" \
    TONGUES_TEST_COQUI_REFERENCE="$reference" \
    cargo test --release -p tongues-tts \
        burn_vits::tests::published_checkpoint_stage_parity \
        -- --ignored --exact --nocapture

echo "Comparing native FastPitch duration, pitch, and mel stages: $reference"
env \
    TONGUES_TEST_COQUI_FASTPITCH_CONFIG="$fastpitch_dir/config.json" \
    TONGUES_TEST_COQUI_FASTPITCH_MODEL="$fastpitch_dir/model_file.pth" \
    TONGUES_TEST_COQUI_REFERENCE="$reference" \
    cargo test --release -p tongues-tts \
        burn_fast_pitch::tests::published_checkpoint_stage_parity \
        -- --ignored --exact --nocapture

echo "Comparing native MultiBand-MelGAN and PQMF output: $reference"
env \
    TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG="$multiband_dir/config.json" \
    TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL="$multiband_dir/model_file.pth" \
    TONGUES_TEST_COQUI_REFERENCE="$reference" \
    cargo test --release -p tongues-tts \
        burn_vocoder::tests::published_multiband_melgan_checkpoint_parity \
        -- --ignored --exact --nocapture

echo "Comparing native MelGAN output against the pinned Descript checkpoint: $reference"
env \
    TONGUES_TEST_DESCRIPT_MELGAN_CONFIG="$repo_root/fixtures/speech/descript-melgan-linda-johnson-config.json" \
    TONGUES_TEST_DESCRIPT_MELGAN_MODEL="$melgan_model" \
    TONGUES_TEST_COQUI_REFERENCE="$reference" \
    cargo test --release -p tongues-tts \
        burn_vocoder::tests::published_melgan_checkpoint_parity \
        -- --ignored --exact --nocapture

echo "Inspecting both MelGAN checkpoint layouts through the safe package importer"
env \
    TONGUES_TEST_DESCRIPT_MELGAN_CONFIG="$repo_root/fixtures/speech/descript-melgan-linda-johnson-config.json" \
    TONGUES_TEST_DESCRIPT_MELGAN_MODEL="$melgan_model" \
    TONGUES_TEST_COQUI_MULTIBAND_MELGAN_CONFIG="$multiband_dir/config.json" \
    TONGUES_TEST_COQUI_MULTIBAND_MELGAN_MODEL="$multiband_dir/model_file.pth" \
    cargo test --release -p tongues-tts \
        melgan_fixture_uses_common_importer \
        -- --nocapture

echo "Synthesizing and validating the registered ONNX voice: $output_dir/onnx"
SPEECH_SMOKE_CASES=onnx \
SPEECH_SMOKE_CPU=1 \
SPEECH_SMOKE_TEXT="Morning light rested on the cedar trees while the kettle began to sing." \
    scripts/speech-smoke.sh "$output_dir/onnx"

echo "Speech conformance passed; evidence is under $output_dir"
