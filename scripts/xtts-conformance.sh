#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for command in cargo cmp ffmpeg ffprobe; do
    if ! command -v "$command" >/dev/null; then
        echo "required XTTS conformance command is missing: $command" >&2
        exit 2
    fi
done

package="${TONGUES_XTTS_PACKAGE:?set TONGUES_XTTS_PACKAGE to an imported XTTS v2 package}"
reference_a="${TONGUES_XTTS_REFERENCE_A:?set TONGUES_XTTS_REFERENCE_A to a reference WAV}"
reference_b="${TONGUES_XTTS_REFERENCE_B:?set TONGUES_XTTS_REFERENCE_B to a second reference WAV}"
output_dir="${TONGUES_XTTS_CONFORMANCE_OUTPUT:-$repo_root/target/xtts-conformance}"
mkdir -p "$output_dir"

for path in "$package/model.json" "$package/model.safetensors" "$package/vocab.json" \
    "$reference_a" "$reference_b"; do
    if [[ ! -e "$path" ]]; then
        echo "required XTTS conformance input is missing: $path" >&2
        exit 2
    fi
done

device=()
if [[ "${TONGUES_XTTS_CPU:-0}" == "1" ]]; then
    device=(--cpu)
elif [[ -n "${TONGUES_XTTS_CUDA_DEVICE:-}" ]]; then
    device=(--cuda-device "$TONGUES_XTTS_CUDA_DEVICE")
fi

cargo build --release --bin tongues

synthesize() {
    local language="$1"
    local reference="$2"
    local output="$3"
    local text="$4"
    shift 4
    /usr/bin/time -v target/release/tongues "${device[@]}" speak \
        --backend xtts \
        --model "$package" \
        --model-language "$language" \
        --voice-wav "$reference" \
        --seed 17 \
        --timings \
        --output "$output" \
        "$@" \
        "$text" \
        2>"$output.timings.txt"
    ffprobe -v error \
        -show_entries stream=sample_rate,channels,duration \
        -of default=noprint_wrappers=1 "$output" >"$output.probe.txt"
    ffmpeg -v error -i "$output" -f null -
}

echo "Synthesizing the two-language, two-reference XTTS matrix"
synthesize en "$reference_a" "$output_dir/en-reference-a.wav" "Hello friend."
synthesize fr "$reference_a" "$output_dir/fr-reference-a.wav" "Bonjour mon ami."
synthesize en "$reference_b" "$output_dir/en-reference-b.wav" "Hello friend."
synthesize fr "$reference_b" "$output_dir/fr-reference-b.wav" "Bonjour mon ami."

echo "Comparing concatenated streaming output with one-shot decoding"
synthesize en "$reference_a" "$output_dir/en-reference-a-one-shot.wav" \
    "Hello friend." --no-tts-chunking
if ! cmp --silent \
    "$output_dir/en-reference-a.wav" \
    "$output_dir/en-reference-a-one-shot.wav"; then
    echo "XTTS streaming and one-shot WAVs differ" >&2
    exit 1
fi

echo "XTTS conformance passed: $output_dir"
