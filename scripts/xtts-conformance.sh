#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for command in cargo cmp ffmpeg ffprobe jq; do
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
        --benchmark-runs 2 \
        --seed 17 \
        --timings \
        --output "$output" \
        "$@" \
        "$text" \
        >"$output.log" \
        2>"$output.timings.txt"
    sed -n 's/^inference_profile_json: //p' "$output.log" >"$output.profile.jsonl"
    if [[ ! -s "$output.profile.jsonl" ]]; then
        echo "missing XTTS inference_profile_json output: $output.log" >&2
        exit 1
    fi
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

primary_profile="$output_dir/en-reference-a.wav.profile.jsonl"
primary_timing="$output_dir/en-reference-a.wav.timings.txt"
peak_ram_bytes="$(awk -F: '/Maximum resident set size \(kbytes\)/ {
    gsub(/^[ \t]+/, "", $2);
    print ($2 + 0) * 1024;
    exit
}' "$primary_timing")"
if [[ -z "$peak_ram_bytes" ]]; then
    peak_ram_bytes="null"
fi
if command -v nvidia-smi >/dev/null; then
    peak_gpu_memory_bytes="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        2>/dev/null | awk 'NR==1 {print ($1 + 0) * 1024 * 1024}')"
else
    peak_gpu_memory_bytes=""
fi
if [[ -z "$peak_gpu_memory_bytes" ]]; then
    peak_gpu_memory_bytes="null"
fi

jq -cs \
    --argjson peak_ram_bytes "$peak_ram_bytes" \
    --argjson peak_gpu_memory_bytes "$peak_gpu_memory_bytes" '
    def stage_total_ms($profile; $stage):
        ([ $profile.stages[]? | select(.stage == $stage) | .elapsed_ms ] | add // 0);
    def stage_last_ms($profile; $stage):
        ([ $profile.stages[]? | select(.stage == $stage) ] | last | .elapsed_ms // 0);
    def first_profile($kind):
        ([ .[] | select(.temperature == $kind) ][0]);

    (first_profile("cold")) as $cold |
    (first_profile("warm")) as $warm |
    {
        reference_preparation_ms: stage_total_ms($cold; "reference_conditioning"),
        gpt_first_code_latency_ms: stage_total_ms($cold; "autoregressive_first_code"),
        stable_first_audio_latency_ms: $cold.first_playable_audio_latency_ms,
        steady_real_time_factor: ($warm.real_time_factor // null),
        overlap_recompute_decode_ms: (
            (stage_total_ms($cold; "waveform_decoder")
                - stage_last_ms($cold; "waveform_decoder"))
            | if . < 0 then 0 else . end
        ),
        cancellation: {
            measured: false,
            note: "native sink errors abort synthesis immediately; CLI timings do not yet expose cancellation latency"
        },
        peak_ram_bytes: $peak_ram_bytes,
        peak_gpu_memory_bytes: $peak_gpu_memory_bytes
    }
' "$primary_profile" >"$output_dir/xtts-benchmark.json"

echo "XTTS conformance passed: $output_dir"
echo "XTTS benchmark summary: $output_dir/xtts-benchmark.json"
