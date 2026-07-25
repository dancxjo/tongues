#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-target/speech-benchmark/$run_id}"
binary="${TONGUES_BIN:-$repo_root/target/release/tongues}"
devices="${TONGUES_BENCH_DEVICES:-cpu cuda}"
runs="${TONGUES_BENCH_RUNS:-3}"
warm_rtf_target="${TONGUES_WARM_RTF_TARGET:-1.0}"
mkdir -p "$output_dir"

if [[ ! -x "$binary" ]]; then
    echo "release binary is missing; building $binary"
    cargo build --release -p tongues-cli
fi

short_text="Morning light rested on the cedar trees."
paragraph_text="Morning light rested on the cedar trees while the kettle began to sing. A cool breeze moved through the open window, and the old clock marked the quiet start of another day."
results_jsonl="$output_dir/results.jsonl"
: > "$results_jsonl"
failures=0

run_case() {
    local device="$1"
    local backend="$2"
    local input_name="$3"
    local text="$4"
    local speaker_args=()
    local device_args=()
    local name="${device}-${backend}-${input_name}"
    local wav="$output_dir/$name.wav"
    local log="$output_dir/$name.log"

    if [[ "$device" == "cpu" ]]; then
        device_args=(--cpu)
    fi
    if [[ "$backend" == "vits" ]]; then
        speaker_args=(--speaker p225)
    fi

    printf '\n==> %s\n' "$name"
    "$binary" "${device_args[@]}" --verbose speak \
        --backend "$backend" \
        --benchmark-runs "$runs" \
        --timings \
        --seed 27 \
        --output "$wav" \
        "${speaker_args[@]}" \
        "$text" >"$log" 2>&1
    local status=$?
    if [[ $status -ne 0 ]]; then
        failures=$((failures + 1))
        jq -cn \
            --arg name "$name" \
            --arg log "$log" \
            --argjson exit_code "$status" \
            '{name: $name, exit_code: $exit_code, log: $log}' >>"$results_jsonl"
        echo "FAILED: $name (see $log)"
        return
    fi

    sed -n 's/^inference_profile_json: //p' "$log" |
        jq -c \
            --arg name "$name" \
            --arg device "$device" \
            --arg backend "$backend" \
            --arg input "$input_name" \
            --arg wav "$wav" \
            --arg log "$log" \
            '. + {
                name: $name,
                device: $device,
                backend: $backend,
                input: $input,
                wav: $wav,
                log: $log,
                exit_code: 0
            }' >>"$results_jsonl"

    local worst_warm_rtf
    worst_warm_rtf="$(
        sed -n 's/^inference_profile_json: //p' "$log" |
            jq -s '[.[] | select(.temperature == "warm") | .real_time_factor] | max // 0'
    )"
    if [[ "$device" == "cuda" ]] &&
        ! awk -v actual="$worst_warm_rtf" -v target="$warm_rtf_target" \
            'BEGIN { exit !(actual < target) }'; then
        failures=$((failures + 1))
        echo "TARGET MISSED: $name warm RTF $worst_warm_rtf >= $warm_rtf_target"
    else
        echo "PASS: $name warm worst RTF=$worst_warm_rtf"
    fi
}

for device in $devices; do
    for backend in burn vits; do
        run_case "$device" "$backend" short "$short_text"
        run_case "$device" "$backend" paragraph "$paragraph_text"
    done
done

jq -s '.' "$results_jsonl" >"$output_dir/results.json"
jq -r '
    ["case", "run", "kind", "synthesis_ms", "first_audio_ms", "audio_s", "RTF"],
    (.[] | select(.exit_code == 0) | [
        .name,
        .run,
        .temperature,
        (.total_synthesis_ms | tostring),
        (.first_playable_audio_latency_ms | tostring),
        (.audio_seconds | tostring),
        (.real_time_factor | tostring)
    ]) | @tsv
' "$output_dir/results.json" | column -t -s $'\t'

echo
echo "Results: $output_dir/results.json"
exit "$failures"
