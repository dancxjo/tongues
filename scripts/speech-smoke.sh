#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-target/speech-smoke/$run_id}"
binary="${TONGUES_BIN:-$repo_root/target/release/tongues}"
text="${SPEECH_SMOKE_TEXT:-The quick brown fox jumps over the sleeping dog.}"
results_jsonl="$output_dir/results.jsonl"
results_json="$output_dir/results.json"
if ! mkdir -p "$output_dir"; then
    echo "could not create smoke output directory: $output_dir" >&2
    exit 2
fi

if [[ ! -x "$binary" ]]; then
    echo "release binary is missing; building $binary"
    cargo build --release -p tongues-cli
fi

device_args=()
device_request="auto"
if [[ "${SPEECH_SMOKE_CPU:-0}" == "1" ]]; then
    device_args=(--cpu)
    device_request="cpu"
fi

git_revision="$(git rev-parse HEAD)"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
host_name="$(hostname)"
failures=0

run_case() {
    local name="$1"
    shift
    local wav="$output_dir/$name.wav"
    local log="$output_dir/$name.log"
    local timing="$output_dir/$name.time"
    local started_at
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    printf '\n==> %s -> %s\n' "$name" "$wav"
    /usr/bin/time -f '%e\t%M' -o "$timing" \
        "$binary" "${device_args[@]}" --verbose speak \
        --output "$wav" "$@" "$text" >"$log" 2>&1
    local status=$?
    local elapsed_seconds="0"
    local max_rss_kb="0"
    if [[ -s "$timing" ]]; then
        IFS=$'\t' read -r elapsed_seconds max_rss_kb < <(tail -n 1 "$timing")
    fi

    if [[ $status -ne 0 || ! -s "$wav" ]]; then
        failures=$((failures + 1))
        jq -cn \
            --arg name "$name" \
            --arg started_at "$started_at" \
            --arg git_revision "$git_revision" \
            --arg device_request "$device_request" \
            --arg gpu_name "$gpu_name" \
            --arg host "$host_name" \
            --arg log "$log" \
            --argjson exit_code "$status" \
            --arg elapsed_seconds "$elapsed_seconds" \
            --arg max_rss_kb "$max_rss_kb" \
            '{
                name: $name,
                started_at: $started_at,
                git_revision: $git_revision,
                host: $host,
                gpu_name: $gpu_name,
                device_request: $device_request,
                exit_code: $exit_code,
                elapsed_seconds: ($elapsed_seconds | tonumber),
                max_rss_kb: ($max_rss_kb | tonumber),
                log: $log
            }' >> "$results_jsonl"
        echo "FAILED: $name (see $log)"
        return
    fi

    local probe
    probe="$(ffprobe -v error -select_streams a:0 \
        -show_entries stream=codec_name,sample_rate,channels,bits_per_sample \
        -show_entries format=duration,size \
        -of json "$wav")"
    local volume
    volume="$(ffmpeg -nostats -i "$wav" -af volumedetect -f null - 2>&1)"
    local mean_db
    local peak_db
    mean_db="$(sed -n 's/.*mean_volume: \([-0-9.]*\) dB.*/\1/p' <<<"$volume" | tail -n 1)"
    peak_db="$(sed -n 's/.*max_volume: \([-0-9.]*\) dB.*/\1/p' <<<"$volume" | tail -n 1)"
    local sha256
    sha256="$(sha256sum "$wav" | awk '{print $1}')"

    jq -cn \
        --arg name "$name" \
        --arg started_at "$started_at" \
        --arg git_revision "$git_revision" \
        --arg device_request "$device_request" \
        --arg gpu_name "$gpu_name" \
        --arg host "$host_name" \
        --arg wav "$wav" \
        --arg log "$log" \
        --arg sha256 "$sha256" \
        --arg elapsed_seconds "$elapsed_seconds" \
        --arg max_rss_kb "$max_rss_kb" \
        --arg mean_db "${mean_db:--inf}" \
        --arg peak_db "${peak_db:--inf}" \
        --argjson probe "$probe" \
        '{
            name: $name,
            started_at: $started_at,
            git_revision: $git_revision,
            host: $host,
            gpu_name: $gpu_name,
            device_request: $device_request,
            exit_code: 0,
            elapsed_seconds: ($elapsed_seconds | tonumber),
            max_rss_kb: ($max_rss_kb | tonumber),
            wav: $wav,
            log: $log,
            sha256: $sha256,
            audio: {
                codec: $probe.streams[0].codec_name,
                sample_rate_hz: ($probe.streams[0].sample_rate | tonumber),
                channels: $probe.streams[0].channels,
                bits_per_sample: $probe.streams[0].bits_per_sample,
                duration_seconds: ($probe.format.duration | tonumber),
                file_bytes: ($probe.format.size | tonumber),
                mean_dbfs: (if $mean_db == "-inf" then $mean_db else ($mean_db | tonumber) end),
                peak_dbfs: (if $peak_db == "-inf" then $peak_db else ($peak_db | tonumber) end)
            },
            real_time_factor: (
                ($elapsed_seconds | tonumber) / ($probe.format.duration | tonumber)
            )
        }' >> "$results_jsonl"
    echo "PASS: $name (${elapsed_seconds}s)"
}

run_case burn --backend burn
run_case vits-p225 --backend vits --speaker p225
run_case vits-p330 --backend vits --speaker p330
run_case onnx --backend onnx
run_case styletts2 --backend styletts2 --quality fast

jq -s '.' "$results_jsonl" > "$results_json"
printf '\n%-14s %9s %9s %8s %8s %8s\n' \
    "case" "wall(s)" "audio(s)" "RTF" "mean dB" "peak dB"
jq -r '.[] | [
    .name,
    (.elapsed_seconds | tostring),
    ((.audio.duration_seconds // 0) | tostring),
    ((.real_time_factor // 0) | tostring),
    ((.audio.mean_dbfs // "n/a") | tostring),
    ((.audio.peak_dbfs // "n/a") | tostring)
] | @tsv' "$results_json" | while IFS=$'\t' read -r name wall audio rtf mean peak; do
    printf '%-14s %9s %9s %8s %8s %8s\n' "$name" "$wall" "$audio" "$rtf" "$mean" "$peak"
done

echo
echo "Results: $results_json"
exit "$failures"
