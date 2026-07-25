#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scope="${1:-all}"

run_check() {
    printf '\n==> %s\n' "$*"
    "$@"
}

check_workspace() {
    run_check cargo check --workspace --all-targets
    run_check cargo check --workspace --all-targets --no-default-features
    run_check cargo check --workspace --all-targets --all-features
}

check_package_features() {
    run_check cargo check -p speaking --all-targets --no-default-features
    run_check cargo check -p speaking --all-targets --no-default-features \
        --features asr-whisper

    run_check cargo check -p tongues-tts --all-targets --no-default-features
    run_check cargo check -p tongues-tts --all-targets --no-default-features \
        --features onnx-tts

    run_check cargo check -p tongues-cli --all-targets --no-default-features
    run_check cargo check -p tongues-cli --all-targets --no-default-features \
        --features styletts2-onnx
    run_check cargo check -p tongues-cli --all-targets --no-default-features \
        --features onnx-tts
    run_check cargo check -p tongues-cli --all-targets --no-default-features \
        --features styletts2-onnx,onnx-tts

    local style_features=(
        ""
        "styletts2-onnx"
        "styletts2-onnx-cuda"
        "styletts2-onnx-onednn"
        "styletts2-onnx-xnnpack"
        "styletts2-onnx-cuda,styletts2-onnx-onednn"
        "styletts2-onnx-cuda,styletts2-onnx-xnnpack"
        "styletts2-onnx-onednn,styletts2-onnx-xnnpack"
        "styletts2-onnx-cuda,styletts2-onnx-onednn,styletts2-onnx-xnnpack"
    )
    for features in "${style_features[@]}"; do
        if [[ -z "$features" ]]; then
            run_check cargo check -p styletts2 --all-targets --no-default-features
        else
            run_check cargo check -p styletts2 --all-targets --no-default-features \
                --features "$features"
        fi
    done
}

case "$scope" in
    workspace)
        check_workspace
        ;;
    powerset)
        check_package_features
        ;;
    all)
        check_workspace
        check_package_features
        ;;
    *)
        echo "usage: $0 [workspace|powerset|all]" >&2
        exit 2
        ;;
esac
