#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

scope="${1:-all}"

run_clippy_gate() {
    printf '\n==> %s\n' "$*"
    "$@" --no-deps -- -D clippy::all
}

run_speaking_baseline() {
    printf '\n==> %s (visible baseline; warning policy is maintained with speaking)\n' "$*"
    "$@"
}

clippy_workspace() {
    # `speaking` historically carried its own warning policy. Keep its findings
    # visible without reintroducing a workspace-wide suppression; cleanup is
    # coordinated in the speaking-specific work tracked by issue #108.
    run_speaking_baseline cargo clippy -p speaking --all-targets --locked
    run_clippy_gate cargo clippy --workspace --exclude speaking --all-targets --locked
    run_clippy_gate cargo clippy --workspace --exclude speaking --all-targets \
        --no-default-features --locked
    run_clippy_gate cargo clippy --workspace --exclude speaking --all-targets \
        --all-features --locked
}

clippy_package_features() {
    run_speaking_baseline cargo clippy -p speaking --all-targets --no-default-features --locked
    run_speaking_baseline cargo clippy -p speaking --all-targets --no-default-features \
        --features asr-whisper --locked

    run_clippy_gate cargo clippy -p tongues-tts --all-targets --no-default-features --locked
    run_clippy_gate cargo clippy -p tongues-tts --all-targets --no-default-features \
        --features onnx-tts --locked

    run_clippy_gate cargo clippy -p tongues-cli --all-targets --no-default-features --locked
    run_clippy_gate cargo clippy -p tongues-cli --all-targets --no-default-features \
        --features styletts2-onnx --locked
    run_clippy_gate cargo clippy -p tongues-cli --all-targets --no-default-features \
        --features onnx-tts --locked
    run_clippy_gate cargo clippy -p tongues-cli --all-targets --no-default-features \
        --features styletts2-onnx,onnx-tts --locked

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
            run_clippy_gate cargo clippy -p styletts2 --all-targets --no-default-features --locked
        else
            run_clippy_gate cargo clippy -p styletts2 --all-targets --no-default-features \
                --features "$features" --locked
        fi
    done
}

case "$scope" in
    workspace)
        clippy_workspace
        ;;
    powerset)
        clippy_package_features
        ;;
    all)
        clippy_workspace
        clippy_package_features
        ;;
    *)
        echo "usage: $0 [workspace|powerset|all]" >&2
        exit 2
        ;;
esac
