set positional-arguments

default:
    @just --list

# Fetch checksum-pinned Cargo assets and prove the linguistic layer builds offline
prepare-assets:
    cargo fetch
    CARGO_NET_OFFLINE=true cargo check -p speaking
    python3 scripts/check-linguistic-assets.py

# Create a new model-family crate/config/artifact scaffold
new-family family:
    cargo run -q -p xtask -- new-family "{{family}}"

# Run a compact round-trip inference benchmark across G2P2G and Wiktionary models
race *args:
    @cargo run -q -p xtask -- race "$@"

# Generate text chunks, derive phones, and synthesize speech continuously
continue *args:
    @cargo run -q -p xtask -- continue "$@"

# Stream an Ollama story through resident head2phones and ONNX speech playback
be *args:
    @cargo run -q --bin tongues -- be "$@"

# Stream stdin through the sentence parser and emit one sentence per line
parse *args:
    @cargo run -q --bin tongues -- sentence-parser stream "$@"

# Quick split demos
split target='sentences':
    #!/usr/bin/env bash
    set -euo pipefail

    case "{{target}}" in
        sentences) just split-sentences ;;
        *)
            echo "Unknown split demo: {{target}}" >&2
            echo "Available: sentences" >&2
            exit 2
            ;;
    esac

# Demonstrate the current head2phones model pronouncing whole sentences on CPU
split-sentences:
    #!/usr/bin/env bash
    set -euo pipefail

    model="${HEAD2PHONES_MODEL:-models/head2phones/v0}"
    examples=(
        "en-US|To be, or not to be?"
        "fr-FR-Standard|Je pense, donc je suis."
        "de-DE-Standard|Am Brunnen vor dem Tore."
        "es-ES-Castilian|En un lugar de la Mancha."
        "eo|Ho, mia kor!"
        "la-Classical|Arma virumque cano."
    )

    echo "head2phones CPU sentence demo"
    echo "model: $model"
    echo

    for example in "${examples[@]}"; do
        variety="${example%%|*}"
        sentence="${example#*|}"
        echo "[$variety] $sentence"
        cargo run -q --bin tongues -- --cpu head2phones infer --model "$model" --variety "$variety" "$sentence"
        echo
    done

# Forward directly to the tongues CLI
run *args:
    cargo run --bin tongues -- "$@"

# Normalize, validate, split, and batch an LJSpeech/VCTK/generic corpus
speech-corpus *args:
    cargo run --bin tongues -- speech-corpus "$@"

# Package or extract versioned release artifacts
release action family:
    #!/usr/bin/env bash
    set -euo pipefail

    case "{{family}}" in
        head2phones)
            version="v0"
            release_dir="releases/head2phones-${version}"
            archive="${release_dir}/head2phones-${version}.tar.gz"
            source_dir="models/head2phones/${version}"
            ;;
        *)
            echo "Unknown release family: {{family}}" >&2
            echo "Available: head2phones" >&2
            exit 2
            ;;
    esac

    case "{{action}}" in
        package)
            mkdir -p "$release_dir"
            (
                cd "$source_dir"
                sha256sum \
                    model.bin \
                    model-epoch-4.bin \
                    vocab.json \
                    head2phones_config.json \
                    manifest.json \
                    model_config.json \
                    train_config.json \
                    train_state.json \
                    > SHA256SUMS
            )
            tar --sort=name \
                --mtime='2026-06-20 00:00:00Z' \
                --owner=0 \
                --group=0 \
                --numeric-owner \
                -czf "$archive" \
                -C "$source_dir" \
                SHA256SUMS \
                head2phones_config.json \
                manifest.json \
                model-epoch-4.bin \
                model.bin \
                model_config.json \
                train_config.json \
                train_state.json \
                vocab.json
            sha256sum "$archive" | sed "s#${release_dir}/##" > "${release_dir}/SHA256SUMS"
            echo "Packaged $archive"
            ;;
        extract)
            test -f "$archive" || {
                echo "Missing release archive: $archive" >&2
                exit 1
            }
            tmp="$(mktemp -d)"
            trap 'rm -rf "$tmp"' EXIT
            tar -xzf "$archive" -C "$tmp"
            (cd "$tmp" && sha256sum -c SHA256SUMS)
            mkdir -p "$source_dir"
            cp -a "$tmp"/. "$source_dir"/
            echo "Extracted $archive to $source_dir"
            ;;
        *)
            echo "Unknown release action: {{action}}" >&2
            echo "Available: package, extract" >&2
            exit 2
            ;;
    esac

# Forward a model-family command to the tongues CLI
g2p2g *args:
    cargo run --bin tongues -- g2p2g "$@"

# Forward a model-family command to the tongues CLI
wiktionary *args:
    cargo run --bin tongues -- wiktionary "$@"

# Forward a model-family command to the tongues CLI
sentence-parser *args:
    cargo run --bin tongues -- sentence-parser "$@"

# Forward a model-family command to the tongues CLI
head2phones *args:
    cargo run --bin tongues -- head2phones "$@"

# Forward a model-family command to the tongues CLI
interpretation *args:
    cargo run --bin tongues -- interpretation "$@"

# Forward a model-family command to the tongues CLI
common-phone *args:
    cargo run --bin tongues -- common-phone "$@"

# Alias for the canonical common-phone spelling
commonphone *args:
    cargo run --bin tongues -- common-phone "$@"

common-phone-prepare:
    cargo run --bin tongues -- common-phone prepare --input data/common-phone/raw --out models/common-phone/common-phone-v0

common-phone-show:
    cargo run --bin tongues -- common-phone show --data models/common-phone/common-phone-v0 --index 0

common-phone-train:
    cargo run --bin tongues -- common-phone train --data models/common-phone/common-phone-v0 --model models/common-phone/common-phone-v0-phone-ctc --task frames2phones

common-phone-eval:
    cargo run --bin tongues -- common-phone eval --data models/common-phone/common-phone-v0 --model models/common-phone/common-phone-v0-phone-ctc --split valid --task frames2phones

common-phone-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    fixture="/tmp/tongues-common-phone-fixture"
    rm -rf "$fixture" /tmp/tongues-common-phone-mini /tmp/tongues-common-phone-mini-model /tmp/tongues-common-phone-mini-train-eval.json
    mkdir -p "$fixture/audio"
    python3 - <<'PY' "$fixture"
    import json, math, struct, sys, wave
    root = sys.argv[1]
    rows = [
        ("mini_tip", "train", "tip", ["t", "ɪ", "p"], 220),
        ("mini_pit", "train", "pit", ["p", "ɪ", "t"], 260),
        ("mini_sip", "valid", "sip", ["s", "ɪ", "p"], 300),
    ]
    with open(f"{root}/metadata.jsonl", "w", encoding="utf-8") as meta:
        for utt, split, text, phones, freq in rows:
            wav = f"audio/{utt}.wav"
            with wave.open(f"{root}/{wav}", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                frames = []
                for n in range(4800):
                    frames.append(struct.pack("<h", int(10000 * math.sin(2 * math.pi * freq * n / 16000))))
                w.writeframes(b"".join(frames))
            meta.write(json.dumps({
                "utterance_id": utt,
                "language": "eng",
                "split": split,
                "wav_path": wav,
                "text": text,
                "phones": phones,
            }, ensure_ascii=False) + "\n")
    PY
    cargo run --bin tongues -- common-phone prepare --input "$fixture" --out /tmp/tongues-common-phone-mini
    cargo run --bin tongues -- common-phone show --data /tmp/tongues-common-phone-mini --index 0
    cargo run --bin tongues -- common-phone train --data /tmp/tongues-common-phone-mini --model /tmp/tongues-common-phone-mini-model --task frames2phones --epochs 100 --dropout 0 --lr 0.003 --device cpu
    cargo run --bin tongues -- common-phone eval --data /tmp/tongues-common-phone-mini --model /tmp/tongues-common-phone-mini-model --split train --task frames2phones --samples 2 > /tmp/tongues-common-phone-mini-train-eval.json
    python3 - <<'PY' /tmp/tongues-common-phone-mini-train-eval.json
    import json, sys
    with open(sys.argv[1], encoding="utf-8") as f:
        report = json.load(f)
    nonempty = [sample for sample in report["samples"] if sample["phone_prediction"]]
    blank_ratio = report["blank_ratio"]
    raw = report.get("raw_argmax", {})
    print(json.dumps({
        "split": report["split"],
        "blank_ratio": blank_ratio,
        "mean_prediction_length": report["mean_prediction_length"],
        "raw_argmax": raw,
        "samples": report["samples"],
    }, ensure_ascii=False, indent=2))
    if blank_ratio >= 1.0 or not nonempty:
        raise SystemExit("Common Phone tiny overfit failed: train split decoded only blanks")
    PY
    cargo run --bin tongues -- common-phone eval --data /tmp/tongues-common-phone-mini --model /tmp/tongues-common-phone-mini-model --split valid --task frames2phones

# Forward a model-family command to the tongues CLI
emotions *args:
    cargo run --bin tongues -- emotions "$@"

# Common typo for the interpretation recipe
interpreation *args:
    cargo run --bin tongues -- interpretation "$@"

# Forward a model-family command to the tongues CLI
# Prepare OpenEPD data splits and build vocabulary (runs prepare)
prepare *args:
    cargo run --bin tongues -- g2p2g prepare --out datasets/g2p2g/openepd-v0 "$@"

# Fetch/Download pronunciation lexicon data files
fetch *args:
    cargo run --bin tongues -- fetch-cmudict --out data/cmudict.dict "$@"
    cargo run --bin tongues -- fetch-lexique --out data/Lexique383.tsv "$@"

# Update the markdown table of pronunciation discrepancies across pronouncers
discrepancies *args:
    cargo run --bin tongues -- discrepancies "$@"

# Alias for the common misspelling
discrepencies *args:
    @just discrepancies "$@"

# Alias for the common misspelling
discrepency *args:
    @just discrepancies "$@"

# Move generated data, prepared runs, and model outputs aside for a fresh start
archive:
    #!/usr/bin/env bash
    set -euo pipefail

    archive_dir="archive/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$archive_dir"

    moved=0
    for path in data runs models; do
        if [ -e "$path" ]; then
            mv "$path" "$archive_dir/"
            moved=1
        fi
    done

    if [ "$moved" -eq 0 ]; then
        rmdir "$archive_dir"
        echo "Nothing to archive."
    else
        echo "Archived generated data, runs, and models to $archive_dir"
    fi

# Synthesize speech using any configured backend
speak *args:
    cargo run --bin tongues -- speak "$@"

# Benchmark cold and resident warm native speech inference on CPU and CUDA
speech-benchmark *args:
    scripts/speech-benchmark.sh "$@"

# Play shuffled sentences through resident speech backends, preferring CUDA
speech-demo *args:
    @cargo run -q -p xtask -- speech-demo "$@"

# Demonstrate the speaking library across every built-in language variety
speaking *args:
    cargo run --bin tongues -- speaking-demo "$@"

# Phonemize text into an IPA sequence
phonemes *args:
    cargo run --bin tongues -- phonemes "$@"

# Print narrow phonetic phones transcription
phones *args:
    cargo run --bin tongues -- phones "$@"

# Run translation prediction (graphemes to phonemes or vice-versa)
infer *args:
    cargo run --bin tongues -- g2p2g infer "$@"

# Train the tongues translation model with an even mix of both directions
train *args:
    cargo run --bin tongues -- g2p2g train --data datasets/g2p2g/openepd-v0 --out models/g2p2g/openepd-v0 --task both "$@"

# Refine the model on validation/test pronunciation discrepancies
refine *args:
    cargo run --bin tongues -- g2p2g refine --model models/g2p2g/openepd-v0 --data datasets/g2p2g/openepd-v0 --out models/g2p2g/openepd-v0-refined --verbose "$@"

# Fine-tune the model on the built-in Dolch sight-word list
sight-words *args:
    cargo run --bin tongues -- g2p2g refine --model models/g2p2g/openepd-v0 --data datasets/g2p2g/openepd-v0 --out models/g2p2g/openepd-v0-sight-words --source sight-words --task both --verbose "$@"

# Fetch all public emotion audio datasets into datasets/
fetch-corpora *args:
    cargo run --bin tongues -- fetch-corpora "$@"

# Start the web interface on 0.0.0.0:3000 plus HTTPS on 0.0.0.0:443
serve *args:
    cargo run --bin tongues-server "$@"
