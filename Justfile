set positional-arguments

default:
    @just --list

# Create a new model-family crate/config/artifact scaffold
new-family family:
    cargo run -q -p xtask -- new-family "{{family}}"

# Run a compact round-trip inference benchmark across G2P2G and Wiktionary models
race *args:
    @cargo run -q -p xtask -- race "$@"

# Generate text chunks, derive phones, and synthesize speech continuously
continue *args:
    @cargo run -q -p xtask -- continue "$@"

# Stream an Ollama story through resident CPU head2phones and Piper playback
be *args:
    @cargo run -q --bin tongues -- --cpu be "$@"

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

# Synthesize speech using StyleTTS2 or Piper backends
speak *args:
    cargo run --bin tongues -- speak "$@"

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
