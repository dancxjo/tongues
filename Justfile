set positional-arguments

default:
    @just --list

# Prepare CMUdict data splits and build vocabulary (runs prepare)
prepare *args:
    cargo run --bin pronlex -- prepare --input data/cmudict.dict --out runs/cmudict-v0 "$@"

# Fetch/Download the CMUdict lexicon data file
fetch *args:
    cargo run --bin pronlex -- fetch-cmudict --out data/cmudict.dict "$@"

# Synthesize speech using StyleTTS2 or Piper backends
speak *args:
    cargo run --bin pronlex -- speak "$@"

# Phonemize text into an IPA sequence
phonemes *args:
    cargo run --bin pronlex -- phonemes "$@"

# Print narrow phonetic phones transcription
phones *args:
    cargo run --bin pronlex -- phones "$@"

# Run translation prediction (graphemes to phonemes or vice-versa)
infer *args:
    cargo run --bin pronlex -- predict "$@"

# Train the pronlex translation model with an even mix of both directions
train *args:
    cargo run --bin pronlex -- train --data runs/cmudict-v0 --out models/cmudict-v0 --task both "$@"

# Refine the model on validation/test pronunciation discrepancies
refine *args:
    cargo run --bin pronlex -- refine --model models/cmudict-v0 --data runs/cmudict-v0 --out models/cmudict-v0-refined --verbose "$@"

# Fine-tune the model on the built-in Dolch sight-word list
sight-words *args:
    cargo run --bin pronlex -- refine --model models/cmudict-v0 --data runs/cmudict-v0 --out models/cmudict-v0-sight-words --source sight-words --task both --verbose "$@"
