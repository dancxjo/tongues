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

# Train the pronlex masked-phone predictor model
train *args:
    cargo run --bin pronlex -- train --data runs/cmudict-v0 --out models/cmudict-v0 "$@"
