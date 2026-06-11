set positional-arguments

default:
    @just --list

# Prepare CMUdict data splits and build vocabulary (runs prepare)
prepare *args:
    cargo run -- prepare --input data/cmudict.dict --out runs/cmudict-v0 "$@"

# Fetch/Download the CMUdict lexicon data file
fetch *args:
    cargo run -- fetch-cmudict --out data/cmudict.dict "$@"

# Synthesize speech using StyleTTS2 or Piper backends
speak *args:
    cargo run -- speak "$@"

# Phonemize text into an IPA sequence
phonemes *args:
    cargo run -- phonemes "$@"

# Print narrow phonetic phones transcription
phones *args:
    cargo run -- phones "$@"

# Train the pronlex masked-phone predictor model
train *args:
    cargo run -- train --data runs/cmudict-v0 --out models/cmudict-v0 "$@"
