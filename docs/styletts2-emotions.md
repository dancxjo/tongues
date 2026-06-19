# StyleTTS2 Emotion Vectors

Tongues can derive StyleTTS2 emotion controls from labeled reference WAV files.
The workflow has three artifacts:

- `labels.jsonl`: maps each WAV to an emotion and speaker.
- `style_vectors.jsonl`: one 256-value StyleTTS2 style vector per reference WAV.
- `emotion_signatures.json`: one 256-value emotion delta per emotion, suitable for the web UI and `speak`.

## Fetch Audio

Fetch the public RAVDESS speech audio and generate labels:

```sh
just run fetch-corpora --out-dir datasets/emotions
```

This creates:

```text
datasets/emotions/RAVDESS/
datasets/emotions/labels.jsonl
```

`labels.jsonl` contains canonical absolute WAV paths. Keep it paired with the extracted audio directory that produced it.

## Encode Style Vectors

Style vectors come from StyleTTS2's style encoder. Encode the fetched WAV files:

```sh
just run styletts2 encode-style \
  datasets/emotions/RAVDESS \
  --labels datasets/emotions/labels.jsonl \
  --out datasets/emotions/style_vectors.jsonl
```

Each line in `style_vectors.jsonl` contains:

```json
{
  "id": 0,
  "path": "/absolute/path/to/reference.wav",
  "emotion": "happy",
  "speaker": "ravdess_02",
  "vector": [0.0, 0.0]
}
```

The real `vector` has 256 finite `f32` values.

## Build Emotion Signatures

Convert per-WAV style vectors into speaker-normalized emotion deltas:

```sh
just run styletts2 emotion-signatures \
  datasets/emotions/style_vectors.jsonl \
  --out emotion_signatures.json
```

The default method is `speaker-neutral-delta`:

1. Group vectors by speaker and emotion.
2. Compute each speaker's neutral mean vector.
3. Compute each non-neutral emotion mean vector.
4. Store `emotion_mean - neutral_mean` per speaker.
5. Average those deltas across speakers.

The output is keyed by emotion:

```json
{
  "happy": {
    "kind": "styletts2.emotion_signature.v1",
    "emotion": "happy",
    "method": "speaker-neutral-delta",
    "dims": 256,
    "vector": [0.0, 0.0],
    "stats": {
      "n_speakers": 24
    },
    "recommended_strength": {
      "subtle": 0.25,
      "normal": 0.65,
      "strong": 1.1
    }
  }
}
```

## Use With Synthesis

From the CLI:

```sh
just run speak \
  --emotion-signatures emotion_signatures.json \
  --emotion happy \
  --emotion-strength 0.65 \
  --output happy.wav \
  "That actually worked."
```

`--emotion-strength` scales the emotion delta before adding it to the base StyleTTS2 reference style vector.

## Web UI And Server

Start the server:

```sh
just serve
```

The server exposes:

| Endpoint | Purpose |
|---|---|
| `GET /api/emotions` | Returns emotion signature metadata and the full 256-value vectors. |
| `POST /api/speak` | Accepts text plus an optional emotion name, emotion vector, and strength; returns `audio/wav`. |

`GET /api/emotions` looks for `emotion_signatures.json` in the workspace root. If it is missing, the server can build it from either:

```text
style_vectors.jsonl
datasets/emotions/style_vectors.jsonl
```

The frontend stores the returned vectors in memory. When you synthesize with an emotion selected, it posts:

```json
{
  "text": "That actually worked.",
  "emotion": "happy",
  "emotion_vector": [0.0, 0.0],
  "emotion_strength": 0.65
}
```

The server validates that posted vectors have 256 finite values, writes a temporary emotion signature file, invokes `tongues speak`, then removes the temporary file after synthesis.

## Common Issues

If the frontend shows no emotions, make sure at least one of these exists:

```text
emotion_signatures.json
datasets/emotions/style_vectors.jsonl
style_vectors.jsonl
```

If `encode-style` reports zero encoded files, regenerate `labels.jsonl` and style vectors from the same audio directory. Label paths are matched by canonical absolute path.

If synthesis says an emotion is missing, inspect the signature keys:

```sh
jq 'keys' emotion_signatures.json
```
