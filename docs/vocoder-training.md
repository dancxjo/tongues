# Native vocoder training and fine-tuning

Tongues provides Python-free Burn training runners for HiFi-GAN, MelGAN, and
MultiBand-MelGAN. They reuse the same generators and configuration contracts as
native inference, including the normal SpeedySpeech, FastPitch, and Glow-TTS
composition paths.

The smallest complete acceptance run is:

```sh
just vocoder-fixture
```

The fixture trains all three families on compact deterministic data. For each
family it interrupts after the first batch, restores the exact epoch and batch
cursor plus both optimizer records, finishes training, verifies that the
enabled reconstruction objective improves, writes finite WAV samples and JSON
metrics, exports SafeTensors, and reloads the result through `BurnVocoder`.

## Prepared data and initialization

Use a prepared speech corpus with `train.jsonl`, `valid.jsonl`, `test.jsonl`,
and a `vocoder-features/` cache containing the referenced mel/audio artifacts.
Initialize a run with a versioned recipe and the matching native model config:

```sh
cargo run --release --bin tongues -- vocoder initialize \
  --kind hifigan \
  --data target/ljspeech-prepared \
  --recipe recipes/hifigan.json \
  --config configs/hifigan.json \
  --out models/vocoders/my-hifigan
```

For fine-tuning, add `--source-checkpoint PATH`, `--source-license LICENSE`,
and `--source-provenance TEXT`. Initialization records the source checkpoint
SHA-256, config, dataset license, and dataset provenance in
`run-manifest.json`. A source checkpoint is restored with the safe native
loader before optimization starts.

`--kind` accepts `hifigan`, `melgan`, and `multiband-melgan`. A recipe records
the architecture-specific generator and discriminator configuration, shared
mel contract, deterministic batching seed, adversarial update schedule,
optimizer and scheduler settings, gradient clipping, checkpoint/sample
intervals, and all loss weights. An enabled mel objective uses the
differentiable native STFT/mel path; enabled losses never silently become zero.

## Train, resume, evaluate, and export

```sh
cargo run --release --bin tongues -- vocoder train \
  --run models/vocoders/my-hifigan

cargo run --release --bin tongues -- vocoder resume \
  --run models/vocoders/my-hifigan

cargo run --release --bin tongues -- vocoder evaluate \
  --run models/vocoders/my-hifigan --split test

cargo run --release --bin tongues -- vocoder export \
  --run models/vocoders/my-hifigan
```

CPU is the default; select CUDA with the CLI's global device option. Training
prints the durable paths before the first update:

- `train_state.json`, including exact epoch/batch cursor and learning rates;
- `trainer-latest.bin`;
- `optim-generator-latest.bin` and `optim-discriminator-latest.bin`;
- `model-latest.safetensors` and best `model.safetensors`; and
- `samples/validation-step-N.wav`.

Model and optimizer records are written through `.part` files before the
training state is atomically replaced, so the state never advertises an
uncommitted step. Evaluation reports the enabled reconstruction objective,
spectral and waveform L1 diagnostics, finite-audio status, real-time factor,
generated sample count, and parameter-memory estimate.

`export` validates the best checkpoint by loading it with the normal native
vocoder adapter. The resulting checkpoint and config can therefore be used by
the existing acoustic-model composition paths without a training-only loader.
