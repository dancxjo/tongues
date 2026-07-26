# Native VITS training and fine-tuning

Tongues provides a staged, Python-free Burn path for the first supported VITS
training slice. It reuses the native inference modules and the shared speech
corpus/audio contracts; it is not a port of Coqui Trainer.

## Supported milestone

The supported graph consists of:

- the existing native text prior, stochastic-duration prior, residual coupling
  flow, and integrated waveform decoder;
- a training-only spectrogram posterior encoder;
- hard maximum-path monotonic alignment;
- deterministic segment selection and waveform slicing;
- multi-period and multi-scale discriminators;
- adversarial, feature-matching, mel, duration, and KL loss terms; and
- learned speaker IDs, language IDs, and externally supplied reference
  d-vectors when the imported checkpoint declares them.

Recipes reject a future schema version and require stochastic durations plus
maximum-path alignment. The first milestone deliberately does not promise every
Coqui VITS option, the complete Coqui Trainer API, or production-quality voice
cloning.

## Ordered prerequisites

The epic was decomposed along reusable boundaries before this VITS slice:

1. [#15](https://github.com/dancxjo/tongues/issues/15) supplies normalized
   manifests, deterministic splits, length-aware batches, cached features, and
   VCTK/LJSpeech formatters.
2. [#17](https://github.com/dancxjo/tongues/issues/17) supplies native
   WAV/resampling/STFT/mel preprocessing with a serialized audio contract.
3. [#16](https://github.com/dancxjo/tongues/issues/16) supplies safe,
   versioned Coqui checkpoint import and package metadata.
4. [#37](https://github.com/dancxjo/tongues/issues/37) supplies the
   model-neutral Burn training/evaluation/export hook boundary.
5. Issue #9 supplies the architecture-specific VITS posterior, alignment,
   segmentation, losses, checkpoint export, and durable run contract described
   here.

## Data contract

Prepare raw LJSpeech, VCTK, or generic metadata with the shared pipeline:

```sh
cargo run --release --bin tongues -- speech-corpus prepare \
  --input /data/VCTK-Corpus-0.92 \
  --out target/vctk-prepared \
  --format vctk \
  --language en-GB \
  --split-by-speaker \
  --seed 42
```

`VitsDatasetManifest` then identifies the normalized manifest, train/validation/
test splits, feature cache, sample rate, channel count, split seed, record
counts, data license, and data provenance. Training requires mono audio and
non-empty train and validation splits. Token IDs remain checkpoint-local;
speaker and language IDs are metadata, not linguistic variety identifiers.

## Recipe and fine-tuning

`VitsTrainingRecipe` is versioned, serializable data. It records:

- seed, epochs, batch size, and segment frames;
- CPU or CUDA policy;
- generator/discriminator Adam settings;
- epoch learning-rate decay;
- step/epoch checkpoint and evaluation-sample intervals;
- loss weights; and
- independently frozen text, posterior, duration, flow, decoder, speaker, and
  language parameter groups.

`VitsInferenceExport::load_coqui_checkpoint` restores the inference-side
weights of the pinned VCTK checkpoint directly through the safe Rust loader.
The training-only posterior and discriminators are initialized separately.
Callers may freeze any restored group for a short targeted fine-tune. Record
the baseline and best value of the chosen validation metric in
`VitsTrainingManifest`; a fine-tune is successful only when the recorded best
value improves on the baseline.

The native training graph returns every loss component independently. The
outer trainer should compute differentiable target/generated mel tensors with
the shared audio geometry, call `combine_vits_generator_losses`, and update
generator and discriminator parameter groups separately.

## Checkpoints and resume

Before training, call `initialize_vits_run_with_progress`. It creates and
reports these paths:

| Path | Purpose |
|---|---|
| `recipe.json` | Exact versioned optimizer, scheduler, freeze, and loss recipe |
| `training-manifest.json` | Model/data source, checksums, licenses, provenance, and metric |
| `README.md` | Reproducible model card and compute requirements |
| `train_state.json` | Epoch, global step, batch cursor, shuffle seed, learning rates, best metric, and optimizer paths |
| `model-epoch-N.safetensors` | Completed epoch checkpoint |
| `model-latest.safetensors` | Most recent recoverable checkpoint |
| `model.safetensors` | Best inference checkpoint |
| `optim-generator-latest.bin` | Generator optimizer state |
| `optim-discriminator-latest.bin` | Discriminator optimizer state |
| `samples/` | Periodic validation synthesis |

JSON and documentation are written to `.part`, flushed, synced, and renamed.
Model and optimizer writers must follow the same rule. Advance
`train_state.json` only after both model and optimizer files are durable.
Resume uses the stored batch cursor and shuffle seed, not a newly shuffled
epoch. If a trainer checkpoints only at epoch end, its startup output must say
so explicitly.

`VitsTrainingProgress` exposes initialization, writes, resume position, epoch,
batch/global-step counts, checkpoints, samples, and completion. CLI renderers
should include the active output path and report the first few batches and then
bounded intervals.

## Inference compatibility

`VitsInferenceExport::save_inference_safetensors` excludes posterior and
discriminator tensors, converts Burn layouts to the checkpoint layout, and
writes the root names used by the existing inference adapter. The resulting
SafeTensors file is passed directly to `BurnVitsSpeech::load` with the same
VITS config and speaker/language maps; there is no conversion command or
Python runtime between training and inference.

The CPU fixture tests cover:

- monotonic, complete maximum-path alignment;
- finite posterior sampling, KL, and all named generator losses;
- deterministic latent/waveform segment slicing;
- a tiny posterior fixture whose training loss decreases without non-finite
  values;
- non-empty finite waveform decoding; and
- direct reload of a training export through `BurnVitsSpeech`.

Run them with:

```sh
cargo test -p tongues-tts --lib \
  burn_vits_training::tests --no-default-features
cargo test -p tongues-tts --lib \
  vits_recipe::tests --no-default-features
```

Published-checkpoint fine-tuning remains opt-in because the licensed model and
corpus are large. A recorded run must pin the checkpoint SHA-256, retain its
license evidence, state the target metric and baseline, write periodic finite
WAV samples, and verify the best exported checkpoint with
`BurnVitsSpeech::load`.

## Compute guidance

CPU is retained for fixture overfit, correctness checks, recovery, and very
small debugging batches. It is not considered time-feasible for a full VITS
run. CUDA is recommended for fine-tuning and required for practical
from-scratch training. Device choice does not change the recipe, dataset
identity, checkpoint layout, or evaluation metric.
