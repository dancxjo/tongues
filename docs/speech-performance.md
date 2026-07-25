# Native Speech Performance

Issue #28 is measured with release builds and resident, repeated inference. A
fresh process reports model/cache startup separately, synthesizes the same
already-planned input twice, and treats the second pass as warm. Audio playback
is excluded.

## 2026-07-25 CPU and CUDA run

Hardware was an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM and driver
595.84. The CPU backend was Burn NdArray. Both paths used seed 27; VITS used
speaker `p225`. Times are synchronized `--timings` measurements.

| Backend | Device | Input | Audio | Cold synthesis | Cold end-to-end RTF | Warm synthesis | First audio | Warm RTF |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SpeedySpeech + HiFi-GAN | CPU | short | 2.531 s | 1.127 s | 1.057 | 1.179 s | 1.179 s | 0.466 |
| VITS | CPU | short | 2.543 s | 11.678 s | 5.388 | 17.238 s | 17.238 s | 6.780 |
| SpeedySpeech + HiFi-GAN | CUDA | short | 2.531 s | 7.774 s | 6.189 | 0.102 s | 0.102 s | **0.040** |
| VITS | CUDA | short | 2.705 s | 22.091 s | 11.272 | 1.069 s | 1.069 s | **0.395** |
| SpeedySpeech + HiFi-GAN | CPU | paragraph | 9.683 s | 4.529 s | 0.628 | 4.738 s | 4.737 s | 0.489 |
| VITS | CPU | paragraph | 8.150 s | 35.979 s | 4.666 | 43.132 s | 43.132 s | 5.292 |
| SpeedySpeech + HiFi-GAN | CUDA | paragraph | 9.683 s | 7.351 s | 1.604 | 0.570 s | 0.569 s | **0.059** |
| VITS | CUDA | paragraph | 8.290 s | 25.197 s | 4.128 | 2.948 s | 2.947 s | **0.356** |

The short input was:

> Morning light rested on the cedar trees.

The paragraph was:

> Morning light rested on the cedar trees while the kettle began to sing. A
> cool breeze moved through the open window, and the old clock marked the quiet
> start of another day.

Both native CUDA paths are below the initial warm RTF target of 1.0. Cold CUDA
time is dominated by model upload plus one-time Burn fusion/kernel compilation;
the resident server avoids paying that cost on each request.

First playable latency currently equals total synthesis latency because
SpeedySpeech emits one waveform and optimized VITS performs one full decoder
launch before slicing host audio into sink chunks. That trade makes the
latency/throughput relationship explicit; reintroducing low-latency VITS
streaming should use cached or non-overlapping decoder state rather than
recomputing overlapping windows.

## Quality gate

Throughput is recorded only after the SpeedySpeech positional table matches the
Coqui float32 layout. A previous multiplication by the frequency divisor was
corrected to division. The always-on unit test checks representative positional
values without model files.

The opt-in published-checkpoint test then validates encoder, duration
expansion, positioned features, mel values, waveform RMS, and waveform samples
against Coqui probes for a multi-word sentence. Run it with the pinned local
artifacts:

```sh
model_home="${MORTAR_SEA_HOME:-$HOME/.local/share/mortar-sea}"
TONGUES_TEST_COQUI_SPEEDY_MODEL="$model_home/models/speech/coqui/en/ljspeech/speedy-speech/model_file.pth" \
TONGUES_TEST_COQUI_SPEEDY_CONFIG="$model_home/models/speech/coqui/en/ljspeech/speedy-speech/config.json" \
TONGUES_TEST_COQUI_HIFIGAN_MODEL="$model_home/models/speech/coqui/en/ljspeech/hifigan-v2/model_file.pth" \
TONGUES_TEST_COQUI_HIFIGAN_CONFIG="$model_home/models/speech/coqui/en/ljspeech/hifigan-v2/config.json" \
cargo test -p tongues-tts \
  burn_speedy_speech::tests::published_checkpoint_stage_parity \
  -- --exact --ignored
```

The 2026-07-25 run passed all stage and waveform probes. VITS retains the
checkpoint-local multi-word token projection and seeded multi-speaker tests,
while the optimized decoder now follows the model's full-sequence inference
shape instead of stitching repeatedly decoded overlapping windows.

## CUDA execution evidence

Nsight Systems traces were captured for two resident passes of each path with:

```sh
nsys profile --trace=cuda,osrt --stats=true \
  --output=target/issue-28-nsys/<backend>-cuda \
  target/release/tongues --verbose speak \
    --backend <backend> --benchmark-runs 2 --seed 27 \
    --output target/issue-28-nsys/output.wav \
    "Morning light rested on the cedar trees."
```

The SpeedySpeech/HiFi-GAN trace recorded 2,488 `cuLaunchKernel` calls. Direct
convolution kernels accounted for most GPU time, with the largest group at
41.0%; transposed convolution and fused elementwise kernels followed. The VITS
trace recorded 4,314 `cuLaunchKernel` calls; its largest direct-convolution
group accounted for 50.7% of GPU time and transposed convolution accounted for
another 11.9%.

The traces also identify the main synchronization barrier:
`cuEventSynchronize` represented 48.3% of traced CUDA API time for the component
pipeline and 77.0% for VITS. Host work is explicit in the library profile as
checkpoint projection, the duration-driven shape decision, audio sink work,
and the final device-to-host waveform copy. Decoder time is backed by the
kernel trace rather than inferred from the `device: CUDA GPU` label.

## Reproducing

Run the checked-in matrix:

```sh
just speech-benchmark
```

Useful controls:

```sh
TONGUES_BENCH_RUNS=4 scripts/speech-benchmark.sh
TONGUES_BENCH_DEVICES=cpu scripts/speech-benchmark.sh
TONGUES_WARM_RTF_TARGET=0.5 scripts/speech-benchmark.sh
```

The JSON result contains every native stage and its token, frame, or sample
dimensions, along with cold/warm classification, first-audio latency, total
synthesis time, audio duration, and RTF.
