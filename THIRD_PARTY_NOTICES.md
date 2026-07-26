# Third-Party Notices

Tongues is an original speech platform with deliberate compatibility
relationships to third-party research, source projects, checkpoint formats,
model weights, and voice packages. The repository-level MIT license applies to
Tongues-authored code only; it does not replace the licenses of third-party
material.

This file records known relationships that are especially important to the
speech-synthesis implementation. It is not a substitute for the per-dependency
license metadata in `Cargo.lock`, and it is not a legal opinion.

## Coqui TTS

Project: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)

Tongues imports published Coqui checkpoint configurations and weights for
SpeedySpeech, FastPitch, Tacotron2-DDC, Glow-TTS, HiFi-GAN v2,
MultiBand-MelGAN, VITS, and YourTTS, and targets the corresponding tensor names, shapes,
token contracts, defaults, and inference behavior. Coqui reference executions
also produced numerical conformance fixtures.

`crates/tongues-tts/src/burn_tacotron.rs` and
`crates/tongues-tts/src/tacotron_config.rs` adapt the Tacotron 2 inference graph
and configuration contract from Coqui TTS revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, including
`TTS/tts/layers/tacotron/tacotron2.py`,
`TTS/tts/layers/tacotron/attentions.py`,
`TTS/tts/layers/tacotron/common_layers.py`,
`TTS/tts/models/tacotron2.py`, and the Tacotron config modules. Those files are
MPL-2.0 covered modifications rather than MIT relicensing.

`crates/tongues-tts/src/burn_glow_tts.rs` and its configuration adapter adapt
the Glow-TTS inference graph from Coqui TTS revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, including
`TTS/tts/models/glow_tts.py` and `TTS/tts/layers/glow_tts/`. Those files are
MPL-2.0 covered modifications rather than MIT relicensing.

`crates/tongues-tts/src/speaker_encoder.rs` adapts the ResNet speaker encoder,
attentive-statistics pooling, evaluation-crop behavior, and angular
prototypical training semantics from Coqui TTS revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8`, principally
`TTS/speaker_encoder/models/resnet.py` and the speaker-encoder loss modules.
That file is an MPL-2.0 covered modification rather than MIT relicensing.

The Coqui TTS `v0.6.1` source tag is licensed under
[MPL-2.0](https://github.com/coqui-ai/TTS/blob/v0.6.1/LICENSE.txt), not
Apache-2.0. If a Tongues source file is found to contain or adapt Coqui source
expression, that file must retain the applicable MPL notice and be handled
under the MPL's file-level requirements, including distribution of the full
license text.

The license of source code and the license of model weights are separate. At
`v0.6.1`, Coqui's model registry labels the SpeedySpeech and FastPitch entries
`TBD` and leaves the relevant HiFi-GAN v2 and VCTK VITS license fields empty.
The archives inspected by this project contain weights and configuration but no
license file. Tongues therefore records those artifacts as `NOASSERTION`; it
does not infer Apache-2.0 from the source repository license. The
LJSpeech Tacotron2-DDC registry entry separately declares `Apache 2.0`, and the
MultiBand-MelGAN registry entry declares `MPL`; imports must retain the
applicable artifact-specific evidence. The multilingual YourTTS registry entry
declares `CC BY-NC-ND 4.0`; Tongues records that separate non-commercial,
no-derivatives model license in its catalog.

Repository history includes commit `8e3a9c6`, titled
`Import/adapt/reverse engineer some components from coqui`. That title does not
identify which expressions, if any, were translated. The affected VITS modules
remain explicitly marked `audit-required` in
[`docs/provenance.md`](docs/provenance.md) until a file-by-file comparison
records an exact classification and upstream revision.

## Descript MelGAN

Project: [descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips)

Tongues uses the published Linda Johnson checkpoint as the licensed plain
MelGAN conformance artifact and supports its root state-dictionary tensor
layout. The upstream repository and checkpoint are distributed under MIT; the
conformance harness pins revision
`6488045bfba1975602288de07a58570c7b4d66ea` and verifies the checkpoint
checksum before reference inference.

## Piper

Project: [rhasspy/piper](https://github.com/rhasspy/piper)

Tongues runs VITS-family ONNX voices distributed through the Piper voice
ecosystem and reads their accompanying configuration contract. Piper's source
repository is MIT licensed:

> MIT License
>
> Copyright (c) 2022 Michael Hansen
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Each Piper-distributed voice has its own model/data license. The model catalog
records that license and its evidence per voice; the Piper source license does
not automatically apply to voice weights.

## Apache License 2.0

A complete reference copy is included at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) for Apache-2.0 third-party
components and artifacts that actually declare that license. Its presence does
not establish Apache-2.0 for the Coqui model archives discussed above.

## Research Architectures

Tongues implements established architectures including SpeedySpeech, FastPitch,
Tacotron, Tacotron 2, Capacitron, Glow-TTS, HiFi-GAN, MelGAN,
MultiBand-MelGAN, VITS, and StyleTTS2. Architecture names identify neural model
families, not ownership by Tongues and not a claim that Tongues invented the
models. Papers describe ideas and behavior; source licenses govern copied or
adapted source expression; model and dataset licenses govern distributed
artifacts.
