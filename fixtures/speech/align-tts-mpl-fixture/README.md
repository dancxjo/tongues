# Align-TTS MPL conformance fixture

This directory contains a deterministic, tiny Align-TTS checkpoint produced by
the unmodified Coqui TTS implementation at revision
`0cf3265a4686d7e856bd472cdaf1572d61cab2b8`. It is a valid upstream checkpoint
layout with random seed `210021`, not a pretrained voice and not a speech
quality sample.

The checkpoint, generated config, and numerical reference are distributed
under MPL-2.0; `LICENSE.txt` contains the full license. The small artifact is
committed so CPU CI can prove safe package import and Python-free native
inference. `reference.json` records exact duration/alignment shapes and sparse
mel probes; native mel tolerance is `3e-4`.

Regenerate it with the pinned speech-conformance image:

```sh
docker run --rm \
  --entrypoint python \
  --volume "$PWD:/workspace" \
  tongues-coqui-reference:latest \
  scripts/align-tts-fixture.py \
  --out /workspace/fixtures/speech/align-tts-mpl-fixture
```
