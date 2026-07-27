# Speech Studio dataflow editor

`/speech-dataflow.html` is the execution graph workspace linked from Speech
Studio. It discovers source kinds, cleanup stages, language detectors, ASR
providers/models, response providers, TTS compositions, and CLI capability IDs
from the running backend. Provider and model names are never duplicated in the
browser source.

Five starter templates cover transcription, multilingual transcription,
meeting transcripts, spoken interpretation, and full conversation. Nodes can
be added, removed, reordered with buttons or arrow keys, duplicated, bypassed,
or replaced by a compatible stage. Typed port validation explains mismatches
before execution. Saved schema-v1 graphs retain capability IDs and resolve the
current registry again when opened; unsupported versions produce a migration
error.

The deterministic execution preview uses the server fixture provider and
visually separates provisional/revised events from committed recognition.
WaveDeck remains the sibling evidence editor over the shared timeline session,
so graph changes and human corrections never become one mutable authority.
