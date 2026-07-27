# Streaming language identification and routing

Language identification is provider-neutral in `speaking`. A detector returns
ranked language hypotheses with confidence, evidence duration, and provenance
for an ordered segment. `LanguageRouter` keeps that evidence attached to the
route even when the chosen ASR provider uses a documented fallback.

Users can select a fixed language or detection with an optional candidate set.
Detection defaults require confidence `0.65`, at least `300 ms` of evidence, a
`0.15` margin over the active language, and two consecutive segments before a
switch. Ambiguous evidence retains the active language rather than flapping.
Out-of-order segment routing and mismatched detection sequence numbers fail
explicitly.

An ASR capability names its provider, exact model identity, installation state,
and supported languages. Unsupported languages either fail or use an explicit
configured provider/language fallback. A language identifier is not treated as
evidence that an ASR model is installed.

The same metadata is available from:

```sh
tongues language-routing
curl -s http://127.0.0.1:3000/api/language-routing/capabilities
```

Speech Studio consumes the server representation. It allows fixed-language
selection and exposes the shared hysteresis/minimum-evidence controls.
Auto-detection is disabled with a clear explanation when the advertised
detector model is not installed. Set `TONGUES_WHISPER_MODEL` to an existing
compatible model file before server startup to advertise the Whisper detector
and multilingual ASR route as installed.

Focused fixtures cover fixed routing, stable English-to-Spanish switching,
ambiguous speech without flapping, visible French evidence through an English
fallback, and rejection of reordered segments.
