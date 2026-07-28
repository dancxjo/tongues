# Duplex Linguistic Evidence and Commit Policy

`tongues-duplex` treats linguistic analysis as evidence for ranking and repair,
not as authority to invent observed speech. This document defines the boundary
between provider hypotheses, linguistic claims, the commit frontier, and audio
that may already have reached a listener.

## Evidence and scoring

Each completion hypothesis may reference linguistic claim and resolution IDs
from a `LinguisticEvidenceArtifact`. Its normalized score keeps these components
separate:

- acoustic likelihood;
- provider prior;
- lexical evidence;
- grammar parse rank;
- prosody compatibility;
- user markup;
- direct observation.

A claim is attributed to at most one component according to its evidence
source, so the same claim cannot increase multiple terms. Missing components
remain unavailable instead of becoming zero-valued contrary evidence.
Provider priors are normalized across the beam; component weights and the
resulting combined score are finite, bounded, deterministic, and serializable.

Grammar, lexical context, and prosody can rerank uncommitted homophones,
heteronyms, pronunciations, and realization plans. They cannot become direct
support merely by agreeing with a prediction. A branch with overwhelming
acoustic contradiction is capped even when its parse is otherwise tidy.

## Selection and commitment

Selection, commitment, and verification are distinct:

1. Reranking selects a posterior-mass set while retaining every competitor.
2. The set's longest common morpheme prefix becomes a candidate frontier.
3. Every newly committed morpheme must reference direct text or acoustic
   evidence for the same normalized key.
4. If a proposal declares morpheme, word, or pronunciation identity evidence,
   each declared layer must contain a committed claim or a saved resolution
   whose active winner agrees.
5. Low score margin or provider disagreement causes abstention.
6. Closed-loop verification occurs only after commitment and records a separate
   status.

The direct-support check is repeated while replaying a
`CommitFrontierAdvanced` event. A journal therefore cannot smuggle predicted
content across the authority boundary even if the event was produced elsewhere.

## Transcript replacement and stable identity

`revise_evidence` replaces an observation without changing its evidence ID. It
compares normalized morphemes, retains occurrence IDs for the stable prefix, and
invalidates eligible uncommitted claims that overlap the revised text range.
Affected hypotheses are repaired, withdrawn, or reranked; their prior audit
entries remain available.

Committed morphemes and committed linguistic claims are append-only history.
A late contradiction does not silently rewrite them. It instead crosses the
delivery policy below.

## Synthesis and playback repair

Each emitted synthesis unit records a delivery state:

```text
planned -> prepared -> held -> played -> verified
                 \-> invalidated
```

When revised evidence affects planned, prepared, or held audio, Duplex
invalidates that unit and emits `ReplaceHeldAudio`. When it affects audio
already played, Duplex preserves the played record and emits
`DeliverPostPlaybackCorrection`. The correction is explicit journal state,
not a mutation of what the listener already heard.

## Journal and diagnostics

The simulator journal records:

- observed and revised evidence;
- created, revised, and invalidated linguistic claims;
- proposed, repaired, and withdrawn hypotheses;
- deterministic reranking and retained competitors;
- commit decisions and machine-readable block reasons;
- commit-frontier advances and verification;
- synthesis delivery transitions and repair policy decisions.

`SimulatorState::rankings`, `commit_diagnostics`, `hypothesis_audit`,
`deliveries`, and `repair_delivery` are bounded inspection surfaces for tests,
operator tools, and future telemetry. A failure can therefore identify whether
the cause was acoustic contradiction, unresolved identity, missing direct
support, provider disagreement, or a low score margin without treating a
prediction as observed fact.
