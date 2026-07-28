# Linguistic Claims and Conflict Resolution

Tongues represents competing linguistic interpretations as durable claims
rather than overwriting one analysis with another. The shared contract lives in
`speaking::evidence`, below every producer and consumer:

```text
grammar / lexicon / acoustics / morphology / user markup
                         |
                         v
             LinguisticEvidenceArtifact
                         |
                         v
interpretation / duplex policy / server and CLI inspection
```

`speaking` owns the serializable data model and deterministic resolver. It does
not depend on interpretation, duplex, or server crates, so those crates can use
the same artifact without a dependency cycle.

## Claim shape

Every `LinguisticClaim` has:

- a durable string `LinguisticClaimId`;
- an utterance-scoped `LinguisticTarget`;
- a typed `LinguisticClaimKind` and matching `LinguisticClaimValue`;
- `EvidenceProvenance`, normalized confidence, and a machine/human rationale;
- explicit `supports` and `conflicts_with` claim-ID edges;
- a lifecycle state and source priority.

Targets cover an utterance, text range, token, word, morpheme, phoneme, phone,
boundary, syntax link, pronunciation, or parse. A target may carry a
`TextRange`; those ranges preserve claim identity for an unchanged transcript
prefix and identify only the claims affected by a later repair. Range offsets
count Unicode scalar values.

Claim values cover part of speech, dependency links, lexical identity,
pronunciation, morphology, phoneme and phone realization, reduction, prosodic
role, stress, boundaries, and parses. Constructors for grammar, lexicon,
acoustic, morphology, user-markup, and manual-override producers assign
standard provenance and default priority.

## Lifecycle and revisions

The lifecycle is append-only:

```text
hypothesis -> stable -> committed
     |          |
     +----------+-> revised
     +----------+-> invalidated
```

Each transition records its sequence, reason, and optional replacement claim.
Revised and invalidated claims stay in the artifact for diagnostics but are
never eligible to win. Committed claims are locked: a later revision or
invalidation fails before mutation. A repair beginning at text offset `n`
invalidates eligible, uncommitted claims whose ranges extend beyond `n`, while
claims wholly inside the stable prefix retain their IDs and state.

## Resolution policy

Resolution is deterministic and inspectable. If a committed candidate exists,
it stays selected. Otherwise candidates are ordered by:

1. source priority;
2. normalized confidence;
3. count of active supporting claims;
4. lifecycle stability;
5. lexicographic claim ID.

The default source order is:

```text
manual override
  > manual evidence
  > user markup
  > committed acoustics
  > acoustic model / forced alignment / ASR
  > lexicon
  > grammar
  > morphology
  > prosody
  > punctuation
  > rule
  > imported data
  > G2P
  > learned prediction / inference
  > memory
  > TTS plan
  > unknown
```

Confidence is compared only after source priority. Scores must be finite and in
the inclusive range `[0, 1]`; producers should name a calibration policy when
one exists. Every resolution preserves all candidates and records a
machine-readable reason, a human-readable explanation, and whether the
candidate conflicts with the winner. A manual override therefore wins without
erasing the automatic claim it replaced.

## Artifact and event schema

`LinguisticEvidenceArtifact` uses `schema_version = 1`. JSON readers require
the field and reject unsupported versions with the found and expected values.
There is no pre-v1 durable claim artifact to migrate. Future schema changes
must add an explicit decoder/migration before incrementing
`LINGUISTIC_EVIDENCE_SCHEMA_V1`.

The artifact contains claims, lifecycle history, and saved resolutions. It can
be carried through the existing provider-neutral stream envelope as:

```json
{
  "type": "derived_artifact",
  "data": {
    "stage": "linguistic_claims",
    "artifact_id": "claims:utterance-42",
    "value": {
      "schema_version": 1,
      "utterance_id": "utterance-42",
      "claims": [],
      "lifecycle": [],
      "resolutions": []
    }
  }
}
```

CLI JSONL, server APIs, interpretation, and duplex may serialize or inspect the
same shape. Grammar analyses already project every retained parse, dependency
link, POS assertion, and prosodic-role assertion into this artifact through
`GrammarAnalysis::to_linguistic_evidence`; parse candidates explicitly conflict
and are supported by their component claims.

## Pronunciation resolution

`PhonemicizeOutput` carries the same artifact in `linguistic_evidence`.
Each `LexicalPronunciationCandidates` entry now records:

- the selected stable candidate ID and the complete claim resolution;
- every selected and rejected alternative, including phoneme IDs and lexical
  stress;
- provider provenance, confidence, variety/POS/context/style constraints, and
  the resolver's explanation.

Lexicon, morphology, G2P, rules, user markup, and manual overrides retain their
own provenance. Grammar POS/link claims and nearby lexical-context claims are
support edges where they contribute to a contextual choice. Weak-form
rationales record phrase position, conservative POS, careful/emphasis/citation
intent, and the known following phonetic onset. If the following candidates do
not agree on a vowel/consonant onset, article selection remains conservative
instead of using spelling or candidate zero.

The resolved claim—not vector position—is passed into phoneme planning and
phone realization. Where no contextual evidence separates candidates, the
provider's deterministic ordering is retained as an explicit
`pronunciation.provider_rank_fallback` claim. Hyphenated segment alternatives
are combined and resolved through the same path, with a documented bound of 64
combinations and an explicit warning if that bound is reached.

`PhonemicizeStyle` adds optional emphasized/citation word indices and
pronunciation overrides. These deserialize to empty lists, while the new output
evidence fields have defaults, so existing request and stored-output JSON
remain readable. Non-English varieties without the English weak-form rules
serialize reduction applicability as unknown/not applicable rather than a
false claim.

Interpretation can attach this artifact to Duplex observations. Duplex
hypotheses reference claim and resolution IDs, attribute each claim to one
normalized score component, require declared identity layers to agree before
commit, and invalidate only the revised tail. The policy and delivery boundary
are documented in
[Duplex Linguistic Evidence and Commit Policy](duplex-linguistic-evidence.md).
The versioned, paginated CLI/API/Operate projection is documented in
[Interpretation Evidence Inspection](interpretation-evidence-inspection.md).
