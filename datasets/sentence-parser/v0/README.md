# Sentence boundary dataset

Dataset: `v0`

Sources: 0 Project Gutenberg-style text files
Detected sentences: 0
Training rows: 0
Naive-discrepancy correction rows: 0

Input shape:

```text
<task:sentence_boundary><ctx:previous><previous sentence><ctx:cursor><cursor prefix>
```

Targets:

```text
<boundary:emit><sentence>\n
<boundary:continue>
<boundary:missing_head><tail fragment>
<boundary:repair><repaired sentence>
```

The source intentionally includes only the previously parsed sentence and current cursor prefix. No following sentence is provided.
