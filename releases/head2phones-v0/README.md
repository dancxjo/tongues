# head2phones-v0 Release Bundle

Archive: `head2phones-v0.tar.gz`

The archive was built by following the repository's `models` symlink into the
private generated-artifact store and packaging the dereferenced files from
`models/head2phones/v0`. The `models` path itself remains a tracked symlink, so
the release archive is the versionable artifact.

Archive SHA-256:

```text
f045838ba466f860f5c67106e1d73d0f433edacfced107c06bb293fdb4b60011  head2phones-v0.tar.gz
```

Contents:

- `model.bin`
- `model-epoch-4.bin`
- `vocab.json`
- `head2phones_config.json`
- `manifest.json`
- `model_config.json`
- `train_config.json`
- `train_state.json`
- `SHA256SUMS`

Install locally by extracting the archive into `models/head2phones/v0`:

```sh
just release extract head2phones
```

Verify:

```sh
(cd models/head2phones/v0 && sha256sum -c SHA256SUMS)
```

See `docs/models/head2phones-v0.md` for model state, data mix, usage, and
release caveats.

Regenerate the archive from the current symlinked local artifact:

```sh
just release package head2phones
```
