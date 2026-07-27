# Dependency Reproducibility And Audit Policy

`Cargo.lock` is the reviewed definition of the Rust dependency graph. Use the
repository toolchain and `--locked` for validation and release builds:

```sh
cargo check --workspace --all-targets --locked
cargo test --workspace --no-fail-fast --locked
cargo build --release -p tongues-cli --locked
cargo audit
cargo deny --locked check
```

CI rejects a missing or stale lockfile. Dependabot proposes grouped Cargo and
GitHub Actions updates; its pull requests run the same CI matrix as other pull
requests. The dependency-audit workflow checks RustSec on pull requests, on
`main`, and weekly, and also enforces the reviewed license/source policy in
`deny.toml`.

## Exceptions

Fix or update a vulnerable or policy-violating dependency first. An exception
is allowed only when no immediate safe update exists and the affected behavior
is understood. A pull request adding an exception must update both the relevant
configuration (`.cargo/audit.toml` and `deny.toml` for RustSec) and this table.
The owner must review or remove it no later than the listed date.

| ID/package | Owner | Rationale and exposure | Added | Review or expiry |
|---|---|---|---|---|
| _No active exceptions_ | — | — | — | — |

Expired or undocumented exceptions fail review even if the tools still accept
their syntax. License clarifications must additionally pin the exact package
version and checksum of the files used as evidence.

## What Cargo Covers

Cargo and the lockfile cover Rust packages from crates.io, their checksums,
features, and source identities. `cargo audit` compares that graph with RustSec.
`cargo deny` rejects unknown registries and Git sources, yanked/vulnerable
packages, and licenses outside the reviewed allowlist. Wildcard path
requirements and duplicate versions remain warnings because the workspace uses
local path members and the current audio/model stacks contain parallel major
versions; security-sensitive or materially large cases should be promoted to
targeted bans.

RustSec informational notices such as unmaintained or unsound transitive crates
remain visible in audit output but do not have the same failure status as a
reported vulnerability. They must not be added to an ignore list merely to
quiet the report.

The reviewed license set includes the repository's permissive Rust ecosystem
licenses plus the already-documented BSD-4-Clause CMUdict packages,
CC-BY-SA-4.0 OpenEPD data crate, MPL compatibility code, and bzip2 library.
This is permission for those known dependency uses, not a claim that their
terms become MIT or that model/data redistribution obligations disappear.

## What Cargo Does Not Cover

The dependency inventory does not attest to these external layers:

- Ubuntu CI installs `libasound2-dev`; Linux audio builds therefore depend on
  the system ALSA library and its distribution security updates.
- Native crates can invoke the host C/C++ compiler and linker. Their bundled or
  system libraries require their own advisory and license review; RustSec
  coverage of the Rust wrapper is not coverage of that native code.
- ONNX Runtime, CUDA drivers/toolkits, audio devices, and platform libraries
  are operator-provided runtime dependencies when the corresponding features
  are used.
- Downloaded lexicons, datasets, model weights, voice packages, and generated
  artifacts are data assets, not Cargo packages. Their URLs, checksums,
  provenance, and terms are governed by the asset manifests,
  `docs/licensing.md`, and `docs/provenance.md`.

Release-tag and manual runs of `release-inventory.yml` upload `Cargo.lock`,
`deny.toml`, and `cargo metadata --locked` JSON. The JSON records every Cargo
package's version, source, checksum, declared license, and dependency edges;
it is the machine-readable Cargo dependency/license inventory for the release.
