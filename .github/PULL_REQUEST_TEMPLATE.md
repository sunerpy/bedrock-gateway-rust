<!--
The PR title must be a Conventional Commit with an English, imperative subject
and no scope, for example `fix: preserve image support in Responses translation`.
Pull requests are squash-merged, so the title becomes the commit on main.
-->

## Summary

<!-- What does this change and why? Link related issues, e.g. "Closes #123". -->

## Verification

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features`
- [ ] Golden fixture added under `tests/golden/` (translation or rendering changes)
- [ ] Docs updated where behavior or configuration changed
