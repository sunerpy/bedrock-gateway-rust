//! `cargo test --test golden` entry point.
//!
//! The real harness lives in [`golden`] (`tests/golden/harness.rs`). It is
//! pulled in via an explicit `#[path]` so the harness file can sit inside the
//! `tests/golden/` directory next to its `fixtures/` and `README.md` without
//! colliding with this top-level target file.
//!
//! This suite runs fully OFFLINE — it has no AWS dependencies and performs no
//! network access. See `tests/golden/README.md` for the fixture format.

#[path = "golden/harness.rs"]
mod golden;
