# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Deepgram WASM backend. Mirrors the recipe names
# used by the main super-stt repo (`just check`, `just ci`, etc.).

# Default: build the component
default: build-component

# Build the wasi:http component (wasm32-wasip2). Requires the target:
#   rustup target add wasm32-wasip2
build-component *args:
    cargo build --release --locked --target wasm32-wasip2 {{ args }}

# Lint the shipped component on its real target (wasm32-wasip2). The host build
# only sees the pure helpers (the wasi handler is cfg'd out), so the component's
# own code is linted here. Gating: `-D warnings`.
check *args:
    cargo clippy --target wasm32-wasip2 --lib --release {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Lint the host-side test harness + unit tests (default/host target).
check-host *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run the test suite. The wasmtime harness loads the prebuilt component, so build
# it first. Usage: just test [--verbose]
test *args: build-component
    cargo test --locked {{ args }}

# Full local CI gate: format, lint (component + host), build, test
ci: fmt-check check check-host build-component test
