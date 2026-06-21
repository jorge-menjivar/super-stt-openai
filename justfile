# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone OpenAI WASM backend. Mirrors the recipe names
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

# Measure code coverage of the host-compiled code (requires cargo-llvm-cov). The
# component runs in wasmtime (wasm32) and isn't host-instrumentable, so this
# covers the pure helpers in src/; test code under tests/ is excluded, and
# --remap-path-prefix keeps report paths relative (src/lib.rs, not the absolute
# build path). The harness loads the prebuilt component, so build it first.
# Usage: just coverage [--html]
coverage *args: build-component
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov: build-component
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# Full local CI gate: format, lint (component + host), build, test
ci: fmt-check check check-host build-component test
