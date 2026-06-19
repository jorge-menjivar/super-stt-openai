// SPDX-License-Identifier: GPL-3.0-only
//! CI-runnable contract smoke test — no upstream needed. Loads the real
//! component and drives the no-egress `/v1` routes the daemon hits first.
#![allow(clippy::doc_markdown)]

mod common;

use common::{WasmBackend, component_path};

/// `GET /v1/ping` and `GET /v1/status` against the live component. A cloud
/// backend reports `ready` immediately (no weights to load).
#[tokio::test]
async fn ping_and_status() {
    let Some(path) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let backend = WasmBackend::new(&path, Vec::new(), "whisper-1".to_string(), Vec::new())
        .expect("load backend");

    let ping = backend.ping().await.expect("ping");
    assert_eq!(ping["status"], "success");
    assert_eq!(ping["message"], "pong");

    let status = backend.status().await.expect("status");
    assert_eq!(status["status"], "success");
    assert_eq!(status["state"], "ready");
}

/// `POST /v1/transcribe` with no API key returns the structured
/// `missing_secret_*` error (the component never reaches the network).
#[tokio::test]
async fn transcribe_without_key_reports_missing_secret() {
    let Some(path) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let mut backend = WasmBackend::new(&path, Vec::new(), "whisper-1".to_string(), Vec::new())
        .expect("load backend");

    let err = backend
        .transcribe_audio(&[0.0_f32; 100], 16000)
        .await
        .expect_err("must fail without an API key");
    assert!(
        err.to_string().contains("OpenAI API key not set"),
        "expected the missing-secret detail, got: {err}"
    );
}
