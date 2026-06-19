// SPDX-License-Identifier: GPL-3.0-only
//! End-to-end against a mock OpenAI upstream: the component shapes the request
//! (Bearer auth, `multipart/form-data` body with the model + audio.wav, fixed
//! path), the host enforces the egress allowlist + SSRF guard, and the `text`
//! field is parsed back out. This is the standalone port of the daemon's
//! `wasm_openai.rs` — daemon and upstream are both mocked, the component is real.
#![allow(clippy::doc_markdown)]

mod common;

use common::{WasmBackend, component_path};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SECRET: &str = "x-stt-secret-openai_api_key";
const BASE_URL: &str = "x-stt-option-base_url";

/// Happy path: Bearer auth + `multipart/form-data` body reach the allowlisted
/// upstream at `/v1/audio/transcriptions`, and the response `text` comes back as
/// the transcription.
#[tokio::test]
async fn transcribe_round_trip() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        // OpenAI uses `Bearer` auth.
        .and(header("authorization", "Bearer test-key"))
        // The component frames the audio as multipart with a fixed boundary.
        .and(header(
            "content-type",
            "multipart/form-data; boundary=----superstt7MA4YWxkTrZu0gW",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "hello world" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    // The mock upstream is on loopback (wiremock binds 127.0.0.1); the SSRF guard
    // blocks loopback for untrusted backends, so opt in for the test.
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let audio = vec![0.0_f32; 1600];
    let text = backend
        .transcribe_audio(&audio, 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hello world");
}

/// The allowlist blocks egress to a host the configuration does not permit, even
/// though a server is listening there.
#[tokio::test]
async fn allowlist_blocks_disallowed_host() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "nope" })),
        )
        .mount(&server)
        .await;

    let mut backend = WasmBackend::new(
        &component,
        // Allowlist a different host than the mock is listening on.
        vec!["api.openai.com".to_string()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), server.uri()),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&[0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "outbound call to a non-allowlisted host must be blocked"
    );
}

/// SSRF guard: an allowlisted *hostname* that resolves to loopback is blocked,
/// even though the host string is on the allowlist.
#[tokio::test]
async fn ssrf_blocks_hostname_resolving_to_loopback() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "nope" })),
        )
        .mount(&server)
        .await;

    let port = server.address().port();
    let mut backend = WasmBackend::new(
        &component,
        // `localhost` is allowlisted by name, but resolves to 127.0.0.1 / ::1.
        vec!["localhost".to_string()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://localhost:{port}")),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&[0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "a hostname resolving to loopback must be blocked by the SSRF guard"
    );
}
