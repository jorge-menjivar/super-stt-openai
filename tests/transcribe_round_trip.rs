// SPDX-License-Identifier: GPL-3.0-only
//! End-to-end against a mock OpenAI upstream: the component shapes the request
//! (Bearer auth, `multipart/form-data` body with the model + audio.wav, the
//! endpoint path derived from `base_url`), the host enforces the egress
//! allowlist + SSRF guard, and the `text` field is parsed back out. This is the
//! standalone port of the daemon's `wasm_openai.rs` — daemon and upstream are
//! both mocked, the component is real. The harness stands in for the daemon, so
//! `base_url` values are written in the canonical form the daemon injects:
//! lowercase scheme, no userinfo, no trailing slash, no query.
#![allow(clippy::doc_markdown)]

mod common;

use common::{WasmBackend, component_path};
use wiremock::matchers::{header, method, path};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

const SECRET: &str = "x-stt-secret-openai_api_key";
const BASE_URL: &str = "x-stt-option-base_url";
const CUSTOM_MODEL: &str = "x-stt-option-custom_model";

/// A `wiremock` matcher asserting the request body contains `needle`. Decodes
/// lossily because the multipart body carries binary WAV bytes (invalid UTF-8);
/// the `language` field is plain ASCII and precedes the audio, so it survives.
struct BodyContains(&'static str);
impl Match for BodyContains {
    fn matches(&self, req: &Request) -> bool {
        String::from_utf8_lossy(&req.body).contains(self.0)
    }
}

/// A `wiremock` matcher asserting a header is *absent* — a keyless request must
/// omit `authorization` rather than send an empty one.
struct NoHeader(&'static str);
impl Match for NoHeader {
    fn matches(&self, req: &Request) -> bool {
        !req.headers.contains_key(self.0)
    }
}

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

/// A `base_url` carrying a path prefix — the shape every OpenAI-compatible
/// gateway publishes (`https://api.groq.com/openai/v1`) — puts that prefix in
/// front of the endpoint path instead of a hardcoded `/v1`. The explicit port
/// rides along in the `Host` header, unaltered.
#[tokio::test]
async fn base_url_path_prefix_is_honored() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    let authority = server.address().to_string();
    Mock::given(method("POST"))
        // The prefix from `base_url`, not `/v1`.
        .and(path("/openai/v1/audio/transcriptions"))
        // The authority keeps the port the value named; nothing is synthesized.
        .and(header("host", authority.as_str()))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "prefixed" })),
        )
        .mount(&server)
        .await;

    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (
                BASE_URL.to_string(),
                format!("http://{authority}/openai/v1"),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "prefixed");
}

/// An origin-only `base_url` gets OpenAI's own `/v1`, so a value that predates
/// the path split still resolves to a served endpoint.
#[tokio::test]
async fn base_url_without_path_defaults_to_v1() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "defaulted" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
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

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "defaulted");
}

/// A bracketed IPv6 authority (`http://[::1]:8080/v1`) survives the split: the
/// `:` inside the brackets is not a path separator and the brackets reach the
/// `Host` header intact.
#[tokio::test]
async fn ipv6_authority_round_trips() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };
    let Ok(listener) = std::net::TcpListener::bind("[::1]:0") else {
        eprintln!("skipping: no IPv6 loopback on this host");
        return;
    };

    let server = MockServer::builder().listener(listener).start().await;
    let authority = server.address().to_string();
    assert!(
        authority.starts_with("[::1]:"),
        "expected a bracketed authority, got {authority}"
    );
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("host", authority.as_str()))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "over v6" })),
        )
        .mount(&server)
        .await;

    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "over v6");
}

/// A chosen `language` is forwarded to OpenAI as a `language` multipart field.
#[tokio::test]
async fn language_forwarded_in_multipart() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        // The component adds a `language` form field carrying the chosen code.
        .and(BodyContains("name=\"language\"\r\n\r\nes\r\n"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "hola mundo" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
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
        .transcribe_with_language(&audio, 16000, Some("es"))
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hola mundo");
}

/// Selecting the manifest's `other` model sends the `custom_model` option as the
/// `model` form field, so a server behind `base_url` can serve a model this
/// manifest never lists.
#[tokio::test]
async fn custom_model_name_replaces_the_other_placeholder() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        // The configured name, not the `other` placeholder.
        .and(BodyContains(
            "name=\"model\"\r\n\r\nSystran/faster-whisper-large-v3\r\n",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "custom" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "other".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
            (
                CUSTOM_MODEL.to_string(),
                "Systran/faster-whisper-large-v3".to_string(),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "custom");
}

/// `other` with no `custom_model` set is reported to the user instead of asking
/// the server for a model literally named `other`. Nothing leaves the sandbox.
#[tokio::test]
async fn other_without_a_custom_model_name_is_rejected() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    // Any upstream request at all is a failure here, so mount nothing: wiremock
    // answers an unmatched request with 404, which would surface as an error too,
    // so assert on the request count as well.
    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "other".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let err = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect_err("an unset custom model name must be an error");
    assert!(
        err.to_string().contains("Custom model name not set"),
        "expected a user-facing detail, got: {err}"
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "no upstream request should be attempted"
    );
}

/// A self-hosted endpoint needs no API key: with none set, the request still
/// goes out and carries no `authorization` header at all.
#[tokio::test]
async fn keyless_request_to_a_custom_endpoint_sends_no_authorization() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(NoHeader("authorization"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "keyless" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "other".to_string(),
        vec![
            // No `x-stt-secret-openai_api_key` at all.
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
            (CUSTOM_MODEL.to_string(), "local-whisper".to_string()),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "keyless");
}

/// A key set to blank is the same as no key — no `Bearer` with nothing after it.
#[tokio::test]
async fn blank_api_key_sends_no_authorization() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(NoHeader("authorization"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "blank" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "   ".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "blank");
}

/// A listed model ignores `custom_model`: the option speaks only for `other`.
#[tokio::test]
async fn custom_model_name_is_ignored_for_listed_models() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(BodyContains("name=\"model\"\r\n\r\nwhisper-1\r\n"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "listed" })),
        )
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "whisper-1".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}/v1")),
            (CUSTOM_MODEL.to_string(), "leftover-name".to_string()),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "listed");
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
