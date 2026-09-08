// SPDX-License-Identifier: GPL-3.0-only
//! The wasm-only `wasi:http` / `super-stt:realtime` component. Compiled for
//! `wasm32-wasip2` only (the `wit-bindgen` bindings don't build for the host);
//! the pure helpers it calls live in the crate root and are unit-tested there.

// wit-bindgen's generated `Vec::from_raw_parts` glue trips this pedantic lint.
#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "wit",
    world: "realtime-backend",
    generate_all,
    // The bundled WASI 0.2.0 dep WITs gate a handful of interfaces behind
    // `@unstable` feature flags (e.g. `wasi:clocks/timezone`). wit-bindgen
    // parses every file in `wit/`, so those gates must be enabled for the
    // directory to resolve even though `realtime-backend` does not use them.
    features: [
        "clocks-timezone",
    ],
});

use crate::{
    DEFAULT_BASE_URL, build_multipart, encode_wav, endpoint_url, needs_api_key, parse_base,
    parse_transcript, resolve_model, unreachable_detail, upstream_status_detail,
};

use exports::wasi::http::incoming_handler::Guest;
use wasi::http::types::{
    Fields, IncomingBody, IncomingRequest, Method, OutgoingBody, OutgoingRequest, OutgoingResponse,
    ResponseOutparam, Scheme,
};
use wasi::io::streams::StreamError;

mod realtime;

struct Component;

impl Guest for Component {
    fn handle(request: IncomingRequest, outparam: ResponseOutparam) {
        let (status, body) = route(&request);
        send_response(outparam, status, &body);
    }
}

export!(Component);

/// Dispatch a `/v1` request to its handler, returning `(status, json_bytes)`.
fn route(request: &IncomingRequest) -> (u16, Vec<u8>) {
    let method = request.method();
    let full = request.path_with_query().unwrap_or_default();
    let path = full.split('?').next().unwrap_or("");

    match (&method, path) {
        (Method::Get, "/v1/ping") => ok(&serde_json::json!({
            "status": "success", "message": "pong"
        })),
        (Method::Get, "/v1/status") => ok(&serde_json::json!({
            "status": "success", "state": "ready", "device": "cpu"
        })),
        (Method::Post, "/v1/load") => (
            202,
            to_vec(&serde_json::json!({ "status": "success", "message": "Loading started" })),
        ),
        (Method::Post, "/v1/cancel") => ok(&serde_json::json!({
            "status": "success", "message": "Cancelled"
        })),
        (Method::Post, "/v1/transcribe") => transcribe(request),
        _ => err(404, "not_found"),
    }
}

/// Handle `POST /v1/transcribe`: read the injected secret/option headers and
/// the audio body, forward to OpenAI, and return the transcription.
fn transcribe(request: &IncomingRequest) -> (u16, Vec<u8>) {
    let entries = request.headers().entries();
    // Read the configurable base URL from the daemon-injected header, falling
    // back to the default OpenAI API endpoint. The default carries `/v1` for
    // the same reason a user-set value does: it is an SDK-style base URL.
    let base_url = crate::header(&entries, "x-stt-option-base_url")
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    // The key is optional: a self-hosted endpoint usually wants no
    // `authorization` header at all. Against OpenAI itself a missing key is
    // still worth naming, since the alternative is an opaque 401.
    let api_key =
        crate::header(&entries, "x-stt-secret-openai_api_key").filter(|k| !k.trim().is_empty());
    if api_key.is_none() && needs_api_key(&base_url) {
        // Include a human-readable `detail` that the daemon surfaces to the user.
        return (
            400,
            to_vec(&serde_json::json!({
                "status": "error",
                "message": "missing_secret_openai_api_key",
                "detail": "OpenAI API key not set. Add it in Settings \u{2192} Models \u{2192} OpenAI.",
            })),
        );
    }
    // The manifest's `other` entry carries no name of its own; the model the
    // server actually serves comes from the `custom_model` option.
    let selected =
        crate::header(&entries, "x-stt-model").unwrap_or_else(|| "whisper-1".to_string());
    let custom = crate::header(&entries, "x-stt-option-custom_model");
    let Some(model) = resolve_model(&selected, custom.as_deref()) else {
        return (
            400,
            to_vec(&serde_json::json!({
                "status": "error",
                "message": "missing_option_custom_model",
                "detail": "Custom model name not set. Add it in Settings \u{2192} Models \u{2192} OpenAI, or pick a listed model.",
            })),
        );
    };

    let Ok(body) = request.consume() else {
        return err(400, "no_body");
    };
    let raw = match read_all(body) {
        Ok(r) => r,
        Err(e) => return err(500, &e),
    };
    let req: serde_json::Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => return err(400, "invalid_json"),
    };

    let Some(audio) = req.get("audio_data").and_then(|v| v.as_array()) else {
        return err(400, "invalid_audio");
    };
    let audio: Vec<f32> = audio
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();
    let sample_rate = u32::try_from(
        req.get("sample_rate")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(16000),
    )
    .unwrap_or(16000);
    // OpenAI auto-detects when `language` is omitted, so the reserved `auto`
    // (and a missing field) map to "no language"; a specific code forwards.
    let language = match req.get("language").and_then(|v| v.as_str()) {
        Some("auto") | None => None,
        Some(code) => Some(code),
    };

    match call_openai(
        &base_url,
        api_key.as_deref(),
        &model,
        language,
        &audio,
        sample_rate,
    ) {
        Ok(text) => (
            200,
            to_vec(&serde_json::json!({ "status": "success", "transcription": text })),
        ),
        Err(detail) => (
            502,
            to_vec(&serde_json::json!({
                "status": "error", "message": "upstream_error", "detail": detail
            })),
        ),
    }
}

/// Send the audio to OpenAI's transcription API and return the text.
fn call_openai(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
    language: Option<&str>,
    audio: &[f32],
    sample_rate: u32,
) -> Result<String, String> {
    let wav = encode_wav(audio, sample_rate);
    let boundary = "----superstt7MA4YWxkTrZu0gW";
    let multipart = build_multipart(boundary, model, language, &wav);
    let (https, authority, prefix) = parse_base(base_url);
    let scheme = if https { Scheme::Https } else { Scheme::Http };
    let path = format!("{prefix}/audio/transcriptions");
    // Named in every failure below: the daemon shows `detail` to the user,
    // and this is what points them at the setting to check.
    let endpoint = endpoint_url(https, &authority, &path);
    // A request this backend could not even assemble means the base URL is
    // not something a URL can be built from. Unreachable in practice — the
    // daemon canonicalizes the value before injecting it — so it names the
    // setting rather than trying to diagnose further.
    let malformed = |cause: &str| {
        format!("Could not build a request for {endpoint} ({cause}). Check the API base URL.")
    };

    let headers = Fields::new();
    // No key means no `authorization` header — an empty `Bearer` is worse
    // than none to a server that does not authenticate.
    if let Some(key) = api_key {
        headers
            .append("authorization", format!("Bearer {key}").as_bytes())
            .map_err(|e| malformed(&format!("header: {e:?}")))?;
    }
    headers
        .append(
            "content-type",
            format!("multipart/form-data; boundary={boundary}").as_bytes(),
        )
        .map_err(|e| malformed(&format!("header: {e:?}")))?;

    let request = OutgoingRequest::new(headers);
    request
        .set_method(&Method::Post)
        .map_err(|()| malformed("set_method"))?;
    request
        .set_scheme(Some(&scheme))
        .map_err(|()| malformed("set_scheme"))?;
    request
        .set_authority(Some(&authority))
        .map_err(|()| malformed("set_authority"))?;
    request
        .set_path_with_query(Some(&path))
        .map_err(|()| malformed("set_path"))?;

    // Obtain the body handle, start the request, then stream the body — the
    // canonical wasi:http outbound order.
    //
    // Everything from here to the response is a failure to *reach* the
    // server. The write is the usual place a dead connection surfaces,
    // because `handle` returns before the connection is established and the
    // body is what first tries to use it.
    let unreachable = |cause: &str| unreachable_detail(&endpoint, https, cause);
    let out_body = request.body().map_err(|()| unreachable("request_body"))?;
    let future = wasi::http::outgoing_handler::handle(request, None)
        .map_err(|e| unreachable(&format!("{e:?}")))?;
    write_all(&out_body, &multipart).map_err(|e| unreachable(&e))?;
    OutgoingBody::finish(out_body, None).map_err(|e| unreachable(&format!("{e:?}")))?;

    let pollable = future.subscribe();
    pollable.block();
    let response = future
        .get()
        .ok_or_else(|| unreachable("no_response"))?
        .map_err(|()| unreachable("future_taken"))?
        .map_err(|e| unreachable(&format!("{e:?}")))?;

    let status = response.status();
    let body = response
        .consume()
        .map_err(|()| format!("{endpoint} returned a response that could not be read."))?;
    let bytes = read_all(body)
        .map_err(|e| format!("{endpoint} returned a response that could not be read ({e})."))?;
    if !(200..300).contains(&status) {
        return Err(upstream_status_detail(&endpoint, status, &bytes));
    }

    // OpenAI response: { "text": "…" }
    parse_transcript(&bytes).map_err(|e| {
        format!("{endpoint} replied without a transcript ({e}). It may not be an OpenAI-compatible transcription endpoint.")
    })
}

// ── helpers ─────────────────────────────────────────────────────────────

fn ok(value: &serde_json::Value) -> (u16, Vec<u8>) {
    (200, to_vec(value))
}

fn err(status: u16, message: &str) -> (u16, Vec<u8>) {
    (
        status,
        to_vec(&serde_json::json!({ "status": "error", "message": message })),
    )
}

fn to_vec(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

/// Drain an incoming body to bytes.
fn read_all(body: IncomingBody) -> Result<Vec<u8>, String> {
    let stream = body.stream().map_err(|()| "no_stream".to_string())?;
    let mut out = Vec::new();
    loop {
        match stream.blocking_read(65536) {
            Ok(chunk) => out.extend_from_slice(&chunk),
            Err(StreamError::Closed) => break,
            Err(StreamError::LastOperationFailed(_)) => return Err("read_failed".to_string()),
        }
    }
    drop(stream);
    let _ = IncomingBody::finish(body);
    Ok(out)
}

/// Write all bytes to an outgoing body in ≤4096-byte flushes.
fn write_all(body: &OutgoingBody, data: &[u8]) -> Result<(), String> {
    let stream = body.write().map_err(|()| "write_stream".to_string())?;
    for chunk in data.chunks(4096) {
        stream
            .blocking_write_and_flush(chunk)
            .map_err(|_| "write_failed".to_string())?;
    }
    drop(stream);
    Ok(())
}

/// Build the response and hand it to the outparam.
fn send_response(outparam: ResponseOutparam, status: u16, body_bytes: &[u8]) {
    let headers = Fields::new();
    let _ = headers.append("content-type", b"application/json");
    let response = OutgoingResponse::new(headers);
    let _ = response.set_status_code(status);
    let Ok(body) = response.body() else {
        ResponseOutparam::set(outparam, Ok(response));
        return;
    };
    ResponseOutparam::set(outparam, Ok(response));
    if let Ok(stream) = body.write() {
        for chunk in body_bytes.chunks(4096) {
            let _ = stream.blocking_write_and_flush(chunk);
        }
        drop(stream);
    }
    let _ = OutgoingBody::finish(body, None);
}
