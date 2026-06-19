// SPDX-License-Identifier: GPL-3.0-only
//! OpenAI speech-to-text backend as a `wasi:http` proxy component.
//!
//! Implements the Super STT `/v1` backend contract
//! (`docs/protocol/backend/contract.md`) by exporting
//! `wasi:http/incoming-handler` and dispatching on method + path. The
//! component is stateless: the daemon injects the API key as the
//! `x-stt-secret-openai_api_key` request header and the model in the
//! transcribe body, and the component forwards audio to the OpenAI
//! transcription API over `wasi:http/outgoing-handler`.
//!
//! The `wasi:http` handler compiles only for `wasm32`; the pure helpers
//! (`encode_wav`, `build_multipart`, `parse_base`, `parse_transcript`) build for
//! the host too, so they are unit-tested natively while the component as a whole
//! is exercised by the wasmtime harness in `tests/`.

// Casts are intentional in audio/WAV encoding; doc lint trips on brand names.
#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

// ── pure helpers (host-testable) ────────────────────────────────────────────
// `pub` so the non-test host build that backs the integration harness does not
// flag them as dead code (the wasm handler that calls them is cfg'd out there).

/// Split a base URL into `(is_https, authority)` where authority is
/// `host[:port]`. A bare host (no scheme) is treated as HTTPS.
#[must_use]
pub fn parse_base(base: &str) -> (bool, String) {
    if let Some(rest) = base.strip_prefix("https://") {
        (true, rest.trim_end_matches('/').to_string())
    } else if let Some(rest) = base.strip_prefix("http://") {
        (false, rest.trim_end_matches('/').to_string())
    } else {
        (true, base.trim_end_matches('/').to_string())
    }
}

/// Extract the transcript from an OpenAI transcription response (the `text`
/// field).
///
/// # Errors
/// Returns an error string if the body is not valid JSON or lacks the field.
pub fn parse_transcript(bytes: &[u8]) -> Result<String, String> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .map_err(|e| format!("parse: {e}"))?
        .get("text")
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "no_text_field".to_string())
}

/// Encode f32 samples as a 16-bit PCM mono WAV (mirrors the daemon's
/// `encode_wav_in_memory`).
#[must_use]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample: u32 = 2;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes()); // byte rate
    buf.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Build a `multipart/form-data` body with `model` and `file` (audio.wav).
#[must_use]
pub fn build_multipart(boundary: &str, model: &str, wav: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

// ── wasi:http component (wasm32 only) ───────────────────────────────────────
#[cfg(target_arch = "wasm32")]
mod component {
    use super::{build_multipart, encode_wav, parse_base, parse_transcript};

    use wasi::exports::http::incoming_handler::Guest;
    use wasi::http::types::{
        Fields, IncomingBody, IncomingRequest, Method, OutgoingBody, OutgoingRequest,
        OutgoingResponse, ResponseOutparam, Scheme,
    };
    use wasi::io::streams::StreamError;

    struct Component;

    impl Guest for Component {
        fn handle(request: IncomingRequest, outparam: ResponseOutparam) {
            let (status, body) = route(&request);
            send_response(outparam, status, &body);
        }
    }

    wasi::http::proxy::export!(Component);

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
        let Some(api_key) = header(&entries, "x-stt-secret-openai_api_key") else {
            // Include a human-readable `detail` that the daemon surfaces to the user.
            return (
                400,
                to_vec(&serde_json::json!({
                    "status": "error",
                    "message": "missing_secret_openai_api_key",
                    "detail": "OpenAI API key not set. Add it in Settings \u{2192} Models \u{2192} OpenAI.",
                })),
            );
        };
        let base_url = header(&entries, "x-stt-option-base_url")
            .unwrap_or_else(|| "https://api.openai.com".to_string());
        let model = header(&entries, "x-stt-model").unwrap_or_else(|| "whisper-1".to_string());

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

        match call_openai(&base_url, &api_key, &model, &audio, sample_rate) {
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
        api_key: &str,
        model: &str,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<String, String> {
        let wav = encode_wav(audio, sample_rate);
        let boundary = "----superstt7MA4YWxkTrZu0gW";
        let multipart = build_multipart(boundary, model, &wav);
        let (https, authority) = parse_base(base_url);
        let scheme = if https { Scheme::Https } else { Scheme::Http };

        let headers = Fields::new();
        headers
            .append("authorization", format!("Bearer {api_key}").as_bytes())
            .map_err(|e| format!("header: {e:?}"))?;
        headers
            .append(
                "content-type",
                format!("multipart/form-data; boundary={boundary}").as_bytes(),
            )
            .map_err(|e| format!("header: {e:?}"))?;

        let request = OutgoingRequest::new(headers);
        request
            .set_method(&Method::Post)
            .map_err(|()| "set_method")?;
        request
            .set_scheme(Some(&scheme))
            .map_err(|()| "set_scheme")?;
        request
            .set_authority(Some(&authority))
            .map_err(|()| "set_authority")?;
        request
            .set_path_with_query(Some("/v1/audio/transcriptions"))
            .map_err(|()| "set_path")?;

        // Obtain the body handle, start the request, then stream the body — the
        // canonical wasi:http outbound order.
        let out_body = request.body().map_err(|()| "request_body")?;
        let future = wasi::http::outgoing_handler::handle(request, None)
            .map_err(|e| format!("handle: {e:?}"))?;
        write_all(&out_body, &multipart)?;
        OutgoingBody::finish(out_body, None).map_err(|e| format!("finish: {e:?}"))?;

        let pollable = future.subscribe();
        pollable.block();
        let response = future
            .get()
            .ok_or("no_response")?
            .map_err(|()| "future_taken")?
            .map_err(|e| format!("http: {e:?}"))?;

        let status = response.status();
        let body = response.consume().map_err(|()| "response_consume")?;
        let bytes = read_all(body)?;
        if !(200..300).contains(&status) {
            return Err(format!(
                "status {status}: {}",
                String::from_utf8_lossy(&bytes)
            ));
        }

        // OpenAI response: { "text": "…" }
        parse_transcript(&bytes)
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

    /// Case-insensitive header lookup over `Fields::entries()`.
    fn header(entries: &[(String, Vec<u8>)], want: &str) -> Option<String> {
        entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(want))
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
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
}

#[cfg(test)]
mod tests {
    use super::{build_multipart, encode_wav, parse_base, parse_transcript};

    #[test]
    fn parse_base_splits_scheme_and_authority() {
        assert_eq!(
            parse_base("https://api.openai.com"),
            (true, "api.openai.com".to_string())
        );
        assert_eq!(
            parse_base("http://localhost:8080/"),
            (false, "localhost:8080".to_string())
        );
        // A bare host (no scheme) defaults to HTTPS.
        assert_eq!(
            parse_base("api.openai.com"),
            (true, "api.openai.com".to_string())
        );
    }

    #[test]
    fn parse_transcript_reads_text_field() {
        let bytes = serde_json::to_vec(&serde_json::json!({ "text": "hello world" })).unwrap();
        assert_eq!(parse_transcript(&bytes).unwrap(), "hello world");
    }

    #[test]
    fn parse_transcript_errors_on_missing_field() {
        let bytes = serde_json::to_vec(&serde_json::json!({ "not_text": "x" })).unwrap();
        assert!(parse_transcript(&bytes).is_err());
        assert!(parse_transcript(b"not json").is_err());
    }

    #[test]
    fn encode_wav_writes_a_valid_header() {
        let wav = encode_wav(&[0.0, 1.0, -1.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 3 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 6);
        // Full-scale samples clamp to i16::MAX / -i16::MAX.
        let max = i16::from_le_bytes([wav[46], wav[47]]);
        let min = i16::from_le_bytes([wav[48], wav[49]]);
        assert_eq!(max, i16::MAX);
        assert_eq!(min, -i16::MAX);
    }

    #[test]
    fn build_multipart_frames_model_and_file() {
        let body = build_multipart("BOUNDARY", "whisper-1", b"WAVDATA");
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("--BOUNDARY\r\n"));
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("whisper-1"));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        assert!(text.contains("WAVDATA"));
        // Closing boundary.
        assert!(text.ends_with("--BOUNDARY--\r\n"));
    }
}
