// SPDX-License-Identifier: GPL-3.0-only
//! OpenAI speech-to-text backend targeting the `super-stt:realtime` world.
//!
//! The component exports both halves of that world:
//!
//! * `wasi:http/incoming-handler` — the Super STT `/v1` batch contract
//!   (`docs/protocol/backend/contract.md`), dispatching on method + path. This
//!   path is stateless: the daemon injects the API key as the
//!   `x-stt-secret-openai_api_key` request header (absent when the user set
//!   none) and the model in the transcribe body, and the component forwards
//!   audio to the OpenAI transcription API — or to whatever OpenAI-compatible
//!   endpoint the `base_url` option names — over
//!   `wasi:http/outgoing-handler`.
//! * `super-stt:realtime/ws-server` — the realtime WebSocket session handler.
//!   It bridges a consumer WebSocket to OpenAI's realtime transcription API
//!   over the daemon-implemented `super-stt:realtime/ws` import.
//!
//! The `wit-bindgen` / `wasi:http` handler is **wasm-only**, so it lives behind
//! `#[cfg(target_arch = "wasm32")]` in [`mod@component`]. The pure audio,
//! request-shaping, and realtime-payload helpers stay host-compiled here and
//! are unit-tested natively (a pure-wasm crate could not test them).

// Casts are intentional in audio/WAV encoding and resampling; doc lint trips on
// brand names.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use base64::Engine as _;

#[cfg(target_arch = "wasm32")]
mod component;

/// Case-insensitive header lookup over a `(name, value)` list. Both transports
/// read the daemon's injected `x-stt-*` context this way — the batch path off
/// `Fields::entries()`, the realtime path off the `ws-server.handle` argument.
#[must_use]
pub fn header(entries: &[(String, Vec<u8>)], want: &str) -> Option<String> {
    entries
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(want))
        .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
}

// ── pure helpers (host-testable) ────────────────────────────────────────────
// `pub` so the non-test host build that backs the integration harness does not
// flag them as dead code (the wasm handler that calls them is cfg'd out there).

/// Split a base URL into `(is_https, authority, path_prefix)`, where authority
/// is `host[:port]` and the prefix is the path the endpoint paths hang off.
///
/// `base_url` means what it means in every OpenAI SDK, so it carries the API
/// version: `https://api.groq.com/openai/v1` yields the prefix `/openai/v1` and
/// a request to `/openai/v1/audio/transcriptions`. An origin-only value gets the
/// `/v1` prefix OpenAI itself serves, so a value without a path still resolves.
///
/// The daemon canonicalizes the value before injecting it (lowercase scheme, no
/// userinfo, no trailing slash, no query or fragment), so splitting the
/// authority at the first `/` is the whole of the work. A bare host (no scheme)
/// cannot reach production but is treated as HTTPS rather than folded into the
/// authority.
#[must_use]
pub fn parse_base(base: &str) -> (bool, String, String) {
    let (https, rest) = if let Some(rest) = base.strip_prefix("https://") {
        (true, rest)
    } else if let Some(rest) = base.strip_prefix("http://") {
        (false, rest)
    } else {
        (true, base)
    };
    let rest = rest.trim_end_matches('/');
    let (authority, prefix) = match rest.split_once('/') {
        Some((authority, path)) => (authority, format!("/{path}")),
        None => (rest, "/v1".to_string()),
    };
    (https, authority.to_string(), prefix)
}

/// Endpoint used when the user overrides nothing. SDK-style, so it carries the
/// API version; its host must stay in step with [`OPENAI_HOST`].
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Host of OpenAI's own API — the one endpoint known to require a key.
pub const OPENAI_HOST: &str = "api.openai.com";

/// Whether a request to `base_url` is worth refusing without an API key.
///
/// OpenAI's own API always needs one, and a keyless request there earns a 401
/// the user has to decode; naming the missing setting is more useful. Any other
/// endpoint may authenticate however it likes — a local server typically not at
/// all — so the component sends what it has and lets the server decide.
#[must_use]
pub fn needs_api_key(base_url: &str) -> bool {
    let (_, authority, _) = parse_base(base_url);
    // Port-stripping is deliberately naive: it only has to be right for a match
    // against a known hostname, and a mangled IPv6 literal simply won't match.
    let host = authority.split(':').next().unwrap_or_default();
    host.eq_ignore_ascii_case(OPENAI_HOST)
}

/// Wire name of the manifest's placeholder model entry: the user is running a
/// model this manifest does not list, on whatever server `base_url` points at.
pub const CUSTOM_MODEL: &str = "other";

/// The realtime counterpart of [`CUSTOM_MODEL`] — the same placeholder on the
/// WebSocket transport. A manifest entry is identified by its name, so the two
/// transports need two entries; both read the same `custom_model` option.
pub const CUSTOM_MODEL_REALTIME: &str = "other-realtime";

/// Whether `name` is a placeholder standing in for the `custom_model` option
/// rather than a name a server serves.
#[must_use]
pub fn is_custom_placeholder(name: &str) -> bool {
    name == CUSTOM_MODEL || name == CUSTOM_MODEL_REALTIME
}

/// Resolve the model name to send upstream.
///
/// A listed model passes through unchanged. [`CUSTOM_MODEL`] and
/// [`CUSTOM_MODEL_REALTIME`] resolve to the `custom_model` option instead, so an
/// OpenAI-compatible server can serve a model this manifest never enumerates.
/// Returns `None` when a placeholder is selected and no custom name is set — the
/// caller turns that into a user-facing error, because `other` is not a name any
/// server serves.
#[must_use]
pub fn resolve_model(selected: &str, custom: Option<&str>) -> Option<String> {
    if !is_custom_placeholder(selected) {
        return Some(selected.to_string());
    }
    custom
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(String::from)
}

/// The URL a request was aimed at, as the user would recognize it.
///
/// Every failure below names it. The daemon shows a backend's `detail` to the
/// user, and the endpoint is the one thing that turns "it did not work" into a
/// setting they can go and check.
#[must_use]
pub fn endpoint_url(https: bool, authority: &str, path: &str) -> String {
    let scheme = if https { "https" } else { "http" };
    format!("{scheme}://{authority}{path}")
}

/// Detail for a request that never got a reply: the connection failed, or died
/// while the body was going out.
///
/// Over `https` this names the mismatch that produces it most often. A base URL
/// naming no scheme is read as `https` unless its host is visibly local, so a
/// plaintext server behind a hostname is reached over TLS and the connection
/// dies — wherever the write happens to notice it, which is why the underlying
/// cause can read as a write problem rather than a connection one.
#[must_use]
pub fn unreachable_detail(endpoint: &str, https: bool, cause: &str) -> String {
    let hint = if https {
        " Check the server is running and reachable on that port, and that it accepts https — a plaintext server reached over https fails exactly this way."
    } else {
        " Check the server is running and reachable on that port."
    };
    format!("Could not reach {endpoint} ({cause}).{hint}")
}

/// Detail for a reply that arrived carrying a non-2xx status.
///
/// The upstream's own body is the useful part — an OpenAI-compatible server
/// explains a bad model name or a rejected key there — so it is passed through,
/// bounded, rather than replaced.
#[must_use]
pub fn upstream_status_detail(endpoint: &str, status: u16, body: &[u8]) -> String {
    const MAX: usize = 300;
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.is_empty() {
        return format!("{endpoint} returned HTTP {status}.");
    }
    let mut shown: String = text.chars().take(MAX).collect();
    if shown.chars().count() < text.chars().count() {
        shown.push('…');
    }
    format!("{endpoint} returned HTTP {status}: {shown}")
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

/// Build a `multipart/form-data` body with `model`, an optional `language`, and
/// `file` (audio.wav).
#[must_use]
pub fn build_multipart(boundary: &str, model: &str, language: Option<&str>, wav: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");
    if let Some(lang) = language {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        body.extend_from_slice(lang.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
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

// ── realtime helpers (pure; host-compiled + unit-tested) ────────────────────

/// The only sample rate OpenAI's realtime API accepts for `audio/pcm`: 24 kHz,
/// mono, 16-bit little-endian. The consumer declares its own rate in the `start`
/// frame (16 kHz, typically), so the session resamples on the way upstream —
/// see [`Resampler`].
pub const REALTIME_SAMPLE_RATE: u32 = 24000;

/// OpenAI's realtime transcription model, and the model a session assumes when
/// the daemon injects no `x-stt-model`. It is also the one model that takes a
/// `languages` list where the others take a singular `language` — sending both
/// spellings is an error, so [`session_update_json`] picks by name.
pub const LIVE_TRANSCRIBE_MODEL: &str = "gpt-live-transcribe";

/// Least audio OpenAI will commit as a turn — 100 ms, which at
/// [`REALTIME_SAMPLE_RATE`] and 2 bytes per sample is 4800 bytes. A shorter
/// recording is answered with an empty transcript rather than relaying the
/// upstream's `input_audio_buffer_commit_empty`, which says nothing to a user
/// who simply tapped the key.
pub const MIN_COMMIT_BYTES: usize = 4800;

/// The realtime WebSocket endpoint for a given API base URL.
///
/// `base_url` is SDK-style and carries the API version, exactly as the batch
/// path reads it (see [`parse_base`]), so `https://api.openai.com/v1` yields
/// `wss://api.openai.com/v1/realtime?intent=transcription`. The model is not in
/// the URL — it is named in the `session.update` the session sends next.
#[must_use]
pub fn realtime_ws_url(base_url: &str) -> String {
    let (https, authority, prefix) = parse_base(base_url);
    let scheme = if https { "wss" } else { "ws" };
    format!("{scheme}://{authority}{prefix}/realtime?intent=transcription")
}

/// Detail for a realtime session whose upstream WebSocket never opened.
///
/// The mirror of [`unreachable_detail`] for the WebSocket transport: it names
/// the endpoint, and over `wss` names the scheme mismatch that produces this
/// most often, since a base URL without an explicit scheme is read as secure.
#[must_use]
pub fn ws_unreachable_detail(endpoint: &str, secure: bool, cause: &str) -> String {
    let hint = if secure {
        " Check the server is running and reachable on that port, and that it accepts wss — a plaintext server reached over wss fails exactly this way."
    } else {
        " Check the server is running and reachable on that port."
    };
    format!("Could not reach {endpoint} ({cause}).{hint}")
}

/// The `session.update` that configures a transcription-only realtime session.
///
/// Turn detection is off: the consumer decides when its turn ends (a `stop`
/// frame or a close), and the session commits the buffer once at that point.
#[must_use]
pub fn session_update_json(model: &str, language: Option<&str>) -> String {
    let mut transcription = serde_json::json!({ "model": model });
    if let Some(code) = language {
        if model == LIVE_TRANSCRIBE_MODEL {
            transcription["languages"] = serde_json::json!([code]);
        } else {
            transcription["language"] = serde_json::Value::String(code.to_string());
        }
    }
    serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": { "type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE },
                    "transcription": transcription,
                    "turn_detection": serde_json::Value::Null,
                },
            },
        },
    })
    .to_string()
}

/// `input_audio_buffer.append` payload carrying base64-standard PCM (s16le mono
/// at [`REALTIME_SAMPLE_RATE`]).
#[must_use]
pub fn audio_append_json(pcm: &[u8]) -> String {
    let audio = base64::engine::general_purpose::STANDARD.encode(pcm);
    serde_json::json!({ "type": "input_audio_buffer.append", "audio": audio }).to_string()
}

/// Parse the consumer's `start` frame: `{"type":"start","sample_rate":N,
/// "language":"xx"}`. Requires `type == "start"`; `sample_rate` defaults to
/// 16000; `language` is optional, and the reserved `auto` means "no language"
/// so the model detects it, matching the batch path.
///
/// # Errors
/// Returns an error string when the JSON is invalid or `type != "start"`.
pub fn parse_start(s: &str) -> Result<(u32, Option<String>), String> {
    let v: serde_json::Value =
        serde_json::from_str(s).map_err(|_| "invalid start frame".to_string())?;
    if v.get("type").and_then(serde_json::Value::as_str) != Some("start") {
        return Err("invalid start frame".to_string());
    }
    let sample_rate = v
        .get("sample_rate")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .unwrap_or(16000);
    let language = match v.get("language").and_then(serde_json::Value::as_str) {
        Some("auto") | None => None,
        Some(code) => Some(code.to_string()),
    };
    Ok((sample_rate, language))
}

/// `true` if `s` is a JSON object with `type == "stop"`.
#[must_use]
pub fn is_stop(s: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(s)
        .ok()
        .and_then(|v| {
            v.get("type")
                .and_then(serde_json::Value::as_str)
                .map(|t| t == "stop")
        })
        .unwrap_or(false)
}

/// Consumer `preview` frame (incremental transcript).
#[must_use]
pub fn preview_json(text: &str) -> String {
    serde_json::json!({ "type": "preview", "text": text }).to_string()
}

/// Consumer `done` frame (final transcript).
#[must_use]
pub fn done_json(text: &str) -> String {
    serde_json::json!({ "type": "done", "transcription": text }).to_string()
}

/// Consumer `error` frame.
#[must_use]
pub fn error_json(msg: &str) -> String {
    serde_json::json!({ "type": "error", "message": msg }).to_string()
}

/// Linear-interpolation resampler for mono s16le PCM.
///
/// The consumer streams audio at whatever rate its `start` frame declared, and
/// OpenAI's realtime API accepts only [`REALTIME_SAMPLE_RATE`], so every chunk
/// is resampled on the way upstream. The fractional read position and the last
/// sample of the previous chunk carry across calls, so a chunk boundary
/// interpolates against its real neighbour instead of restarting — no click
/// every 100 ms, and no drift over a long session.
pub struct Resampler {
    /// Input samples consumed per output sample.
    step: f64,
    /// Where the next output sample reads from, relative to the start of the
    /// next chunk. `-1.0` addresses [`Self::prev`]; always `> -1.0`.
    pos: f64,
    /// Last sample of the previous chunk.
    prev: i16,
}

impl Resampler {
    /// A resampler from `from` Hz to `to` Hz. A zero rate (which the `start`
    /// frame cannot produce) is treated as pass-through rather than a panic.
    #[must_use]
    pub fn new(from: u32, to: u32) -> Self {
        let step = if from == 0 || to == 0 {
            1.0
        } else {
            f64::from(from) / f64::from(to)
        };
        Self {
            step,
            pos: 0.0,
            prev: 0,
        }
    }

    /// Resample one chunk of s16le PCM. A trailing odd byte is dropped — the
    /// consumer frames whole samples, so there is never one to keep.
    pub fn process(&mut self, pcm: &[u8]) -> Vec<u8> {
        let count = pcm.len() / 2;
        if count == 0 {
            return Vec::new();
        }
        // A negative index reads the previous chunk's last sample; an index past
        // the end clamps, which only happens with a zero fraction (weight 0).
        let prev = self.prev;
        let at = |index: isize| -> f64 {
            let Ok(index) = usize::try_from(index) else {
                return f64::from(prev);
            };
            let index = index.min(count - 1);
            f64::from(i16::from_le_bytes([pcm[index * 2], pcm[index * 2 + 1]]))
        };

        let last = (count - 1) as f64;
        let mut out = Vec::new();
        while self.pos <= last {
            let base = self.pos.floor();
            let frac = self.pos - base;
            let index = base as isize;
            let value = at(index) + (at(index + 1) - at(index)) * frac;
            out.extend_from_slice(&(value.round() as i16).to_le_bytes());
            self.pos += self.step;
        }

        self.prev = i16::from_le_bytes([pcm[(count - 1) * 2], pcm[(count - 1) * 2 + 1]]);
        // Rebase onto the next chunk. The loop ran past `last`, so this stays
        // above -1.0 and the next chunk's `prev` remains the right neighbour.
        self.pos -= count as f64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_BASE_URL, LIVE_TRANSCRIBE_MODEL, REALTIME_SAMPLE_RATE, Resampler,
        audio_append_json, build_multipart, done_json, encode_wav, endpoint_url, error_json,
        is_stop, needs_api_key, parse_base, parse_start, parse_transcript, preview_json,
        realtime_ws_url, resolve_model, session_update_json, unreachable_detail,
        upstream_status_detail, ws_unreachable_detail,
    };
    use base64::Engine as _;

    /// The canonical values the daemon injects: scheme present, no trailing
    /// slash, path preserved verbatim.
    #[test]
    fn parse_base_splits_scheme_authority_and_path() {
        assert_eq!(
            parse_base("https://api.openai.com/v1"),
            (true, "api.openai.com".to_string(), "/v1".to_string())
        );
        // A multi-segment prefix stays whole — the authority stops at the first
        // `/`, which is what `set_authority` will accept.
        assert_eq!(
            parse_base("https://api.groq.com/openai/v1"),
            (true, "api.groq.com".to_string(), "/openai/v1".to_string())
        );
        assert_eq!(
            parse_base("http://localhost:4000/v1"),
            (false, "localhost:4000".to_string(), "/v1".to_string())
        );
        // A bracketed IPv6 authority survives: no `/` inside the brackets.
        assert_eq!(
            parse_base("http://[::1]:8080/v1"),
            (false, "[::1]:8080".to_string(), "/v1".to_string())
        );
    }

    /// An origin-only value gets OpenAI's own `/v1`, so every value that worked
    /// before this split keeps working.
    #[test]
    fn parse_base_defaults_an_empty_path_to_v1() {
        assert_eq!(
            parse_base("https://api.openai.com"),
            (true, "api.openai.com".to_string(), "/v1".to_string())
        );
        assert_eq!(
            parse_base("http://localhost:8080"),
            (false, "localhost:8080".to_string(), "/v1".to_string())
        );
    }

    /// Defensive branches for values the daemon's canonicalization rules out.
    #[test]
    fn parse_base_tolerates_non_canonical_values() {
        // A bare host (no scheme) defaults to HTTPS rather than becoming part of
        // the authority.
        assert_eq!(
            parse_base("api.openai.com"),
            (true, "api.openai.com".to_string(), "/v1".to_string())
        );
        // A trailing slash is not a path prefix.
        assert_eq!(
            parse_base("http://localhost:8080/"),
            (false, "localhost:8080".to_string(), "/v1".to_string())
        );
        assert_eq!(
            parse_base("https://gateway.example.com/v1/"),
            (true, "gateway.example.com".to_string(), "/v1".to_string())
        );
    }

    /// OpenAI's own API is the one endpoint the component refuses to call
    /// keyless, so the missing setting is named instead of a 401 being relayed.
    #[test]
    fn needs_api_key_only_for_openais_own_api() {
        assert!(needs_api_key(DEFAULT_BASE_URL));
        assert!(needs_api_key("https://api.openai.com"));
        // Canonicalization lowercases the scheme, not the host.
        assert!(needs_api_key("https://API.OpenAI.com/v1"));
        assert!(needs_api_key("https://api.openai.com:443/v1"));
    }

    /// Anywhere else authenticates on its own terms — commonly not at all.
    #[test]
    fn needs_api_key_is_false_for_other_endpoints() {
        assert!(!needs_api_key("http://localhost:8000/v1"));
        assert!(!needs_api_key("http://[::1]:8080/v1"));
        assert!(!needs_api_key("https://api.groq.com/openai/v1"));
        // A lookalike host is a different host.
        assert!(!needs_api_key("https://api.openai.com.evil.test/v1"));
        assert!(!needs_api_key(
            "https://proxy.example.com/api.openai.com/v1"
        ));
    }

    /// The default endpoint is the host `needs_api_key` recognizes; a rename of
    /// one without the other would silently drop the keyless refusal.
    #[test]
    fn default_base_url_targets_the_known_openai_host() {
        let (https, authority, prefix) = parse_base(DEFAULT_BASE_URL);
        assert!(https);
        assert_eq!(authority, super::OPENAI_HOST);
        assert_eq!(prefix, "/v1");
    }

    /// A listed model is sent as-is, whether or not a custom name is configured
    /// — the option only speaks for the `other` entry.
    #[test]
    fn resolve_model_passes_listed_models_through() {
        assert_eq!(resolve_model("whisper-1", None).unwrap(), "whisper-1");
        assert_eq!(
            resolve_model("gpt-4o-transcribe", Some("ignored")).unwrap(),
            "gpt-4o-transcribe"
        );
    }

    /// `other` is a placeholder, so it resolves to the configured name.
    #[test]
    fn resolve_model_substitutes_the_custom_name() {
        assert_eq!(
            resolve_model("other", Some("Systran/faster-whisper-large-v3")).unwrap(),
            "Systran/faster-whisper-large-v3"
        );
        // Surrounding whitespace from a pasted value is not part of the name.
        assert_eq!(
            resolve_model("other", Some("  my-model \n")).unwrap(),
            "my-model"
        );
    }

    /// `other` with nothing configured has no name to send — the caller reports
    /// that rather than asking the server for a model called `other`.
    #[test]
    fn resolve_model_rejects_an_unset_custom_name() {
        assert!(resolve_model("other", None).is_none());
        assert!(resolve_model("other", Some("")).is_none());
        assert!(resolve_model("other", Some("   ")).is_none());
    }

    /// The endpoint is what makes a failure actionable, so it is rebuilt the
    /// way the request was addressed rather than echoing the raw option.
    #[test]
    fn endpoint_url_reads_back_as_the_request_was_addressed() {
        assert_eq!(
            endpoint_url(false, "192.168.0.179:8080", "/v1/audio/transcriptions"),
            "http://192.168.0.179:8080/v1/audio/transcriptions"
        );
        assert_eq!(
            endpoint_url(true, "api.openai.com", "/v1/audio/transcriptions"),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    /// The failure a user actually hit: a plaintext server reached over https.
    /// The cause `write_failed` says nothing on its own, so the detail names
    /// the endpoint and the mismatch that produces it.
    #[test]
    fn unreachable_detail_names_the_endpoint_and_the_scheme_trap() {
        let over_tls = unreachable_detail(
            "https://192.168.0.179:8080/v1/audio/transcriptions",
            true,
            "write_failed",
        );
        assert!(over_tls.contains("192.168.0.179:8080"), "{over_tls}");
        assert!(over_tls.contains("write_failed"), "{over_tls}");
        assert!(over_tls.contains("plaintext server"), "{over_tls}");

        // Over http there is no scheme mismatch to suggest, so it is not
        // suggested — a wrong guess sends the user down the wrong path.
        let plain = unreachable_detail(
            "http://192.168.0.179:8080/v1/audio/transcriptions",
            false,
            "write_failed",
        );
        assert!(plain.contains("192.168.0.179:8080"), "{plain}");
        assert!(!plain.contains("plaintext server"), "{plain}");
    }

    #[test]
    fn upstream_status_detail_passes_the_server_explanation_through() {
        let d = upstream_status_detail(
            "http://gw.local/v1/audio/transcriptions",
            404,
            br#"{"error":"model whisper-tiny not found"}"#,
        );
        assert!(d.contains("404"), "{d}");
        assert!(d.contains("model whisper-tiny not found"), "{d}");

        // An empty body still yields a sentence, not a dangling colon.
        let empty = upstream_status_detail("http://gw.local/v1/audio/transcriptions", 502, b"   ");
        assert!(empty.ends_with("returned HTTP 502."), "{empty}");

        // A server that answers an API call with a whole HTML page does not get
        // to fill the notification.
        let long = upstream_status_detail("http://gw.local/x", 500, &vec![b'x'; 5000]);
        assert!(long.chars().count() < 400, "{}", long.chars().count());
        assert!(long.ends_with('…'), "{long}");
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
        let body = build_multipart("BOUNDARY", "whisper-1", None, b"WAVDATA");
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("--BOUNDARY\r\n"));
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("whisper-1"));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        assert!(text.contains("WAVDATA"));
        // No language part when none is given.
        assert!(!text.contains("name=\"language\""));
        // Closing boundary.
        assert!(text.ends_with("--BOUNDARY--\r\n"));
    }

    #[test]
    fn build_multipart_includes_language_when_present() {
        let body = build_multipart("BOUNDARY", "whisper-1", Some("es"), b"WAVDATA");
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"language\""));
        assert!(text.contains("\r\n\r\nes\r\n"));
    }

    // ── realtime helpers ────────────────────────────────────────────────────

    /// The realtime endpoint hangs off the same SDK-style base URL the batch
    /// path uses, so a gateway's API version prefix survives.
    #[test]
    fn realtime_ws_url_maps_the_scheme_and_keeps_the_api_version() {
        assert_eq!(
            realtime_ws_url(DEFAULT_BASE_URL),
            "wss://api.openai.com/v1/realtime?intent=transcription"
        );
        assert_eq!(
            realtime_ws_url("http://localhost:8000/v1"),
            "ws://localhost:8000/v1/realtime?intent=transcription"
        );
        assert_eq!(
            realtime_ws_url("https://api.groq.com/openai/v1"),
            "wss://api.groq.com/openai/v1/realtime?intent=transcription"
        );
        // An origin-only value gets the `/v1` OpenAI itself serves.
        assert_eq!(
            realtime_ws_url("https://gateway.example.com"),
            "wss://gateway.example.com/v1/realtime?intent=transcription"
        );
    }

    /// The WebSocket mirror of `unreachable_detail`: the endpoint always, and
    /// the scheme trap only where it can apply.
    #[test]
    fn ws_unreachable_detail_names_the_endpoint_and_the_scheme_trap() {
        let secure = ws_unreachable_detail("wss://gw.local/v1/realtime", true, "connect failed");
        assert!(secure.contains("wss://gw.local/v1/realtime"), "{secure}");
        assert!(secure.contains("connect failed"), "{secure}");
        assert!(secure.contains("plaintext server"), "{secure}");

        let plain = ws_unreachable_detail("ws://gw.local/v1/realtime", false, "connect failed");
        assert!(plain.contains("ws://gw.local/v1/realtime"), "{plain}");
        assert!(!plain.contains("plaintext server"), "{plain}");
    }

    #[test]
    fn parse_start_reads_the_rate_and_language() {
        assert_eq!(
            parse_start(r#"{"type":"start","sample_rate":24000,"language":"fr"}"#).unwrap(),
            (24000, Some("fr".to_string()))
        );
        // Defaults the rate to 16000; language optional.
        assert_eq!(parse_start(r#"{"type":"start"}"#).unwrap(), (16000, None));
        // `auto` means "let the model detect it", as on the batch path.
        assert_eq!(
            parse_start(r#"{"type":"start","language":"auto"}"#).unwrap(),
            (16000, None)
        );
        // A zero rate would divide by zero downstream; it is not a rate.
        assert_eq!(
            parse_start(r#"{"type":"start","sample_rate":0}"#).unwrap(),
            (16000, None)
        );
        assert!(parse_start(r#"{"type":"stop"}"#).is_err());
        assert!(parse_start("not json").is_err());
    }

    #[test]
    fn is_stop_detects_stop_frames() {
        assert!(is_stop(r#"{"type":"stop"}"#));
        assert!(!is_stop(r#"{"type":"start"}"#));
        assert!(!is_stop("garbage"));
    }

    /// The session is transcription-only, at the one rate the API accepts, with
    /// turn detection off so the commit is ours to send.
    #[test]
    fn session_update_json_configures_a_transcription_session() {
        let v: serde_json::Value =
            serde_json::from_str(&session_update_json("whisper-1", Some("es"))).unwrap();
        assert_eq!(v["type"], "session.update");
        assert_eq!(v["session"]["type"], "transcription");
        let input = &v["session"]["audio"]["input"];
        assert_eq!(input["format"]["type"], "audio/pcm");
        assert_eq!(input["format"]["rate"], REALTIME_SAMPLE_RATE);
        assert_eq!(input["transcription"]["model"], "whisper-1");
        assert_eq!(input["transcription"]["language"], "es");
        assert!(input["turn_detection"].is_null());
    }

    /// One model spells the language as a list. Sending both spellings is an
    /// upstream error, so only one may ever appear.
    #[test]
    fn session_update_json_uses_the_plural_field_for_live_transcribe() {
        let v: serde_json::Value =
            serde_json::from_str(&session_update_json(LIVE_TRANSCRIBE_MODEL, Some("en"))).unwrap();
        let transcription = &v["session"]["audio"]["input"]["transcription"];
        assert_eq!(transcription["languages"][0], "en");
        assert!(transcription["language"].is_null());

        // No language at all: neither field appears, and the model detects it.
        let v: serde_json::Value =
            serde_json::from_str(&session_update_json(LIVE_TRANSCRIBE_MODEL, None)).unwrap();
        let transcription = &v["session"]["audio"]["input"]["transcription"];
        assert!(transcription["languages"].is_null());
        assert!(transcription["language"].is_null());
    }

    #[test]
    fn audio_append_json_carries_base64_pcm() {
        let v: serde_json::Value = serde_json::from_str(&audio_append_json(&[1, 2, 3, 4])).unwrap();
        assert_eq!(v["type"], "input_audio_buffer.append");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(v["audio"].as_str().unwrap())
            .unwrap();
        assert_eq!(decoded, vec![1, 2, 3, 4]);
    }

    #[test]
    fn consumer_frames_carry_the_documented_shapes() {
        let preview: serde_json::Value = serde_json::from_str(&preview_json("partial")).unwrap();
        assert_eq!(preview["type"], "preview");
        assert_eq!(preview["text"], "partial");

        let done: serde_json::Value = serde_json::from_str(&done_json("final")).unwrap();
        assert_eq!(done["type"], "done");
        assert_eq!(done["transcription"], "final");

        let error: serde_json::Value = serde_json::from_str(&error_json("nope")).unwrap();
        assert_eq!(error["type"], "error");
        assert_eq!(error["message"], "nope");
    }

    /// Build `count` s16le samples from a function of the sample index.
    fn pcm(count: usize, f: impl Fn(usize) -> i16) -> Vec<u8> {
        (0..count).flat_map(|i| f(i).to_le_bytes()).collect()
    }

    /// Read s16le bytes back as samples.
    fn samples(bytes: &[u8]) -> Vec<i16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(i16::from_le_bytes)
            .collect()
    }

    /// 16 kHz in, 24 kHz out: half again as many samples, chunk after chunk.
    /// The count is allowed to drift by a sample per boundary — a resampled
    /// stream cannot land on exact chunk multiples.
    #[test]
    fn resampler_upsamples_16k_to_24k() {
        let mut resampler = Resampler::new(16000, REALTIME_SAMPLE_RATE);
        let chunk = pcm(1600, |_| 0);
        let mut total = 0;
        for _ in 0..4 {
            total += samples(&resampler.process(&chunk)).len();
        }
        let expected = 4 * 1600 * 3 / 2;
        assert!(
            total.abs_diff(expected) <= 4,
            "expected ~{expected} samples, got {total}"
        );
    }

    /// Matching rates are a pass-through: same samples, same order.
    #[test]
    fn resampler_passes_matching_rates_through() {
        let mut resampler = Resampler::new(REALTIME_SAMPLE_RATE, REALTIME_SAMPLE_RATE);
        let input = pcm(8, |i| i16::try_from(i).unwrap() * 100);
        assert_eq!(samples(&resampler.process(&input)), samples(&input));
        // And again on the next chunk — the carried position does not drift.
        assert_eq!(samples(&resampler.process(&input)), samples(&input));
    }

    /// A chunk boundary interpolates against the previous chunk's last sample,
    /// so a rising ramp keeps rising instead of dipping back toward zero.
    #[test]
    fn resampler_is_continuous_across_chunks() {
        let mut resampler = Resampler::new(16000, REALTIME_SAMPLE_RATE);
        let first = pcm(64, |i| i16::try_from(i).unwrap() * 100);
        let second = pcm(64, |i| i16::try_from(i + 64).unwrap() * 100);
        let mut out = samples(&resampler.process(&first));
        out.extend(samples(&resampler.process(&second)));
        assert!(
            out.windows(2).all(|w| w[1] >= w[0]),
            "resampled ramp should be non-decreasing: {out:?}"
        );
        // And it spans the whole input range rather than restarting.
        assert_eq!(*out.first().unwrap(), 0);
        assert!(*out.last().unwrap() >= 12_600, "{:?}", out.last());
    }

    /// The realtime placeholder reads the same `custom_model` option as the
    /// batch one, so one setting serves both transports.
    #[test]
    fn resolve_model_substitutes_the_custom_name_for_the_realtime_placeholder() {
        assert_eq!(
            resolve_model("other-realtime", Some("my-local-whisper")).unwrap(),
            "my-local-whisper"
        );
        assert!(resolve_model("other-realtime", None).is_none());
        assert!(resolve_model("other-realtime", Some("  ")).is_none());
        // A listed realtime model is still sent as-is.
        assert_eq!(
            resolve_model(LIVE_TRANSCRIBE_MODEL, Some("ignored")).unwrap(),
            LIVE_TRANSCRIBE_MODEL
        );
    }
}
