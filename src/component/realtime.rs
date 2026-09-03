// SPDX-License-Identifier: GPL-3.0-only
//! OpenAI realtime WebSocket transcription bridge.
//!
//! Bridges a consumer WebSocket session to OpenAI's realtime transcription API
//! (`wss://api.openai.com/v1/realtime?intent=transcription`), or to whatever
//! OpenAI-compatible endpoint the `base_url` option names.
//!
//! ## Full duplex
//! The session waits on the consumer and the upstream at once, through
//! `subscribe` + `wasi:io/poll` on both resources: audio goes up while
//! transcripts come down, so a `preview` reaches the consumer as soon as
//! OpenAI emits the delta rather than after the take ends. The host must
//! implement `subscribe` on both WS resources.
//!
//! The pure frame-parsing and payload-building helpers (`parse_start`,
//! `is_stop`, `realtime_ws_url`, `session_update_json`, `audio_append_json`,
//! `preview_json`/`done_json`/`error_json`, `Resampler`, `header`) live in the
//! crate root so they compile and unit-test on the host; this module wires them
//! to the wasm-only `super-stt:realtime` resources.

use serde_json::Value;

use super::exports::super_stt::realtime::ws_server::Guest as WsServerGuest;
use super::super_stt::realtime::ws::{self, ConsumerStream, WsError, WsFrame, WsStream};

/// Ends the turn: OpenAI transcribes what is buffered and answers with the
/// delta/completed events. Turn detection is off (see `session_update_json`),
/// so the commit is the session's own doing rather than a VAD's.
const INPUT_AUDIO_COMMIT: &str = r#"{"type":"input_audio_buffer.commit"}"#;

impl WsServerGuest for super::Component {
    fn handle(headers: Vec<(String, Vec<u8>)>, consumer: ConsumerStream) -> Result<(), WsError> {
        run(&headers, &consumer)
    }
}

/// The daemon-injected settings a session runs on, or `None` after telling the
/// consumer which one is missing.
///
/// The batch path reads the same three the same way: the key is optional
/// everywhere except OpenAI's own API, and the `other-realtime` placeholder
/// carries no model name of its own.
fn settings(
    headers: &[(String, Vec<u8>)],
    consumer: &ConsumerStream,
) -> Option<(String, Option<String>, String)> {
    let base_url = crate::header(headers, "x-stt-option-base_url")
        .unwrap_or_else(|| crate::DEFAULT_BASE_URL.to_string());
    let api_key =
        crate::header(headers, "x-stt-secret-openai_api_key").filter(|k| !k.trim().is_empty());
    if api_key.is_none() && crate::needs_api_key(&base_url) {
        let _ = consumer.send_text(&crate::error_json(
            "OpenAI API key not set. Add it in Settings \u{2192} Models \u{2192} OpenAI.",
        ));
        return None;
    }
    let selected = crate::header(headers, "x-stt-model")
        .unwrap_or_else(|| crate::LIVE_TRANSCRIBE_MODEL.to_string());
    let custom = crate::header(headers, "x-stt-option-custom_model");
    let Some(model) = crate::resolve_model(&selected, custom.as_deref()) else {
        let _ = consumer.send_text(&crate::error_json(
            "Custom model name not set. Add it in Settings \u{2192} Models \u{2192} OpenAI, or pick a listed model.",
        ));
        return None;
    };
    Some((base_url, api_key, model))
}

fn run(headers: &[(String, Vec<u8>)], consumer: &ConsumerStream) -> Result<(), WsError> {
    let Some((base_url, api_key, model)) = settings(headers, consumer) else {
        return Ok(());
    };

    // 1. Read and validate the consumer's `start` frame. Its `sample_rate`
    //    decides the resampling ratio and its `language` rides in the session
    //    config, so both are carried forward.
    let (sample_rate, language) = match consumer.recv() {
        Ok(WsFrame::Text(s)) => match crate::parse_start(&s) {
            Ok(start) => start,
            Err(detail) => {
                let _ = consumer.send_text(&crate::error_json(&detail));
                return Ok(());
            }
        },
        // Consumer hung up before starting: nothing to transcribe. The daemon
        // relays a close as a dropped channel, so `Closed` says the same thing.
        Ok(WsFrame::Close(_)) | Err(WsError::Closed) => return Ok(()),
        Ok(WsFrame::Binary(_)) => {
            let _ = consumer.send_text(&crate::error_json("audio before start frame"));
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // 2. Open the upstream WS. The endpoint is named in every failure below —
    //    it is what turns "it did not work" into a setting to check.
    let url = crate::realtime_ws_url(&base_url);
    let (secure, _, _) = crate::parse_base(&base_url);
    // No key means no `authorization` header — an empty `Bearer` is worse than
    // none to a server that does not authenticate.
    let mut upstream_headers = Vec::new();
    if let Some(key) = api_key.as_deref() {
        upstream_headers.push((
            "authorization".to_string(),
            format!("Bearer {key}").into_bytes(),
        ));
    }
    let upstream = match ws::connect(&url, &upstream_headers) {
        Ok(stream) => stream,
        Err(e) => {
            let _ = consumer.send_text(&crate::error_json(&crate::ws_unreachable_detail(
                &url,
                secure,
                &describe(&e),
            )));
            return Ok(());
        }
    };

    // 3. Wait for the session handshake, then configure it: transcription only,
    //    PCM at the one rate OpenAI accepts, no server-side turn detection.
    if !await_session_created(&upstream, consumer, &url) {
        return Ok(());
    }
    if let Err(e) = upstream.send_text(&crate::session_update_json(&model, language.as_deref())) {
        let _ = consumer.send_text(&crate::error_json(&format!(
            "{url} rejected the session configuration ({})",
            describe(&e)
        )));
        return Ok(());
    }

    // 4. Pump audio up and transcripts down, both sockets serviced as either
    //    becomes readable.
    let resampler = crate::Resampler::new(sample_rate, crate::REALTIME_SAMPLE_RATE);
    full_duplex(&upstream, consumer, resampler, &url)?;
    let _ = consumer.close();
    Ok(())
}

/// Both sockets serviced as either becomes readable: audio goes up while
/// transcripts come down, so a `preview` reaches the consumer as soon as the
/// upstream emits it rather than after the take ends.
///
/// Requires a host that implements `subscribe` on both resources — gated on the
/// `x-stt-host-poll` header, because calling it on a host that stubs it traps.
fn full_duplex(
    upstream: &WsStream,
    consumer: &ConsumerStream,
    mut resampler: crate::Resampler,
    url: &str,
) -> Result<(), WsError> {
    let mut appended = 0usize;
    let mut accumulated = String::new();
    // `consumer_done` latches once the consumer has stopped sending, after
    // which only the upstream is polled — polling a finished consumer would
    // spin on its permanently-ready closed state.
    let mut consumer_done = false;
    let mut committed = false;

    let from_consumer = consumer.subscribe();
    let from_upstream = upstream.subscribe();
    loop {
        // Only the sides still worth hearing from. Order matters: it is the
        // index `poll` reports back.
        let mut watching = Vec::with_capacity(2);
        if !consumer_done {
            watching.push(&from_consumer);
        }
        watching.push(&from_upstream);
        let consumer_index = u32::from(!consumer_done);

        for ready in super::wasi::io::poll::poll(&watching) {
            if !consumer_done && ready == 0 {
                match consumer.recv() {
                    // The daemon relays a consumer hang-up as a closed stream,
                    // and drops the send half with it: nothing left to deliver.
                    Err(WsError::Closed) => return Ok(()),
                    Err(e) => return Err(e),
                    Ok(WsFrame::Binary(pcm)) => {
                        let converted = resampler.process(&pcm);
                        if !converted.is_empty() {
                            appended += converted.len();
                            send_up(
                                upstream,
                                consumer,
                                url,
                                &crate::audio_append_json(&converted),
                            )?;
                        }
                    }
                    Ok(WsFrame::Text(s)) if crate::is_stop(&s) => consumer_done = true,
                    Ok(WsFrame::Text(_)) => {} // ignore unknown control frames
                    Ok(WsFrame::Close(_)) => consumer_done = true,
                }
            } else if ready == consumer_index {
                match upstream.recv() {
                    Ok(WsFrame::Text(s)) => {
                        if handle_upstream_event(&s, consumer, &mut accumulated) {
                            return Ok(());
                        }
                    }
                    Ok(WsFrame::Binary(_)) => {} // OpenAI sends JSON text
                    Ok(WsFrame::Close(_)) | Err(WsError::Closed) => {
                        let _ = consumer.send_text(&crate::done_json(accumulated.trim()));
                        return Ok(());
                    }
                    Err(e) => {
                        let _ = consumer.send_text(&crate::error_json(&format!(
                            "{url} stopped sending transcripts ({})",
                            describe(&e)
                        )));
                        return Ok(());
                    }
                }
            }
        }

        // The turn ends once, when the consumer stops. Too little audio to
        // commit is an empty transcript rather than an upstream complaint about
        // a buffer the user never knew existed.
        if consumer_done && !committed {
            committed = true;
            if appended < crate::MIN_COMMIT_BYTES {
                let _ = consumer.send_text(&crate::done_json(accumulated.trim()));
                return Ok(());
            }
            send_up(upstream, consumer, url, INPUT_AUDIO_COMMIT)?;
        }
    }
}

/// Send one control/audio message upstream, telling the consumer if the
/// upstream has stopped listening.
fn send_up(
    upstream: &WsStream,
    consumer: &ConsumerStream,
    url: &str,
    payload: &str,
) -> Result<(), WsError> {
    if let Err(e) = upstream.send_text(payload) {
        let _ = consumer.send_text(&crate::error_json(&format!(
            "{url} stopped accepting audio ({})",
            describe(&e)
        )));
        return Err(WsError::Closed);
    }
    Ok(())
}

/// Read upstream until the session-created handshake event. Returns `false`
/// (after notifying the consumer) if the upstream errors or closes first.
///
/// Both spellings are accepted: the GA API answers `session.created`, and an
/// OpenAI-compatible server built against the earlier transcription-session API
/// answers `transcription_session.created`.
fn await_session_created(upstream: &WsStream, consumer: &ConsumerStream, url: &str) -> bool {
    loop {
        match upstream.recv() {
            Ok(WsFrame::Text(s)) => match event_type(&s).as_deref() {
                Some("session.created" | "transcription_session.created") => return true,
                Some("error") => {
                    let _ = consumer.send_text(&crate::error_json(&format!(
                        "{url} refused the session: {}",
                        error_message(&s)
                    )));
                    return false;
                }
                _ => {} // ignore other handshake chatter
            },
            Ok(WsFrame::Binary(_)) => {}
            Ok(WsFrame::Close(_)) | Err(_) => {
                let _ = consumer.send_text(&crate::error_json(&format!(
                    "{url} closed the connection during the handshake."
                )));
                return false;
            }
        }
    }
}

/// The `type` field of a JSON event, if present.
fn event_type(s: &str) -> Option<String> {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
}

/// The human-readable part of an upstream `error` event, falling back to the
/// raw frame so nothing the server said is lost.
fn error_message(s: &str) -> String {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .or_else(|| v.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| s.to_string())
}

/// A `ws-error` as a sentence fragment, so the consumer sees what the host said
/// rather than a debug-formatted variant.
fn describe(error: &WsError) -> String {
    match error {
        WsError::HostNotAllowed(m)
        | WsError::ConnectFailed(m)
        | WsError::SendFailed(m)
        | WsError::RecvFailed(m)
        | WsError::InvalidUrl(m) => m.clone(),
        WsError::Closed => "connection closed".to_string(),
    }
}

/// Handle one upstream JSON event. Returns `true` when the turn is finished (a
/// completed / failed / error event), `false` to keep draining.
fn handle_upstream_event(s: &str, consumer: &ConsumerStream, accumulated: &mut String) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(s) else {
        return false; // ignore non-JSON frames
    };
    let kind = v.get("type").and_then(Value::as_str).unwrap_or("");

    // A session-level error, or a turn the model could not transcribe. Both are
    // terminal: OpenAI holds the socket open afterwards, so waiting for a close
    // would only stall until the daemon's idle watchdog fires.
    if kind == "error" || kind == "conversation.item.input_audio_transcription.failed" {
        let _ = consumer.send_text(&crate::error_json(&error_message(s)));
        return true;
    }

    if kind == "conversation.item.input_audio_transcription.completed" {
        let transcript = v
            .get("transcript")
            .and_then(Value::as_str)
            .map_or_else(|| accumulated.trim().to_string(), str::to_string);
        let _ = consumer.send_text(&crate::done_json(&transcript));
        return true;
    }

    if kind == "conversation.item.input_audio_transcription.delta" {
        if let Some(delta) = v.get("delta").and_then(Value::as_str) {
            accumulated.push_str(delta);
            let _ = consumer.send_text(&crate::preview_json(accumulated.trim()));
        }
        return false;
    }

    false // unknown event: ignore
}
