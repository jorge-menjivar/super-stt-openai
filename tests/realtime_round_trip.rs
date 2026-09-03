// SPDX-License-Identifier: GPL-3.0-only
//! Realtime `ws-server` sessions against a `tokio-tungstenite` mock of OpenAI's
//! realtime WebSocket upstream. Drives a consumer transport through
//! `WasmBackend::realtime_session` and asserts both directions: what the guest
//! negotiates and sends upstream (session config, resampled PCM, the commit),
//! and what it returns to the consumer (preview + done frames).
#![allow(clippy::doc_markdown)]

mod common;

use std::time::Duration;

use base64::Engine as _;
use common::{ConsumerStreamTransport, WasmBackend, WsFrame};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc::UnboundedSender;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;

const SECRET: &str = "x-stt-secret-openai_api_key";
const BASE_URL: &str = "x-stt-option-base_url";
const CUSTOM_MODEL: &str = "x-stt-option-custom_model";

/// 3200 bytes = 1600 samples = 100 ms at 16 kHz — the shape a consumer streams.
const CHUNK_100MS: usize = 3200;

macro_rules! component_or_skip {
    () => {
        match common::component_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: component not built (run `just build-component`)");
                return;
            }
        }
    };
}

/// What the mock upstream saw the guest send.
#[derive(Default)]
struct Upstream {
    /// The `session.update` frame, verbatim.
    session_update: Option<String>,
    /// Total PCM bytes across every `input_audio_buffer.append`, base64-decoded.
    audio_bytes: usize,
    /// Whether the turn was committed.
    committed: bool,
}

/// Mock OpenAI realtime upstream. Accepts the WS upgrade, sends
/// `session.created`, records the session config and appended audio, and on
/// `input_audio_buffer.commit` replies with two transcription deltas then the
/// completed event. Returns the bound authority and a handle to what it saw.
async fn start_mock_upstream() -> (String, tokio::task::JoinHandle<Upstream>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string(); // "127.0.0.1:PORT"
    let handle = tokio::spawn(async move {
        let mut seen = Upstream::default();
        // Accept exactly one upstream connection (the guest's).
        let Ok((tcp, _)) = listener.accept().await else {
            return seen;
        };
        let Ok(mut ws) = accept_async(tcp).await else {
            return seen;
        };
        // Handshake: the guest waits for `session.created` before configuring.
        let _ = ws
            .send(WsMessage::Text(r#"{"type":"session.created"}"#.into()))
            .await;
        while let Some(Ok(msg)) = ws.next().await {
            let WsMessage::Text(text) = msg else { continue };
            let event: serde_json::Value =
                serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
            match event["type"].as_str() {
                Some("session.update") => seen.session_update = Some(text.to_string()),
                Some("input_audio_buffer.append") => {
                    let audio = event["audio"].as_str().unwrap_or_default();
                    let pcm = base64::engine::general_purpose::STANDARD
                        .decode(audio)
                        .expect("append payload must be base64");
                    seen.audio_bytes += pcm.len();
                }
                Some("input_audio_buffer.commit") => {
                    seen.committed = true;
                    for event in [
                        r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"hello "}"#,
                        r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"world"}"#,
                        r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"hello world"}"#,
                    ] {
                        let _ = ws.send(WsMessage::Text(event.into())).await;
                    }
                    // Done; the guest closes after the completed event.
                    break;
                }
                _ => {}
            }
        }
        seen
    });
    (authority, handle)
}

/// Feed one session's worth of consumer frames, keeping the sender alive until
/// the session ends so the guest never sees an early close.
fn drive_consumer(
    tx: UnboundedSender<WsFrame>,
    start: &'static str,
    chunks: usize,
    chunk_bytes: usize,
) -> tokio::task::JoinHandle<UnboundedSender<WsFrame>> {
    tokio::spawn(async move {
        let mut frames = vec![WsFrame::Text(start.to_string())];
        // s16le mono; silence is fine for the mock.
        frames.extend((0..chunks).map(|_| WsFrame::Binary(vec![0u8; chunk_bytes])));
        frames.push(WsFrame::Text(r#"{"type":"stop"}"#.to_string()));
        for frame in frames {
            // A session can end before every frame is read — a missing setting
            // ends it at the first one — so a closed channel is not a failure.
            if tx.send(frame).is_err() {
                break;
            }
        }
        tx
    })
}

/// Collect the text frames the guest sent to the consumer.
fn consumer_texts(rx: &mut tokio::sync::mpsc::UnboundedReceiver<WsFrame>) -> Vec<String> {
    let mut texts = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        if let WsFrame::Text(s) = frame {
            texts.push(s);
        }
    }
    texts
}

/// The full bridge: `start` → resampled audio → commit → deltas → done. The
/// upstream sees PCM at the one rate OpenAI accepts, not the consumer's.
#[tokio::test]
async fn realtime_round_trip() {
    let path = component_or_skip!();
    let (authority, mock) = start_mock_upstream().await;

    // The guest builds the upstream URL from x-stt-option-base_url. Point it at
    // the mock over plaintext ws:// (http:// -> ws:// in the guest). The mock is
    // on loopback, which the SSRF guard blocks for untrusted backends, so the
    // test opts in via `permit_loopback_egress` below.
    let backend = WasmBackend::new(
        &path,
        vec![authority.clone()],
        "gpt-live-transcribe".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    // Channels: consumer_tx -> guest (incoming); guest -> guest_rx (outgoing).
    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };

    let driver = drive_consumer(
        consumer_tx,
        r#"{"type":"start","sample_rate":16000,"language":"en"}"#,
        4,
        CHUNK_100MS,
    );

    // Run the session with a timeout so a hang fails loudly.
    let session =
        tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport));
    let result = session.await.expect("session timed out");
    let _held = driver.await.unwrap(); // keep consumer_tx alive until session ends
    result.expect("session returned an error");

    let texts = consumer_texts(&mut guest_rx);
    assert!(
        texts.iter().any(|t| t.contains(r#""type":"preview""#)),
        "expected at least one preview frame; got {texts:?}"
    );
    let done = texts
        .iter()
        .find(|t| t.contains(r#""type":"done""#))
        .unwrap_or_else(|| panic!("expected a done frame; got {texts:?}"));
    assert!(
        done.contains("hello world"),
        "done frame should contain the transcript; got {done}"
    );

    let seen = mock.await.expect("mock upstream panicked");
    assert!(seen.committed, "the turn should have been committed");

    let config: serde_json::Value = serde_json::from_str(
        &seen
            .session_update
            .expect("the guest must configure the session"),
    )
    .expect("session.update must be JSON");
    let input = &config["session"]["audio"]["input"];
    assert_eq!(config["session"]["type"], "transcription");
    assert_eq!(input["format"]["type"], "audio/pcm");
    assert_eq!(input["format"]["rate"], 24000);
    assert_eq!(input["transcription"]["model"], "gpt-live-transcribe");
    // This model spells the language as a list; the singular field must not
    // also be present, which is an upstream error.
    assert_eq!(input["transcription"]["languages"][0], "en");
    assert!(input["transcription"]["language"].is_null());
    assert!(input["turn_detection"].is_null());

    // 4 chunks × 1600 samples at 16 kHz resample to 24 kHz: 1.5× the samples,
    // give or take a sample per chunk boundary.
    let out_samples = i64::try_from(seen.audio_bytes / 2).unwrap();
    let expected = 4 * 1600 * 3 / 2;
    assert!(
        (out_samples - expected).abs() <= 4,
        "expected ~{expected} samples upstream, got {out_samples}"
    );
}

/// A custom model on an OpenAI-compatible endpoint: `other-realtime` resolves
/// to the `custom_model` option, and a model that is not `gpt-live-transcribe`
/// takes the singular `language` field.
#[tokio::test]
async fn realtime_custom_model_is_sent_upstream() {
    let path = component_or_skip!();
    let (authority, mock) = start_mock_upstream().await;

    let backend = WasmBackend::new(
        &path,
        vec![authority.clone()],
        "other-realtime".to_string(),
        vec![
            (BASE_URL.to_string(), format!("http://{authority}")),
            (
                CUSTOM_MODEL.to_string(),
                "Systran/faster-whisper".to_string(),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };
    let driver = drive_consumer(
        consumer_tx,
        r#"{"type":"start","sample_rate":16000,"language":"es"}"#,
        2,
        CHUNK_100MS,
    );

    let result = tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport))
        .await
        .expect("session timed out");
    let _held = driver.await.unwrap();
    result.expect("session returned an error");

    let texts = consumer_texts(&mut guest_rx);
    assert!(
        texts.iter().any(|t| t.contains("hello world")),
        "expected the transcript to come back; got {texts:?}"
    );

    let seen = mock.await.expect("mock upstream panicked");
    let config: serde_json::Value = serde_json::from_str(
        &seen
            .session_update
            .expect("the guest must configure the session"),
    )
    .expect("session.update must be JSON");
    let transcription = &config["session"]["audio"]["input"]["transcription"];
    assert_eq!(transcription["model"], "Systran/faster-whisper");
    assert_eq!(transcription["language"], "es");
    assert!(transcription["languages"].is_null());
}

/// Less than the 100 ms OpenAI will commit is answered with an empty transcript
/// — the buffer is never committed, so the upstream never gets to complain
/// about something the user cannot see.
#[tokio::test]
async fn realtime_short_audio_returns_an_empty_transcript() {
    let path = component_or_skip!();
    let (authority, mock) = start_mock_upstream().await;

    let backend = WasmBackend::new(
        &path,
        vec![authority.clone()],
        "gpt-live-transcribe".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };
    // 20 ms at 16 kHz — well under the 100 ms OpenAI needs, even after
    // resampling to 24 kHz.
    let driver = drive_consumer(
        consumer_tx,
        r#"{"type":"start","sample_rate":16000}"#,
        1,
        640,
    );

    let result = tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport))
        .await
        .expect("session timed out");
    let _held = driver.await.unwrap();
    result.expect("session returned an error");

    let texts = consumer_texts(&mut guest_rx);
    let done = texts
        .iter()
        .find(|t| t.contains(r#""type":"done""#))
        .unwrap_or_else(|| panic!("expected a done frame; got {texts:?}"));
    assert!(
        done.contains(r#""transcription":"""#),
        "expected an empty transcript; got {done}"
    );

    let seen = mock.await.expect("mock upstream panicked");
    assert!(
        !seen.committed,
        "a sub-100 ms buffer must not be committed upstream"
    );
}

/// A consumer that just disconnects ends the session cleanly. The daemon relays
/// that as a dropped channel and drops the send half with it, so there is no
/// transcript to deliver — the session stops rather than committing a turn
/// upstream that nothing will read, and it is not reported as a failure.
#[tokio::test]
async fn realtime_consumer_disconnect_ends_the_session_cleanly() {
    let path = component_or_skip!();
    let (authority, mock) = start_mock_upstream().await;

    let backend = WasmBackend::new(
        &path,
        vec![authority.clone()],
        "gpt-live-transcribe".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };
    consumer_tx
        .send(WsFrame::Text(
            r#"{"type":"start","sample_rate":16000}"#.to_string(),
        ))
        .unwrap();
    for _ in 0..4 {
        consumer_tx
            .send(WsFrame::Binary(vec![0u8; CHUNK_100MS]))
            .unwrap();
    }
    // No `stop` frame — just hang up. The queued frames are still delivered.
    drop(consumer_tx);

    let result = tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport))
        .await
        .expect("session timed out");
    result.expect("a consumer disconnect is not a session error");

    let texts = consumer_texts(&mut guest_rx);
    assert!(texts.is_empty(), "nothing is deliverable; got {texts:?}");

    let seen = mock.await.expect("mock upstream panicked");
    assert!(
        !seen.committed,
        "a turn nobody can read should not be committed upstream"
    );
    // The audio still went up before the disconnect was noticed.
    assert!(
        seen.audio_bytes > 0,
        "audio should have reached the upstream"
    );
}

/// No key against OpenAI's own API: the session names the missing setting and
/// ends, without opening a socket.
#[tokio::test]
async fn realtime_without_key_reports_missing_secret() {
    let path = component_or_skip!();

    let backend = WasmBackend::new(
        &path,
        Vec::new(),
        "gpt-live-transcribe".to_string(),
        Vec::new(),
    )
    .expect("load backend");

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };
    let driver = drive_consumer(
        consumer_tx,
        r#"{"type":"start","sample_rate":16000}"#,
        1,
        CHUNK_100MS,
    );

    let result = tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport))
        .await
        .expect("session timed out");
    let _held = driver.await.unwrap();
    result.expect("session returned an error");

    let texts = consumer_texts(&mut guest_rx);
    assert!(
        texts
            .iter()
            .any(|t| t.contains("OpenAI API key not set") && t.contains(r#""type":"error""#)),
        "expected the missing-secret error frame; got {texts:?}"
    );
}

/// The `other-realtime` placeholder without a custom model name is an error
/// about the setting, not a request for a model called `other-realtime`.
#[tokio::test]
async fn realtime_placeholder_without_custom_model_reports_the_setting() {
    let path = component_or_skip!();

    let backend = WasmBackend::new(
        &path,
        Vec::new(),
        "other-realtime".to_string(),
        vec![(SECRET.to_string(), "test-key".to_string())],
    )
    .expect("load backend");

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };
    let driver = drive_consumer(
        consumer_tx,
        r#"{"type":"start","sample_rate":16000}"#,
        1,
        CHUNK_100MS,
    );

    let result = tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport))
        .await
        .expect("session timed out");
    let _held = driver.await.unwrap();
    result.expect("session returned an error");

    let texts = consumer_texts(&mut guest_rx);
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Custom model name not set")),
        "expected the missing-option error frame; got {texts:?}"
    );
}

/// Mock upstream that answers the FIRST audio append with a delta instead of
/// waiting for the commit — so a guest that is listening while it sends has
/// something to hear, and one that is not, does not.
async fn start_eager_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string();
    let handle = tokio::spawn(async move {
        let Ok((tcp, _)) = listener.accept().await else {
            return;
        };
        let Ok(mut ws) = accept_async(tcp).await else {
            return;
        };
        let _ = ws
            .send(WsMessage::Text(r#"{"type":"session.created"}"#.into()))
            .await;
        let mut answered = false;
        while let Some(Ok(msg)) = ws.next().await {
            let WsMessage::Text(text) = msg else { continue };
            let event: serde_json::Value =
                serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
            match event["type"].as_str() {
                Some("input_audio_buffer.append") if !answered => {
                    answered = true;
                    let _ = ws.send(WsMessage::Text(
                        r#"{"type":"conversation.item.input_audio_transcription.delta","delta":"live"}"#.into(),
                    )).await;
                }
                Some("input_audio_buffer.commit") => {
                    let _ = ws.send(WsMessage::Text(
                        r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"live text"}"#.into(),
                    )).await;
                    break;
                }
                _ => {}
            }
        }
    });
    (authority, handle)
}

/// Full duplex: a transcript reaches the consumer *while it is still sending
/// audio*, which is the whole point of `subscribe`.
///
/// The mock answers the first audio append with a delta instead of waiting for
/// the commit, and the driver refuses to send `stop` until a `preview` has come
/// back. A guest reading its two sockets in sequence would be blocked on the
/// consumer and never look upstream, so that preview would never arrive and
/// this test would time out.
#[tokio::test]
async fn realtime_streams_previews_before_stop() {
    let path = component_or_skip!();

    let (authority, mock) = start_eager_upstream().await;
    let backend = WasmBackend::new(
        &path,
        vec![authority.clone()],
        "gpt-live-transcribe".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };

    // Run the session concurrently with the consumer driver; a spawn would
    // need `'static`, and the backend is borrowed.
    let driver = async {
        consumer_tx
            .send(WsFrame::Text(
                r#"{"type":"start","sample_rate":16000}"#.to_string(),
            ))
            .unwrap();
        // Two chunks: one 100 ms chunk resamples to 4798 bytes, just under the
        // 100 ms OpenAI needs to commit a turn, which would end the take with
        // an empty transcript instead of exercising the commit path.
        for _ in 0..2 {
            consumer_tx
                .send(WsFrame::Binary(vec![0u8; CHUNK_100MS]))
                .unwrap();
        }

        // The assertion: a preview arrives with `stop` still unsent.
        let preview = loop {
            match guest_rx.recv().await {
                Some(WsFrame::Text(text)) if text.contains(r#""type":"preview""#) => break text,
                Some(_) => {}
                None => panic!("session ended before any preview arrived"),
            }
        };
        assert!(preview.contains("live"), "got {preview}");

        // Only now finish the turn.
        consumer_tx
            .send(WsFrame::Text(r#"{"type":"stop"}"#.to_string()))
            .unwrap();
        consumer_tx
    };

    let (result, _held) = tokio::time::timeout(
        Duration::from_secs(30),
        futures::future::join(backend.realtime_session(transport), driver),
    )
    .await
    .expect("a preview must arrive while audio is still being sent");
    result.expect("session returned an error");
    mock.await.expect("mock upstream panicked");
}
