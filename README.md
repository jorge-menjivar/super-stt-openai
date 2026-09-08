# Super STT — OpenAI backend

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-openai/coverage.json)](https://jorge-menjivar.github.io/super-stt-openai/)

A speech-to-text backend for **[Super STT](https://github.com/jorge-menjivar/super-stt)**.
It proxies audio to [OpenAI](https://platform.openai.com/docs/guides/speech-to-text)'s
hosted transcription API, so transcription runs in the cloud rather than on your machine.

Super STT is an on-device speech-to-text engine. It doesn't ship any models of its own —
it loads **backends** like this one at runtime. This repo packages OpenAI as one of those
backends, shipped as a sandboxed **WASM component** that serves both transports: batch
transcription over `wasi:http`, and streaming transcription over a WebSocket.

## Using it

You don't run this directly. Super STT discovers it through its backend registry,
downloads the prebuilt `.wasm` from this repo's GitHub release, and runs it in-process in
a WASM sandbox whose only network egress is the allowlisted OpenAI API. To use it,
install Super STT, enable OpenAI from the app, and add your **OpenAI API key** —
see the [Super STT docs](https://github.com/jorge-menjivar/super-stt).

## Models

Chosen by `name` when Super STT loads the backend. These are **online** models: they send
audio to OpenAI and need an OpenAI API key (set in the app); no local GPU or weights are
involved.

| Model (`name`)           | Provider | Type     | Languages | Requires                       |
| ------------------------ | -------- | -------- | --------- | ------------------------------ |
| `whisper-1`              | openai   | online   | en        | OpenAI API key                 |
| `gpt-4o-transcribe`      | openai   | online   | en        | OpenAI API key                 |
| `gpt-4o-mini-transcribe` | openai   | online   | en        | OpenAI API key                 |
| `other`                  | openai   | online   | en        | **Custom model name**          |
| `gpt-live-transcribe`    | openai   | realtime | en        | OpenAI API key                 |
| `other-realtime`         | openai   | realtime | en        | **Custom model name**          |

`other` is a placeholder for a model this backend does not list — one served by
an OpenAI-compatible endpoint. Set **Custom model name** to the name that server
expects (e.g. `Systran/faster-whisper-large-v3`) and it is sent instead of
`other`; point **API base URL** at the server, including the API version
(`http://localhost:8000/v1`). Selecting `other` without a custom model name is an
error rather than a request for a model called `other`. Both settings are ignored
by the listed OpenAI models. `other-realtime` is the same placeholder on the
realtime transport, and reads the same **Custom model name**.

The **OpenAI API key** is optional. It is required to reach `api.openai.com` —
requests there are refused without one, since they can only come back 401 — but a
self-hosted or gateway endpoint set through **API base URL** is called with no
`Authorization` header at all when no key is set. Set a key and it is sent as
`Bearer` to whatever endpoint is configured.

## Realtime

The `realtime` models stream instead of uploading a finished recording. Super STT hands
the component a live consumer WebSocket, and the component bridges it to OpenAI's realtime
transcription API (`wss://api.openai.com/v1/realtime?intent=transcription`) — or to the
same endpoint on whatever server **API base URL** names. Partial transcripts come back as
`preview` frames and the final one as `done`.

Two details are worth knowing:

- **Sample rate.** OpenAI's realtime API accepts 24 kHz PCM only, and Super STT streams
  at whatever rate the session declares (16 kHz, typically), so the component resamples on
  the way upstream. Nothing to configure.
- **Previews arrive late.** The host does not yet implement `wasi:io/poll` for WebSocket
  resources, so the component cannot wait on the consumer and OpenAI at the same time. It
  runs half-duplex: all audio goes up first, then the transcript events come back. The
  partials still arrive, but in a burst near the end rather than as you speak.

## What's in here

A small, self-contained Rust component (`src/lib.rs` plus `src/component/`) that speaks
the Super STT backend protocol — the `/v1` contract over `wasi:http`, and the
`super-stt:realtime` WebSocket session — and forwards audio to OpenAI. It shares no code
with the Super STT project; `wit/` is a vendored copy of the protocol's WIT. The pure
audio, request-shaping, and realtime-payload helpers are unit-tested natively; the
component as a whole is exercised by a wasmtime harness under `tests/` that loads the
built `.wasm` and drives both transports against mock upstreams.

## Building from source

Most people never need to — Super STT downloads prebuilt releases. For development
(requires [`just`](https://github.com/casey/just) and the `wasm32-wasip2` target):

```bash
rustup target add wasm32-wasip2
just build-component   # builds target/wasm32-wasip2/release/super_stt_backend_openai.wasm
just ci                # format, lint, build, and test
```

## License

GPL-3.0-only.
