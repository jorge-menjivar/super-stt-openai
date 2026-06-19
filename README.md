# Super STT — OpenAI backend

A speech-to-text backend for **[Super STT](https://github.com/jorge-menjivar/super-stt)**.
It proxies audio to [OpenAI](https://platform.openai.com/docs/guides/speech-to-text)'s
hosted transcription API, so transcription runs in the cloud rather than on your machine.

Super STT is an on-device speech-to-text engine. It doesn't ship any models of its own —
it loads **backends** like this one at runtime. This repo packages OpenAI as one of those
backends, shipped as a sandboxed **WASM component** (a `wasi:http` proxy).

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

| Model (`name`)           | Provider | Type   | Languages | Requires        |
| ------------------------ | -------- | ------ | --------- | --------------- |
| `whisper-1`              | openai   | online | en        | OpenAI API key  |
| `gpt-4o-transcribe`      | openai   | online | en        | OpenAI API key  |
| `gpt-4o-mini-transcribe` | openai   | online | en        | OpenAI API key  |

The API base URL is overridable (the `base_url` option) for gateways/proxies.

## What's in here

A small, self-contained Rust `wasi:http` component (`src/lib.rs`) that speaks the Super
STT backend protocol (the `/v1` contract) and forwards audio to OpenAI over
`wasi:http/outgoing-handler`. It shares no code with the Super STT project. The pure
audio/parsing helpers are unit-tested natively; the component as a whole is exercised by a
wasmtime harness under `tests/` that loads the built `.wasm` and drives `/v1` against a
mock upstream.

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
