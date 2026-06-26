// SPDX-License-Identifier: GPL-3.0-only
//! Self-contained wasmtime host harness that "plays the daemon": it loads the
//! prebuilt OpenAI component and drives the `/v1` contract in-process, exactly
//! as Super STT's daemon does, while confining the component's outbound egress
//! to an allowlist with the same SSRF guard. This repo shares no code with
//! super-stt — the harness is a minimal, independent reimplementation of the
//! daemon's `WasmBackend` (batch `/v1` only; no realtime).
//!
//! Tests pair this with a `wiremock` mock of the *OpenAI upstream*, so both
//! sides of the contract the component speaks (daemon ⇄ component ⇄ upstream)
//! are mocked.

#![allow(dead_code)] // not every test uses every helper

use std::net::{IpAddr, Ipv4Addr, ToSocketAddrs};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use http_body_util::BodyExt;
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::WasiHttpCtx;
use wasmtime_wasi_http::p2::bindings::ProxyPre;
use wasmtime_wasi_http::p2::bindings::http::types::{ErrorCode, Scheme};
use wasmtime_wasi_http::p2::body::HyperOutgoingBody;
use wasmtime_wasi_http::p2::types::{HostFutureIncomingResponse, OutgoingRequestConfig};
use wasmtime_wasi_http::p2::{
    HttpResult, WasiHttpCtxView, WasiHttpHooks, WasiHttpView, default_send_request,
};

/// Path to the prebuilt component (`just build-component`), or `None` if it
/// isn't built — tests skip gracefully in that case so a partial `cargo test`
/// (no wasm build) still passes; CI builds it first. Pair with `let-else`:
///
/// ```ignore
/// let Some(path) = common::component_path() else {
///     eprintln!("skipping: component not built (run `just build-component`)");
///     return;
/// };
/// ```
#[must_use]
pub fn component_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/super_stt_backend_openai.wasm");
    p.exists().then_some(p)
}

// ── host state + egress allowlist (mirrors the daemon's wasm host) ──────────

struct Host {
    table: ResourceTable,
    wasi: WasiCtx,
    http: WasiHttpCtx,
    hooks: AllowlistHooks,
}

impl WasiView for Host {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for Host {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.http,
            table: &mut self.table,
            hooks: &mut self.hooks,
        }
    }
}

struct AllowlistHooks {
    allowed_hosts: Vec<String>,
    allow_loopback: bool,
}

impl WasiHttpHooks for AllowlistHooks {
    fn send_request(
        &mut self,
        request: hyper::Request<HyperOutgoingBody>,
        config: OutgoingRequestConfig,
    ) -> HttpResult<HostFutureIncomingResponse> {
        let authority = request.uri().authority().map(|a| a.as_str().to_string());
        let host = request.uri().host().map(str::to_string);
        let allowed = self.allowed_hosts.iter().any(|a| {
            Some(a.as_str()) == authority.as_deref() || Some(a.as_str()) == host.as_deref()
        });
        if !allowed {
            return Err(ErrorCode::InternalError(Some(format!(
                "outbound host not allowed: {}",
                authority.or(host).unwrap_or_default()
            )))
            .into());
        }
        if let Some(h) = host.as_deref() {
            let port =
                request
                    .uri()
                    .port_u16()
                    .unwrap_or(if request.uri().scheme_str() == Some("http") {
                        80
                    } else {
                        443
                    });
            if let Err(msg) = guard_egress_host(h, port, self.allow_loopback) {
                return Err(ErrorCode::InternalError(Some(msg)).into());
            }
        }
        Ok(default_send_request(request, config))
    }
}

fn guard_egress_host(host: &str, port: u16, allow_loopback: bool) -> Result<(), String> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        if allow_loopback && ip.is_loopback() {
            return Ok(());
        }
        if is_disallowed_ip(&ip) {
            return Err(format!("host {host} is a disallowed address {ip}"));
        }
        return Ok(());
    }
    match (host, port).to_socket_addrs() {
        Ok(addrs) => {
            for addr in addrs {
                let ip = addr.ip();
                if allow_loopback && ip.is_loopback() {
                    continue;
                }
                if is_disallowed_ip(&ip) {
                    return Err(format!("host {host} resolves to a disallowed address {ip}"));
                }
            }
            Ok(())
        }
        Err(_) => Err(format!("cannot resolve host {host}")),
    }
}

fn is_disallowed_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_disallowed_v4(*v4),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_disallowed_v4(mapped);
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

fn is_disallowed_v4(v4: Ipv4Addr) -> bool {
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
}

// ── the backend driver ──────────────────────────────────────────────────────

/// A loaded OpenAI component, driven over the batch `/v1` contract.
pub struct WasmBackend {
    engine: Engine,
    pre: ProxyPre<Host>,
    allowed_hosts: Vec<String>,
    allow_loopback: bool,
    transcribe_headers: Vec<(String, String)>,
    model_id: String,
}

impl WasmBackend {
    /// Load a component the way the daemon does: secrets/options are the
    /// already-formed `x-stt-secret-*` / `x-stt-option-*` header pairs.
    ///
    /// # Errors
    /// Returns an error if the component cannot be loaded or linked.
    pub fn new(
        component_path: &std::path::Path,
        allowed_hosts: Vec<String>,
        model_id: String,
        transcribe_headers: Vec<(String, String)>,
    ) -> Result<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config)?;
        let component = Component::from_file(&engine, component_path)
            .map_err(|e| anyhow!("loading component {}: {e}", component_path.display()))?;
        let mut linker: Linker<Host> = Linker::new(&engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)?;
        let pre = ProxyPre::new(linker.instantiate_pre(&component)?)?;
        Ok(Self {
            engine,
            pre,
            allowed_hosts,
            allow_loopback: false,
            transcribe_headers,
            model_id,
        })
    }

    /// Permit egress to loopback (for a mock upstream bound to 127.0.0.1). Off
    /// by default — the SSRF guard blocks loopback for untrusted backends.
    #[must_use]
    pub fn permit_loopback_egress(mut self) -> Self {
        self.allow_loopback = true;
        self
    }

    async fn invoke(
        &self,
        method: &str,
        path: &str,
        headers: &[(String, String)],
        body: Vec<u8>,
    ) -> Result<(u16, Vec<u8>)> {
        let host = Host {
            table: ResourceTable::new(),
            wasi: WasiCtx::builder().build(),
            http: WasiHttpCtx::new(),
            hooks: AllowlistHooks {
                allowed_hosts: self.allowed_hosts.clone(),
                allow_loopback: self.allow_loopback,
            },
        };
        let mut store = Store::new(&self.engine, host);

        let mut builder = hyper::Request::builder()
            .method(method)
            .uri(format!("http://backend.local{path}"));
        for (key, value) in headers {
            builder = builder.header(key.as_str(), value.as_str());
        }
        let request = builder
            .body(
                http_body_util::Full::new(bytes::Bytes::from(body))
                    .map_err(|never: std::convert::Infallible| -> ErrorCode { match never {} }),
            )
            .context("building backend request")?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = store
            .data_mut()
            .http()
            .new_incoming_request(Scheme::Http, request)?;
        let out = store.data_mut().http().new_response_outparam(tx)?;
        let proxy = self.pre.instantiate_async(&mut store).await?;
        proxy
            .wasi_http_incoming_handler()
            .call_handle(&mut store, req, out)
            .await?;

        let response = rx
            .await
            .context("backend produced no response")?
            .map_err(|e| anyhow!("backend transport error: {e:?}"))?;
        let status = response.status().as_u16();
        let collected = response.into_body().collect().await?.to_bytes();
        Ok((status, collected.to_vec()))
    }

    /// `GET /v1/ping`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or returns non-JSON.
    pub async fn ping(&self) -> Result<serde_json::Value> {
        let (_, body) = self.invoke("GET", "/v1/ping", &[], Vec::new()).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// `GET /v1/status`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or returns non-JSON.
    pub async fn status(&self) -> Result<serde_json::Value> {
        let (_, body) = self.invoke("GET", "/v1/status", &[], Vec::new()).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// `POST /v1/transcribe`. Returns the transcription, or the backend's own
    /// error message on a non-200 response (mirrors the daemon).
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or the backend
    /// reports a non-success status.
    pub async fn transcribe_audio(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        self.transcribe_with_language(audio, sample_rate, None)
            .await
    }

    /// Like [`Self::transcribe_audio`], with an explicit `language` in the
    /// request body — a BCP-47 tag or the reserved `auto`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or the backend
    /// reports a non-success status.
    pub async fn transcribe_with_language(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut payload = serde_json::json!({
            "audio_data": audio,
            "sample_rate": sample_rate,
        });
        if let Some(lang) = language {
            payload["language"] = serde_json::Value::String(lang.to_string());
        }
        let body = serde_json::to_vec(&payload)?;
        let mut headers = self.transcribe_headers.clone();
        headers.push(("x-stt-model".to_string(), self.model_id.clone()));
        let (status, resp) = self
            .invoke("POST", "/v1/transcribe", &headers, body)
            .await?;
        let json: serde_json::Value =
            serde_json::from_slice(&resp).context("parsing backend transcribe response")?;
        if status == 200 {
            json["transcription"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| anyhow!("backend response missing transcription"))
        } else {
            let msg = json
                .get("detail")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("message").and_then(|v| v.as_str()))
                .unwrap_or("transcription failed");
            bail!("{msg}");
        }
    }
}
