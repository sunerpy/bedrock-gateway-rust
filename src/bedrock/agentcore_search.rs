//! Amazon Bedrock AgentCore Web Search (`InvokeWebSearch`) behind the
//! [`WebSearchBackend`] seam, for `WEB_SEARCH_BACKEND=agentcore`.
//!
//! A dedicated `reqwest` client posts `{"searchQuery", "maxResults"}` to
//! `https://bedrock-agentcore.{region}.amazonaws.com/web-search/web-search.v1/invoke`
//! and signs it with SigV4 for the service `bedrock-agentcore`, with the
//! shared AWS config's credential chain: the ECS task role, the Lambda
//! execution role, an IRSA web identity, or a local profile. AgentCore accepts
//! only SigV4, so the Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`) is never
//! used here. The AWS SDK crate for AgentCore is not used: every version with
//! `InvokeWebSearch` needs a newer smithy runtime than the gateway's other AWS
//! clients, and moving them would change their retry behavior.
//!
//! One search:
//! 1. takes a token from the in-process rate limiter (`WEB_SEARCH_MAX_TPS`),
//!    waiting at most 1 s, or fails as throttled without calling AWS (each
//!    attempt, the retry included, takes its own token);
//! 2. runs inside one time budget (`WEB_SEARCH_TIMEOUT_SECS`) and retries once
//!    on 429, 500, 503 or a transport failure while at least 2 s of the
//!    budget remain, after `min(retryAfterSeconds, 1 s)`;
//! 3. ends in a [`SearchError`] when it fails. Nothing here becomes an
//!    [`AppError`] for a client request: AgentCore errors never reach
//!    `from_bedrock_sdk_error` or `responses_error`, which would report a
//!    `ValidationException` as `context_length_exceeded`.
//!
//! Logs carry the HTTP status, the AWS error type and the AgentCore request
//! id; never the query, the results or an error message.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, SignableBody, SignableRequest, SigningParams, SigningSettings,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio::time::Instant;

use crate::bedrock::web_search::{
    SearchError, SearchOutcome, SearchResult, WebSearchBackend, WebSearchLimits,
    MAX_RESULTS_PER_SEARCH,
};
use crate::config::AppSettings;
use crate::error::AppError;

/// SigV4 signing name and endpoint prefix of AgentCore.
const SERVICE: &str = "bedrock-agentcore";

/// The AWS-managed web search tool.
const WEB_SEARCH_ID: &str = "web-search.v1";

/// `WEB_SEARCH_BACKEND` values.
const BACKEND_OFF: &str = "off";
const BACKEND_AGENTCORE: &str = "agentcore";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest wait for a rate-limit token before a search fails as throttled.
const TOKEN_MAX_WAIT: Duration = Duration::from_secs(1);

/// Cached credentials are refreshed this long before they expire.
const CREDENTIAL_REFRESH_WINDOW: Duration = Duration::from_secs(5 * 60);

const JSON: &str = "application/json";

/// AgentCore Web Search client. Cheap to share behind an `Arc`.
pub struct AgentCoreWebSearch {
    http: reqwest::Client,
    /// Full `InvokeWebSearch` URL.
    endpoint: String,
    /// SigV4 signing region.
    region: String,
    credentials: CredentialCache,
    limiter: TokenBucket,
    timeout: Duration,
    token_max_wait: Duration,
    retry: RetryPolicy,
}

impl AgentCoreWebSearch {
    /// Build the client for `region`. `endpoint` (`scheme://host[:port]`)
    /// replaces the regional AgentCore endpoint; tests point it at a local
    /// server. No network call or credential lookup happens here.
    ///
    /// # Errors
    /// [`AppError::Internal`] when the region or endpoint cannot form the
    /// request URL, or the HTTP client cannot be built.
    pub fn new(
        region: String,
        endpoint: Option<String>,
        credentials: SharedCredentialsProvider,
        limits: &WebSearchLimits,
    ) -> Result<Self, AppError> {
        let valid_region = !region.is_empty()
            && region
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !valid_region {
            return Err(AppError::Internal(format!(
                "invalid AgentCore web search region '{region}'"
            )));
        }
        let base = endpoint.unwrap_or_else(|| format!("https://{SERVICE}.{region}.amazonaws.com"));
        let endpoint = format!(
            "{}/web-search/{WEB_SEARCH_ID}/invoke",
            base.trim_end_matches('/')
        );
        url::Url::parse(&endpoint).map_err(|e| {
            AppError::Internal(format!("invalid AgentCore web search endpoint: {e}"))
        })?;
        // A client of its own: the search budget is seconds, not the 900 s
        // Bedrock read timeout. Like the mantle client, it honors the
        // environment's proxy settings.
        let http = reqwest::Client::builder()
            .timeout(limits.timeout)
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(concat!("bedrock-gateway-rust/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| {
                AppError::Internal(format!(
                    "failed to build the AgentCore web search HTTP client: {e}"
                ))
            })?;
        Ok(Self {
            http,
            endpoint,
            region,
            credentials: CredentialCache::new(credentials),
            limiter: TokenBucket::new(limits.max_tps),
            timeout: limits.timeout,
            token_max_wait: TOKEN_MAX_WAIT,
            retry: RetryPolicy::DEFAULT,
        })
    }

    async fn search_with_retry(
        &self,
        body: &[u8],
        deadline: Instant,
    ) -> Result<SearchOutcome, SearchError> {
        let first = match self.attempt(body).await {
            Ok(outcome) => return Ok(outcome),
            Err(failure) => failure,
        };
        let remaining = deadline.saturating_duration_since(Instant::now());
        let will_retry = first.retryable && remaining >= self.retry.min_budget;
        first.log(1, will_retry);
        if !will_retry {
            return Err(first.error);
        }
        tokio::time::sleep(self.retry.delay(first.retry_after)).await;
        self.attempt(body).await.map_err(|second| {
            second.log(2, false);
            second.error
        })
    }

    /// One signed `InvokeWebSearch` call.
    async fn attempt(&self, body: &[u8]) -> Result<SearchOutcome, AttemptFailure> {
        if !self.limiter.acquire(self.token_max_wait).await {
            return Err(AttemptFailure::local(
                SearchError::Throttled,
                "rate_limited",
            ));
        }
        let credentials = self
            .credentials
            .get()
            .await
            .ok_or_else(|| AttemptFailure::local(SearchError::Unavailable, "credentials"))?;
        let signed = self
            .signed_headers(credentials, body, SystemTime::now())
            .ok_or_else(|| AttemptFailure::local(SearchError::Unavailable, "signing"))?;
        let response = self
            .http
            .post(&self.endpoint)
            .header(CONTENT_TYPE, JSON)
            .headers(signed)
            .body(body.to_vec())
            .send()
            .await
            .map_err(AttemptFailure::transport)?;

        let status = response.status().as_u16();
        let headers = response.headers();
        let request_id = header_text(headers, "x-amzn-requestid")
            .or_else(|| header_text(headers, "x-amz-request-id"))
            .and_then(|id| log_token(&id));
        let header_error_type = header_text(headers, "x-amzn-errortype");
        let bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => {
                let mut failure = AttemptFailure::transport(error);
                failure.status = Some(status);
                failure.request_id = request_id;
                return Err(failure);
            }
        };

        if (200..300).contains(&status) {
            return match parse_results(&bytes) {
                Some(results) => Ok(SearchOutcome {
                    results,
                    request_id,
                }),
                None => Err(AttemptFailure {
                    status: Some(status),
                    request_id,
                    ..AttemptFailure::local(SearchError::Unavailable, "malformed_response")
                }),
            };
        }

        let error_body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        let error_type = header_error_type
            .as_deref()
            .or_else(|| {
                error_body
                    .get("__type")
                    .or_else(|| error_body.get("code"))
                    .and_then(Value::as_str)
            })
            .map(sanitize_error_code)
            .and_then(log_token);
        Err(AttemptFailure {
            error: classify_failure(status, error_type.as_deref()),
            retryable: is_retryable_status(status),
            retry_after: retry_after(&error_body),
            status: Some(status),
            error_type,
            request_id,
            reason: "http_status",
        })
    }

    /// The SigV4 headers (`x-amz-date`, `authorization`, and
    /// `x-amz-security-token` for temporary credentials) for a POST of `body`
    /// to the endpoint with `content-type: application/json`. `host` is signed
    /// from the URL, which is also where the HTTP client takes it from.
    fn signed_headers(
        &self,
        credentials: Credentials,
        body: &[u8],
        time: SystemTime,
    ) -> Option<HeaderMap> {
        let identity = Identity::from(credentials);
        let params: SigningParams<'_> = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(SERVICE)
            .time(time)
            .settings(SigningSettings::default())
            .build()
            .ok()?
            .into();
        let request = SignableRequest::new(
            "POST",
            self.endpoint.as_str(),
            [("content-type", JSON)].into_iter(),
            SignableBody::Bytes(body),
        )
        .ok()?;
        let (instructions, _signature) = sign(request, &params).ok()?.into_parts();
        let (headers, _query) = instructions.into_parts();
        let mut signed = HeaderMap::new();
        for header in headers {
            let name = HeaderName::from_bytes(header.name().as_bytes()).ok()?;
            let mut value = HeaderValue::from_str(header.value()).ok()?;
            value.set_sensitive(header.sensitive() || name == AUTHORIZATION);
            signed.insert(name, value);
        }
        Some(signed)
    }
}

#[async_trait::async_trait]
impl WebSearchBackend for AgentCoreWebSearch {
    async fn search(&self, query: &str, max_results: u32) -> Result<SearchOutcome, SearchError> {
        if query.trim().is_empty() {
            return Err(SearchError::InvalidInput);
        }
        let body = request_body(query, max_results);
        let deadline = Instant::now() + self.timeout;
        match tokio::time::timeout_at(deadline, self.search_with_retry(&body, deadline)).await {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!(
                    outcome = SearchError::Unavailable.code().as_str(),
                    reason = "timeout",
                    timeout_ms = u64::try_from(self.timeout.as_millis()).unwrap_or(u64::MAX),
                    "AgentCore web search ran out of its time budget"
                );
                Err(SearchError::Unavailable)
            }
        }
    }
}

/// The `InvokeWebSearch` body. AgentCore rejects any other top-level field.
fn request_body(query: &str, max_results: u32) -> Vec<u8> {
    json!({
        "maxResults": max_results.clamp(1, MAX_RESULTS_PER_SEARCH),
        "searchQuery": query,
    })
    .to_string()
    .into_bytes()
}

/// `results[]` of a 200 response. A result without a non-empty `text` is
/// dropped; `title`, `url` and `publishedAt` may be absent.
fn parse_results(body: &[u8]) -> Option<Vec<SearchResult>> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let results = value.get("results")?.as_array()?;
    Some(results.iter().filter_map(parse_result).collect())
}

fn parse_result(result: &Value) -> Option<SearchResult> {
    let text = result
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())?;
    let field = |name: &str| result.get(name).and_then(Value::as_str).map(str::to_string);
    Some(SearchResult {
        title: field("title"),
        url: field("url"),
        text: text.to_string(),
        published_at: field("publishedAt"),
    })
}

fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

/// `value` when it is a short code or id (ASCII letters, digits, `-`, `_`,
/// `.`), the only text from an AgentCore response that reaches logs.
fn log_token(value: &str) -> Option<String> {
    let plausible = !value.is_empty()
        && value.len() <= 128
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    plausible.then(|| value.to_string())
}

/// The bare error type of an AWS error code: `ValidationException` from
/// `ValidationException:http://...` (`x-amzn-errortype`) or from
/// `com.amazonaws.bedrockagentcore#ValidationException` (`__type`).
fn sanitize_error_code(raw: &str) -> &str {
    let code = raw.split_once(':').map_or(raw, |(code, _)| code);
    code.split_once('#').map_or(code, |(_, code)| code).trim()
}

/// The error type decides; the status decides only when the response names
/// no type. Configuration problems (403 permissions or signatures, 404, 5xx)
/// are all `Unavailable`: the model cannot fix them by changing the query.
fn classify_failure(status: u16, error_type: Option<&str>) -> SearchError {
    match error_type {
        Some("ValidationException") => SearchError::InvalidInput,
        Some("ThrottlingException") => SearchError::Throttled,
        _ if status == 429 => SearchError::Throttled,
        None if status == 400 => SearchError::InvalidInput,
        _ => SearchError::Unavailable,
    }
}

fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 503)
}

/// `retryAfterSeconds` of a `ServiceUnavailableException` body.
fn retry_after(error_body: &Value) -> Option<Duration> {
    error_body
        .get("retryAfterSeconds")
        .and_then(Value::as_u64)
        .map(Duration::from_secs)
}

/// The single retry of a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetryPolicy {
    /// Retry only while at least this much of the budget remains.
    min_budget: Duration,
    /// Wait before the retry when the error names no `retryAfterSeconds`.
    default_delay: Duration,
    /// Longest wait before the retry.
    max_delay: Duration,
}

impl RetryPolicy {
    const DEFAULT: Self = Self {
        min_budget: Duration::from_secs(2),
        default_delay: Duration::from_millis(250),
        max_delay: Duration::from_secs(1),
    };

    fn delay(&self, retry_after: Option<Duration>) -> Duration {
        retry_after
            .unwrap_or(self.default_delay)
            .min(self.max_delay)
    }
}

/// Why one attempt failed, with what the log line may carry.
#[derive(Debug)]
struct AttemptFailure {
    error: SearchError,
    retryable: bool,
    retry_after: Option<Duration>,
    status: Option<u16>,
    /// AWS error type, for example `AccessDeniedException`.
    error_type: Option<String>,
    request_id: Option<String>,
    reason: &'static str,
}

impl AttemptFailure {
    /// A failure before any HTTP call; never retried.
    fn local(error: SearchError, reason: &'static str) -> Self {
        Self {
            error,
            retryable: false,
            retry_after: None,
            status: None,
            error_type: None,
            request_id: None,
            reason,
        }
    }

    fn transport(error: reqwest::Error) -> Self {
        Self {
            error: SearchError::Unavailable,
            retryable: error.is_connect()
                || error.is_timeout()
                || error.is_request()
                || error.is_body(),
            retry_after: None,
            status: None,
            error_type: None,
            request_id: None,
            reason: if error.is_timeout() {
                "timeout"
            } else {
                "transport"
            },
        }
    }

    /// Failures an operator has to fix (missing permission, wrong tool id,
    /// no usable credentials) log at error; the rest at warn.
    fn needs_operator(&self) -> bool {
        matches!(self.status, Some(403 | 404)) || matches!(self.reason, "credentials" | "signing")
    }

    fn log(&self, attempt: u8, will_retry: bool) {
        let outcome = self.error.code().as_str();
        let error_type = self.error_type.as_deref();
        let request_id = self.request_id.as_deref();
        if self.needs_operator() {
            tracing::error!(
                attempt,
                will_retry,
                outcome,
                reason = self.reason,
                status = self.status,
                error_type,
                agentcore_request_id = request_id,
                "AgentCore web search failed"
            );
        } else {
            tracing::warn!(
                attempt,
                will_retry,
                outcome,
                reason = self.reason,
                status = self.status,
                error_type,
                agentcore_request_id = request_id,
                "AgentCore web search failed"
            );
        }
    }
}

/// Credentials from the provider, cached until shortly before they expire.
struct CredentialCache {
    provider: SharedCredentialsProvider,
    cached: tokio::sync::Mutex<Option<Credentials>>,
}

impl CredentialCache {
    fn new(provider: SharedCredentialsProvider) -> Self {
        Self {
            provider,
            cached: tokio::sync::Mutex::new(None),
        }
    }

    /// Credentials to sign with: the cached ones until
    /// [`CREDENTIAL_REFRESH_WINDOW`] before they expire (credentials without
    /// an expiry stay cached), else fresh ones from the provider. The lock is
    /// held across the provider call, so concurrent searches refresh once.
    /// When a refresh fails, cached credentials that have not expired yet are
    /// still used; `None` means there is nothing to sign with.
    async fn get(&self) -> Option<Credentials> {
        let mut cached = self.cached.lock().await;
        let now = SystemTime::now();
        if let Some(credentials) = cached
            .as_ref()
            .filter(|credentials| !expires_within(credentials, now, CREDENTIAL_REFRESH_WINDOW))
        {
            return Some(credentials.clone());
        }
        match self.provider.provide_credentials().await {
            Ok(fresh) => {
                *cached = Some(fresh.clone());
                Some(fresh)
            }
            Err(_) => cached
                .as_ref()
                .filter(|credentials| !expires_within(credentials, now, Duration::ZERO))
                .cloned(),
        }
    }
}

fn expires_within(credentials: &Credentials, now: SystemTime, window: Duration) -> bool {
    credentials
        .expiry()
        .is_some_and(|expiry| expiry <= now + window)
}

/// In-process token bucket: `max_tps` tokens per second, bursts of `max_tps`.
struct TokenBucket {
    state: Mutex<BucketState>,
}

impl TokenBucket {
    fn new(max_tps: u32) -> Self {
        Self {
            state: Mutex::new(BucketState::full(max_tps, Instant::now())),
        }
    }

    /// Take a token, sleeping for a reserved one when it comes within
    /// `max_wait`. `false` means none comes in time and none was taken.
    async fn acquire(&self, max_wait: Duration) -> bool {
        let reservation = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            reserve(&mut state, Instant::now(), max_wait)
        };
        match reservation {
            Reservation::Ready => true,
            Reservation::Wait(wait) => {
                tokio::time::sleep(wait).await;
                true
            }
            Reservation::Rejected => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BucketState {
    /// Tokens added per second.
    rate: f64,
    /// Most tokens held.
    burst: f64,
    /// Below zero while tokens are reserved by waiting callers.
    tokens: f64,
    updated: Instant,
}

impl BucketState {
    fn full(max_tps: u32, now: Instant) -> Self {
        let rate = f64::from(max_tps.max(1));
        Self {
            rate,
            burst: rate,
            tokens: rate,
            updated: now,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Reservation {
    /// A token was taken.
    Ready,
    /// A token is reserved; it is the caller's after this wait.
    Wait(Duration),
    /// No token within the allowed wait; nothing was taken.
    Rejected,
}

/// Refill the bucket up to `now`, then take or reserve one token.
fn reserve(state: &mut BucketState, now: Instant, max_wait: Duration) -> Reservation {
    let elapsed = now.saturating_duration_since(state.updated).as_secs_f64();
    state.tokens = (state.tokens + elapsed * state.rate).min(state.burst);
    state.updated = state.updated.max(now);
    if state.tokens >= 1.0 {
        state.tokens -= 1.0;
        return Reservation::Ready;
    }
    let wait = Duration::from_secs_f64((1.0 - state.tokens) / state.rate);
    if wait > max_wait {
        return Reservation::Rejected;
    }
    state.tokens -= 1.0;
    Reservation::Wait(wait)
}

/// The web search backend named by `WEB_SEARCH_BACKEND`, built at boot.
///
/// `None` when the setting is `off` (the default), names an unknown backend,
/// or AgentCore cannot be set up; every case but `off` logs a WARN, and boot
/// carries on with web search off. AgentCore uses `aws`'s credential chain
/// and `web_search_region` (default `aws_region`).
#[must_use]
pub fn resolve_web_search_backend(
    settings: &AppSettings,
    aws: &aws_config::SdkConfig,
) -> Option<Arc<dyn WebSearchBackend>> {
    let backend = settings.web_search_backend.trim();
    if backend.is_empty() || backend.eq_ignore_ascii_case(BACKEND_OFF) {
        return None;
    }
    if !backend.eq_ignore_ascii_case(BACKEND_AGENTCORE) {
        tracing::warn!(
            value = backend,
            "WEB_SEARCH_BACKEND is neither off nor agentcore; web search stays off"
        );
        return None;
    }
    let Some(credentials) = aws.credentials_provider() else {
        tracing::warn!(
            "WEB_SEARCH_BACKEND=agentcore needs SigV4 credentials, but the AWS config has no \
             credentials provider; web search stays off"
        );
        return None;
    };
    if settings.bedrock_api_key.is_some() {
        tracing::info!(
            "AgentCore web search signs with the SigV4 credential chain (task role, IRSA or \
             profile); the Bedrock API key is not used for it"
        );
    }
    let region = settings
        .web_search_region
        .as_deref()
        .map(str::trim)
        .filter(|region| !region.is_empty())
        .unwrap_or(settings.aws_region.as_str());
    let limits = WebSearchLimits::from_settings(settings);
    match AgentCoreWebSearch::new(region.to_string(), None, credentials, &limits) {
        Ok(client) => {
            tracing::info!(
                region,
                max_calls = limits.max_calls,
                max_results = limits.max_results,
                result_chars = limits.result_chars,
                timeout_secs = limits.timeout.as_secs(),
                max_tps = limits.max_tps,
                "web search backend: Amazon Bedrock AgentCore Web Search"
            );
            Some(Arc::new(client))
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                "AgentCore web search could not be set up; web search stays off"
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "agentcore_search_tests.rs"]
mod tests;
