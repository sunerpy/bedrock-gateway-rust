//! Tests for the AgentCore Web Search client (sidecar convention, see
//! `AGENTS.md`). Every test talks to a local `wiremock` server or calls pure
//! functions; none reaches AWS except the `#[ignore]`d live test at the end.

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::future::ProvideCredentials as ProvideCredentialsFuture;
use aws_smithy_types::date_time::Format;
use tracing_subscriber::fmt::MakeWriter;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::*;

const INVOKE_PATH: &str = "/web-search/web-search.v1/invoke";
const ACCESS_KEY: &str = "AKIDEXAMPLE";
const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
const SESSION_TOKEN: &str = "session-token-example";

fn test_limits() -> WebSearchLimits {
    WebSearchLimits {
        max_calls: 5,
        max_results: 5,
        result_chars: 1500,
        timeout: Duration::from_secs(5),
        max_tps: 100,
    }
}

fn static_credentials(session_token: Option<&str>) -> SharedCredentialsProvider {
    SharedCredentialsProvider::new(Credentials::new(
        ACCESS_KEY,
        SECRET_KEY,
        session_token.map(str::to_string),
        None,
        "test",
    ))
}

/// A client for `server` whose retry waits milliseconds and ignores the
/// remaining budget, so retry tests run fast.
fn client(
    server: &MockServer,
    credentials: SharedCredentialsProvider,
    limits: WebSearchLimits,
) -> AgentCoreWebSearch {
    let mut client = AgentCoreWebSearch::new(
        "us-east-2".to_string(),
        Some(server.uri()),
        credentials,
        &limits,
    )
    .expect("client builds");
    client.retry = RetryPolicy {
        min_budget: Duration::ZERO,
        default_delay: Duration::from_millis(5),
        max_delay: Duration::from_millis(20),
    };
    client
}

fn ok_response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("x-amzn-requestid", "req-123")
        .set_body_json(json!({
            "results": [{
                "title": "Release highlights",
                "url": "https://docs.example.net/releases/highlights",
                "text": "The January release adds support for scheduled reports.",
                "publishedAt": "2026-01-15T17:00:00Z"
            }]
        }))
}

async fn received(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("request recording is on")
}

/// Search once against a server that always answers `status` with `body` and
/// the `x-amzn-errortype` header when given; checks how many calls were made.
async fn search_against(
    status: u16,
    error_type_header: Option<&str>,
    body: Value,
    expected_calls: u64,
) -> Result<SearchOutcome, SearchError> {
    let server = MockServer::start().await;
    let mut response = ResponseTemplate::new(status).set_body_json(body);
    if let Some(error_type) = error_type_header {
        response = response.insert_header("x-amzn-errortype", error_type);
    }
    Mock::given(method("POST"))
        .and(path(INVOKE_PATH))
        .respond_with(response)
        .expect(expected_calls)
        .mount(&server)
        .await;
    client(&server, static_credentials(None), test_limits())
        .search("aws news", 5)
        .await
}

/// `x-amz-date` (`YYYYMMDDTHHMMSSZ`) as a `SystemTime`.
fn parse_amz_date(amz_date: &str) -> SystemTime {
    let rfc3339 = format!(
        "{}-{}-{}T{}:{}:{}Z",
        &amz_date[0..4],
        &amz_date[4..6],
        &amz_date[6..8],
        &amz_date[9..11],
        &amz_date[11..13],
        &amz_date[13..15]
    );
    let date_time =
        aws_smithy_types::DateTime::from_str(&rfc3339, Format::DateTime).expect("x-amz-date");
    SystemTime::try_from(date_time).expect("representable time")
}

/// The `authorization` value SigV4 gives for what `request` carried on the
/// wire, signed at its own `x-amz-date`.
fn resign(request: &Request, session_token: Option<&str>) -> String {
    let header = |name: &str| {
        request
            .headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .expect(name)
    };
    let identity = Identity::from(Credentials::new(
        ACCESS_KEY,
        SECRET_KEY,
        session_token.map(str::to_string),
        None,
        "test",
    ));
    let params: SigningParams<'_> = v4::SigningParams::builder()
        .identity(&identity)
        .region("us-east-2")
        .name("bedrock-agentcore")
        .time(parse_amz_date(header("x-amz-date")))
        .settings(SigningSettings::default())
        .build()
        .expect("signing params")
        .into();
    let signable = SignableRequest::new(
        "POST",
        request.url.as_str(),
        [
            ("content-type", header("content-type")),
            ("host", header("host")),
        ]
        .into_iter(),
        SignableBody::Bytes(&request.body),
    )
    .expect("signable request");
    let (instructions, _) = sign(signable, &params).expect("signs").into_parts();
    let (headers, _) = instructions.into_parts();
    headers
        .into_iter()
        .find(|header| header.name() == "authorization")
        .map(|header| header.value().to_string())
        .expect("authorization header")
}

#[tokio::test]
async fn signs_with_sigv4_for_bedrock_agentcore() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(INVOKE_PATH))
        .respond_with(ok_response())
        .expect(2)
        .mount(&server)
        .await;
    let tokens = [Some(SESSION_TOKEN), None];
    for token in tokens {
        client(&server, static_credentials(token), test_limits())
            .search("aws news", 5)
            .await
            .expect("search succeeds");
    }

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2);
    let authority = server.uri().trim_start_matches("http://").to_string();
    for (request, token) in requests.iter().zip(tokens) {
        let header = |name: &str| {
            request
                .headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        };
        assert_eq!(request.method.as_str(), "POST");
        assert_eq!(request.url.path(), INVOKE_PATH);
        assert_eq!(header("content-type").as_deref(), Some("application/json"));
        assert_eq!(header("host").as_deref(), Some(authority.as_str()));
        assert_eq!(header("x-amz-security-token").as_deref(), token);

        let amz_date = header("x-amz-date").expect("x-amz-date");
        assert_eq!(amz_date.len(), 16, "{amz_date}");
        assert_eq!(&amz_date[8..9], "T");
        assert!(amz_date.ends_with('Z'));

        let authorization = header("authorization").expect("authorization");
        let signed_headers = if token.is_some() {
            "content-type;host;x-amz-date;x-amz-security-token"
        } else {
            "content-type;host;x-amz-date"
        };
        let prefix = format!(
            "AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/{}/us-east-2/bedrock-agentcore/aws4_request, SignedHeaders={signed_headers}, Signature=",
            &amz_date[..8]
        );
        assert!(authorization.starts_with(&prefix), "{authorization}");
        let signature = &authorization[prefix.len()..];
        assert_eq!(signature.len(), 64);
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));

        // The signature covers what went on the wire: re-signing the received
        // URL, host, content type and body at the same time gives it back.
        assert_eq!(authorization, resign(request, token));
    }
}

#[tokio::test]
async fn body_has_only_search_query_and_max_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(INVOKE_PATH))
        .respond_with(ok_response())
        .expect(3)
        .mount(&server)
        .await;
    let client = client(&server, static_credentials(None), test_limits());
    for max_results in [7, 40, 0] {
        client
            .search("aws news 2026", max_results)
            .await
            .expect("search succeeds");
    }

    let bodies: Vec<Value> = received(&server)
        .await
        .iter()
        .map(|request| request.body_json().expect("JSON body"))
        .collect();
    assert_eq!(
        bodies,
        vec![
            json!({ "searchQuery": "aws news 2026", "maxResults": 7 }),
            // AgentCore takes 1-25.
            json!({ "searchQuery": "aws news 2026", "maxResults": 25 }),
            json!({ "searchQuery": "aws news 2026", "maxResults": 1 }),
        ]
    );

    // A blank query never leaves the process.
    assert_eq!(client.search("  ", 5).await, Err(SearchError::InvalidInput));
    assert_eq!(received(&server).await.len(), 3);
}

#[tokio::test]
async fn parses_results_with_optional_fields() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(INVOKE_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-amzn-requestid", "req-abc")
                .set_body_json(json!({
                    "results": [
                        {
                            "title": "One",
                            "url": "https://a.example/1",
                            "text": "first",
                            "publishedAt": "2026-01-15T17:00:00Z"
                        },
                        { "text": "second" },
                        { "title": "no text" },
                        { "text": "", "url": "https://a.example/3" },
                        { "text": 5 },
                        { "text": "sixth", "title": null, "url": 7 }
                    ],
                    "nextToken": "ignored"
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let outcome = client(&server, static_credentials(None), test_limits())
        .search("q", 5)
        .await
        .expect("search succeeds");

    let plain = |text: &str| SearchResult {
        title: None,
        url: None,
        text: text.to_string(),
        published_at: None,
    };
    assert_eq!(
        outcome,
        SearchOutcome {
            results: vec![
                SearchResult {
                    title: Some("One".to_string()),
                    url: Some("https://a.example/1".to_string()),
                    text: "first".to_string(),
                    published_at: Some("2026-01-15T17:00:00Z".to_string()),
                },
                plain("second"),
                plain("sixth"),
            ],
            request_id: Some("req-abc".to_string()),
        }
    );
}

#[tokio::test]
async fn malformed_success_body_is_unavailable_without_retry() {
    for body in [
        json!({ "items": [] }),
        json!({ "results": "none" }),
        json!([]),
    ] {
        assert_eq!(
            search_against(200, None, body, 1).await,
            Err(SearchError::Unavailable)
        );
    }
}

#[tokio::test]
async fn maps_validation_403_404_to_codes() {
    let cases: [(u16, Option<&str>, Value, u64, SearchError); 9] = [
        (
            400,
            Some("ValidationException:http://internal.amazon.com/coral/com.amazon.coral.validate/"),
            json!({ "message": "bad query", "reason": "FieldValidationFailed" }),
            1,
            SearchError::InvalidInput,
        ),
        (
            403,
            None,
            json!({ "__type": "com.amazonaws.bedrockagentcore#AccessDeniedException", "message": "denied" }),
            1,
            SearchError::Unavailable,
        ),
        (
            403,
            Some("UnrecognizedClientException"),
            json!({ "message": "bad token" }),
            1,
            SearchError::Unavailable,
        ),
        (
            404,
            None,
            json!({ "message": "no such tool" }),
            1,
            SearchError::Unavailable,
        ),
        // A credential problem reported as 400 is not the query's fault.
        (
            400,
            Some("ExpiredTokenException"),
            json!({ "message": "expired" }),
            1,
            SearchError::Unavailable,
        ),
        (
            400,
            None,
            json!("not an error object"),
            1,
            SearchError::InvalidInput,
        ),
        // Retryable statuses are tried twice before the code is returned.
        (
            429,
            Some("ThrottlingException"),
            json!({ "message": "slow down" }),
            2,
            SearchError::Throttled,
        ),
        (
            500,
            Some("InternalServerException"),
            json!({ "message": "oops" }),
            2,
            SearchError::Unavailable,
        ),
        (
            502,
            None,
            json!({ "message": "bad gateway" }),
            1,
            SearchError::Unavailable,
        ),
    ];
    for (status, header, body, calls, expected) in cases {
        assert_eq!(
            search_against(status, header, body, calls).await,
            Err(expected),
            "status {status}, type {header:?}"
        );
    }
}

#[test]
fn error_types_and_statuses_classify() {
    assert_eq!(
        sanitize_error_code("ValidationException:http://internal.amazon.com/coral/"),
        "ValidationException"
    );
    assert_eq!(
        sanitize_error_code("com.amazonaws.bedrockagentcore#ThrottlingException"),
        "ThrottlingException"
    );
    assert_eq!(
        sanitize_error_code("aws.protocols#ServiceUnavailableException:http://x"),
        "ServiceUnavailableException"
    );
    assert_eq!(
        sanitize_error_code("AccessDeniedException"),
        "AccessDeniedException"
    );
    assert_eq!(
        log_token("AccessDeniedException"),
        Some("AccessDeniedException".to_string())
    );
    assert_eq!(
        log_token("1b6f6b44-5f0b-4c43-9d0e-2a1f4b9c7e11"),
        Some("1b6f6b44-5f0b-4c43-9d0e-2a1f4b9c7e11".to_string())
    );
    for not_a_token in ["", "denied: see the query", "a\nb", &"x".repeat(129)] {
        assert_eq!(log_token(not_a_token), None, "{not_a_token:?}");
    }

    assert_eq!(
        classify_failure(400, Some("ValidationException")),
        SearchError::InvalidInput
    );
    assert_eq!(classify_failure(400, None), SearchError::InvalidInput);
    assert_eq!(
        classify_failure(400, Some("SerializationException")),
        SearchError::Unavailable
    );
    assert_eq!(
        classify_failure(429, Some("ServiceQuotaExceededException")),
        SearchError::Throttled
    );
    assert_eq!(
        classify_failure(400, Some("ThrottlingException")),
        SearchError::Throttled
    );
    for status in [403, 404, 500, 503, 504] {
        assert_eq!(classify_failure(status, None), SearchError::Unavailable);
    }
    for status in [429, 500, 503] {
        assert!(is_retryable_status(status), "{status}");
    }
    for status in [400, 403, 404, 502, 504] {
        assert!(!is_retryable_status(status), "{status}");
    }
}

#[tokio::test]
async fn retries_429_and_503_once() {
    // 429, then success.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429).insert_header("x-amzn-errortype", "ThrottlingException"),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ok_response())
        .mount(&server)
        .await;
    let outcome = client(&server, static_credentials(None), test_limits())
        .search("q", 5)
        .await
        .expect("the retry succeeds");
    assert_eq!(outcome.request_id.as_deref(), Some("req-123"));
    assert_eq!(received(&server).await.len(), 2);

    // 503 asking for 3 s: the wait is capped by the policy, then success.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(503)
                .set_body_json(json!({ "message": "busy", "retryAfterSeconds": 3 })),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ok_response())
        .mount(&server)
        .await;
    let started = std::time::Instant::now();
    client(&server, static_credentials(None), test_limits())
        .search("q", 5)
        .await
        .expect("the retry succeeds");
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(received(&server).await.len(), 2);

    // 503 every time: exactly one retry.
    assert_eq!(
        search_against(503, None, json!({ "message": "busy" }), 2).await,
        Err(SearchError::Unavailable)
    );

    // No retry when less than the minimum budget remains (default policy:
    // 2 s, here with a 1 s budget).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({ "message": "busy" })))
        .expect(1)
        .mount(&server)
        .await;
    let short = WebSearchLimits {
        timeout: Duration::from_secs(1),
        ..test_limits()
    };
    let mut client = client(&server, static_credentials(None), short);
    client.retry = RetryPolicy::DEFAULT;
    assert_eq!(client.search("q", 5).await, Err(SearchError::Unavailable));
}

#[test]
fn retry_waits_for_retry_after_up_to_one_second() {
    let policy = RetryPolicy::DEFAULT;
    assert_eq!(policy.min_budget, Duration::from_secs(2));
    assert_eq!(policy.delay(None), Duration::from_millis(250));
    assert_eq!(policy.delay(Some(Duration::ZERO)), Duration::ZERO);
    assert_eq!(
        policy.delay(Some(Duration::from_millis(500))),
        Duration::from_millis(500)
    );
    assert_eq!(
        policy.delay(Some(Duration::from_secs(3))),
        Duration::from_secs(1)
    );

    assert_eq!(
        retry_after(&json!({ "retryAfterSeconds": 2 })),
        Some(Duration::from_secs(2))
    );
    assert_eq!(retry_after(&json!({ "retryAfterSeconds": -1 })), None);
    assert_eq!(retry_after(&json!({ "message": "busy" })), None);
}

#[tokio::test]
async fn transport_failure_is_unavailable_after_one_retry() {
    // Nothing listens on a port whose listener was closed.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    drop(listener);
    let mut client = AgentCoreWebSearch::new(
        "us-east-2".to_string(),
        Some(format!("http://127.0.0.1:{port}")),
        static_credentials(None),
        &test_limits(),
    )
    .expect("client builds");
    client.retry = RetryPolicy {
        min_budget: Duration::ZERO,
        default_delay: Duration::from_millis(5),
        max_delay: Duration::from_millis(5),
    };
    let (logs, _guard) = capture_logs();
    assert_eq!(client.search("q", 5).await, Err(SearchError::Unavailable));
    let rendered = logs.rendered();
    assert!(rendered.contains("\"reason\":\"transport\""), "{rendered}");
    assert!(
        rendered.contains("\"attempt\":2"),
        "the retry ran: {rendered}"
    );
}

#[tokio::test]
async fn times_out_within_budget() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok_response().set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;
    let limits = WebSearchLimits {
        timeout: Duration::from_millis(300),
        ..test_limits()
    };
    let started = std::time::Instant::now();
    let result = client(&server, static_credentials(None), limits)
        .search("q", 5)
        .await;
    assert_eq!(result, Err(SearchError::Unavailable));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
}

fn assert_wait(reservation: Reservation, expected: Duration) {
    match reservation {
        Reservation::Wait(wait) => assert!(
            (wait.as_secs_f64() - expected.as_secs_f64()).abs() < 1e-6,
            "waited {wait:?}, expected {expected:?}"
        ),
        other => panic!("expected a wait of {expected:?}, got {other:?}"),
    }
}

#[test]
fn bucket_reserve_math() {
    let second = Duration::from_secs(1);
    let start = Instant::now();
    let mut state = BucketState::full(5, start);

    // A full bucket serves a burst of five at once.
    for _ in 0..5 {
        assert_eq!(reserve(&mut state, start, second), Reservation::Ready);
    }
    // Then each caller reserves the next token, 200 ms after the previous.
    for n in 1..=5u64 {
        assert_wait(
            reserve(&mut state, start, second),
            Duration::from_millis(200 * n),
        );
    }
    // The sixth waiter would need 1.2 s: rejected, and nothing is taken.
    assert_eq!(reserve(&mut state, start, second), Reservation::Rejected);
    assert!((state.tokens + 5.0).abs() < 1e-9, "{}", state.tokens);

    // One second later the reserved tokens are paid off.
    let later = start + second;
    assert_wait(
        reserve(&mut state, later, second),
        Duration::from_millis(200),
    );

    // After a long idle the bucket is full again, never above the burst.
    let idle = later + Duration::from_secs(60);
    for _ in 0..5 {
        assert_eq!(reserve(&mut state, idle, second), Reservation::Ready);
    }
    assert_wait(
        reserve(&mut state, idle, second),
        Duration::from_millis(200),
    );
}

#[test]
fn rejects_when_wait_exceeds_max() {
    let start = Instant::now();
    let max_wait = Duration::from_millis(500);
    let mut state = BucketState::full(1, start);
    assert_eq!(reserve(&mut state, start, max_wait), Reservation::Ready);
    assert_eq!(reserve(&mut state, start, max_wait), Reservation::Rejected);
    assert_wait(
        reserve(&mut state, start + Duration::from_millis(600), max_wait),
        Duration::from_millis(400),
    );
    assert_eq!(
        reserve(&mut state, start + Duration::from_millis(600), max_wait),
        Reservation::Rejected
    );

    // Zero is treated as one search per second.
    let rate = BucketState::full(0, start).rate;
    assert!((rate - 1.0).abs() < f64::EPSILON, "{rate}");
}

#[tokio::test]
async fn acquire_sleeps_for_its_reserved_token() {
    // Two per second: the third caller holds the token due in 500 ms.
    let bucket = TokenBucket::new(2);
    assert!(bucket.acquire(Duration::ZERO).await);
    assert!(bucket.acquire(Duration::ZERO).await);
    assert!(!bucket.acquire(Duration::from_millis(100)).await);
    let started = std::time::Instant::now();
    assert!(bucket.acquire(Duration::from_secs(1)).await);
    assert!(
        started.elapsed() >= Duration::from_millis(400),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn rate_limited_search_fails_without_calling_aws() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok_response())
        .expect(1)
        .mount(&server)
        .await;
    let limits = WebSearchLimits {
        max_tps: 1,
        ..test_limits()
    };
    let mut client = client(&server, static_credentials(None), limits);
    client.token_max_wait = Duration::from_millis(10);
    client.search("q", 5).await.expect("the first search runs");
    assert_eq!(client.search("q", 5).await, Err(SearchError::Throttled));
}

/// Counts provider calls; serves credentials with a fixed expiry and fails
/// every call after the first `fail_after`.
#[derive(Debug)]
struct CountingProvider {
    calls: Arc<AtomicUsize>,
    expiry: Option<SystemTime>,
    fail_after: Option<usize>,
}

impl ProvideCredentials for CountingProvider {
    fn provide_credentials<'a>(&'a self) -> ProvideCredentialsFuture<'a>
    where
        Self: 'a,
    {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_after.is_some_and(|after| call > after) {
            return ProvideCredentialsFuture::ready(Err(CredentialsError::not_loaded(
                "no credentials in this test",
            )));
        }
        ProvideCredentialsFuture::ready(Ok(Credentials::new(
            ACCESS_KEY,
            SECRET_KEY,
            None,
            self.expiry,
            "counting",
        )))
    }
}

fn counting_cache(
    expiry: Option<SystemTime>,
    fail_after: Option<usize>,
) -> (CredentialCache, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = SharedCredentialsProvider::new(CountingProvider {
        calls: calls.clone(),
        expiry,
        fail_after,
    });
    (CredentialCache::new(provider), calls)
}

#[tokio::test]
async fn credentials_cached_until_near_expiry() {
    let now = SystemTime::now();

    // Valid for an hour: one provider call serves every search.
    let (cache, calls) = counting_cache(Some(now + Duration::from_secs(3600)), None);
    for _ in 0..3 {
        assert!(cache.get().await.is_some());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Without an expiry: cached for good.
    let (cache, calls) = counting_cache(None, None);
    for _ in 0..3 {
        assert!(cache.get().await.is_some());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Inside the five-minute window: refreshed on every use.
    let (cache, calls) = counting_cache(Some(now + Duration::from_secs(240)), None);
    for _ in 0..3 {
        assert!(cache.get().await.is_some());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);

    // A failed refresh keeps using credentials that have not expired yet...
    let (cache, calls) = counting_cache(Some(now + Duration::from_secs(240)), Some(1));
    assert!(cache.get().await.is_some());
    assert!(cache.get().await.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // ...but never expired ones.
    let (cache, _) = counting_cache(Some(now - Duration::from_secs(1)), Some(1));
    assert!(cache.get().await.is_some(), "fresh from the provider");
    assert!(cache.get().await.is_none());
}

#[tokio::test]
async fn credential_failure_is_unavailable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ok_response())
        .expect(0)
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let failing = SharedCredentialsProvider::new(CountingProvider {
        calls: calls.clone(),
        expiry: None,
        fail_after: Some(0),
    });
    let client = client(&server, failing, test_limits());
    assert_eq!(client.search("q", 5).await, Err(SearchError::Unavailable));
    // A local failure is not retried.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

struct CapturedLogWriter(Arc<Mutex<Vec<u8>>>);

impl CapturedLogs {
    fn rendered(&self) -> String {
        String::from_utf8(self.0.lock().expect("log lock").clone()).expect("UTF-8 logs")
    }
}

impl<'a> MakeWriter<'a> for CapturedLogs {
    type Writer = CapturedLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        CapturedLogWriter(Arc::clone(&self.0))
    }
}

impl Write for CapturedLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("log lock poisoned"))?
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Every event of every target at TRACE, on this thread, as JSON lines.
fn capture_logs() -> (CapturedLogs, tracing::subscriber::DefaultGuard) {
    let logs = CapturedLogs::default();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_max_level(tracing::Level::TRACE)
        .with_writer(logs.clone())
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);
    (logs, guard)
}

#[tokio::test]
async fn logs_never_contain_the_query_results_or_error_messages() {
    let (logs, _guard) = capture_logs();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(403)
                .insert_header("x-amzn-errortype", "AccessDeniedException")
                .insert_header("x-amzn-requestid", "req-denied")
                .set_body_json(json!({ "message": "SECRET-ERROR-MESSAGE" })),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "text": "SECRET-RESULT-TEXT", "title": "SECRET-RESULT-TITLE" }]
        })))
        .mount(&server)
        .await;
    let client = client(
        &server,
        static_credentials(Some(SESSION_TOKEN)),
        test_limits(),
    );

    assert_eq!(
        client.search("SECRET-QUERY-MARKER", 5).await,
        Err(SearchError::Unavailable)
    );
    client
        .search("SECRET-QUERY-MARKER", 5)
        .await
        .expect("the second search succeeds");

    let rendered = logs.rendered();
    assert!(rendered.contains("AccessDeniedException"), "{rendered}");
    assert!(rendered.contains("req-denied"), "{rendered}");
    assert!(rendered.contains("\"level\":\"ERROR\""), "{rendered}");
    for secret in [
        "SECRET-QUERY-MARKER",
        "SECRET-RESULT-TEXT",
        "SECRET-RESULT-TITLE",
        "SECRET-ERROR-MESSAGE",
        SECRET_KEY,
        SESSION_TOKEN,
    ] {
        assert!(
            !rendered.contains(secret),
            "{secret} leaked into: {rendered}"
        );
    }
}

fn settings(backend: &str) -> AppSettings {
    AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-west-2".to_string(),
        default_model: "model".to_string(),
        default_embedding_model: "embedding".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching: false,
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
        api_key: None,
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        disable_tool_search: false,
        web_search_backend: backend.to_string(),
        web_search_region: None,
        web_search_max_calls: 5,
        web_search_timeout_secs: 12,
        web_search_max_results: 5,
        web_search_result_chars: 1500,
        web_search_max_tps: 5,
        bind_addr: "127.0.0.1".to_string(),
        port: 8080,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        responses_stream_idle_timeout_secs: 180,
        model_catalog_refresh_secs: 0,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    }
}

fn sdk_config(with_credentials: bool) -> aws_config::SdkConfig {
    let builder = aws_config::SdkConfig::builder().region(aws_config::Region::new("us-west-2"));
    if with_credentials {
        builder
            .credentials_provider(static_credentials(None))
            .build()
    } else {
        builder.build()
    }
}

#[test]
fn resolve_off_by_default() {
    for backend in ["off", " OFF ", ""] {
        assert!(
            resolve_web_search_backend(&settings(backend), &sdk_config(true)).is_none(),
            "{backend:?}"
        );
    }
}

#[test]
fn unknown_backend_warns_and_stays_off() {
    let (logs, _guard) = capture_logs();
    assert!(resolve_web_search_backend(&settings("bing"), &sdk_config(true)).is_none());
    let rendered = logs.rendered();
    assert!(
        rendered.contains("WEB_SEARCH_BACKEND is neither off nor agentcore"),
        "{rendered}"
    );
    assert!(rendered.contains("\"level\":\"WARN\""), "{rendered}");
}

#[test]
fn agentcore_resolves_with_sigv4_credentials_only() {
    let (logs, _guard) = capture_logs();

    let mut in_eu = settings(" AgentCore ");
    in_eu.web_search_region = Some("eu-west-1".to_string());
    assert!(resolve_web_search_backend(&in_eu, &sdk_config(true)).is_some());
    let rendered = logs.rendered();
    assert!(
        rendered.contains("\"region\":\"eu-west-1\""),
        "web_search_region wins over aws_region: {rendered}"
    );

    // Without a credentials provider there is nothing to sign with.
    assert!(resolve_web_search_backend(&settings("agentcore"), &sdk_config(false)).is_none());
    assert!(logs.rendered().contains("needs SigV4 credentials"));

    // A Bedrock API key does not stop SigV4; it only earns a note.
    let mut with_key = settings("agentcore");
    with_key.bedrock_api_key = Some("bedrock-api-key".to_string());
    assert!(resolve_web_search_backend(&with_key, &sdk_config(true)).is_some());
    let rendered = logs.rendered();
    assert!(rendered.contains("the Bedrock API key is not used for it"));
    assert!(!rendered.contains("bedrock-api-key"));

    // A region that cannot form the endpoint keeps web search off.
    let mut bad_region = settings("agentcore");
    bad_region.web_search_region = Some("us east 2".to_string());
    assert!(resolve_web_search_backend(&bad_region, &sdk_config(true)).is_none());
    assert!(logs.rendered().contains("could not be set up"));
}

#[test]
fn limits_are_clamped() {
    let (logs, _guard) = capture_logs();
    let defaults = WebSearchLimits::from_settings(&settings("agentcore"));
    assert_eq!(
        defaults,
        WebSearchLimits {
            max_calls: 5,
            max_results: 5,
            result_chars: 1500,
            timeout: Duration::from_secs(12),
            max_tps: 5,
        }
    );
    assert!(!logs.rendered().contains("out of range"));

    let mut low = settings("agentcore");
    low.web_search_max_calls = 0;
    low.web_search_timeout_secs = 0;
    low.web_search_max_results = 0;
    low.web_search_result_chars = 5;
    low.web_search_max_tps = 0;
    assert_eq!(
        WebSearchLimits::from_settings(&low),
        WebSearchLimits {
            max_calls: 1,
            max_results: 1,
            result_chars: 100,
            timeout: Duration::from_secs(1),
            max_tps: 1,
        }
    );

    let mut high = settings("agentcore");
    high.web_search_max_calls = 50;
    high.web_search_timeout_secs = 3600;
    high.web_search_max_results = 100;
    high.web_search_result_chars = 1_000_000;
    high.web_search_max_tps = 40;
    assert_eq!(
        WebSearchLimits::from_settings(&high),
        WebSearchLimits {
            max_calls: 50,
            max_results: 25,
            result_chars: 20_000,
            timeout: Duration::from_secs(60),
            max_tps: 40,
        }
    );

    let rendered = logs.rendered();
    for setting in [
        "WEB_SEARCH_MAX_CALLS",
        "WEB_SEARCH_TIMEOUT_SECS",
        "WEB_SEARCH_MAX_RESULTS",
        "WEB_SEARCH_RESULT_CHARS",
        "WEB_SEARCH_MAX_TPS",
    ] {
        assert!(rendered.contains(setting), "{setting}: {rendered}");
    }
}

/// L1 (design §11.6): one real `InvokeWebSearch` call in us-east-2. Run with
/// `env -u AWS_REGION -u AWS_DEFAULT_REGION -u AWS_ACCESS_KEY_ID
/// -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN BEDROCK_INTEGRATION=1
/// AWS_PROFILE=<profile> cargo test --lib web_search -- --ignored`; the
/// profile needs `bedrock-agentcore:InvokeWebSearch`.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_invoke_web_search_us_east_2() {
    if std::env::var("BEDROCK_INTEGRATION").is_err() {
        return;
    }
    let aws = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new("us-east-2"))
        .load()
        .await;
    let credentials = aws.credentials_provider().expect("credential chain");
    let limits = WebSearchLimits {
        max_results: 3,
        timeout: Duration::from_secs(12),
        max_tps: 1,
        ..test_limits()
    };
    let backend = AgentCoreWebSearch::new("us-east-2".to_string(), None, credentials, &limits)
        .expect("client builds");

    let outcome = backend
        .search("Amazon Bedrock AgentCore", limits.max_results)
        .await
        .expect("live search succeeds");

    assert!(!outcome.results.is_empty());
    assert!(outcome.results.len() <= 3);
    assert!(outcome.results.iter().all(|result| !result.text.is_empty()));
    assert!(outcome.request_id.is_some());
}
