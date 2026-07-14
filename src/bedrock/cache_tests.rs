//! Tests for [`crate::bedrock::cache`] — prompt-caching `cachePoint` insertion.
//!
//! Test code lives in this sibling file (Option A: no inline `#[cfg(test)]`
//! blocks in the source module). `use super::*;` resolves to the `cache`
//! implementation module via the `#[path = "cache_tests.rs"] mod tests;`
//! declaration there.

use super::*;
use crate::bedrock::capabilities::ConfigModelCapabilities;
use crate::config::ModelCapabilityConfig;
use proptest::prelude::*;

const MODELS_TOML: &str = "config/models.toml";

// Nova has max_cache_tokens=20000 configured ⇒ supports caching.
const NOVA: &str = "us.amazon.nova-pro-v1:0";
// Claude opus-4-8 has NO cache params in the shipped config ⇒ per our
// config-driven gate, it does NOT "support caching" today. This is correct
// de-hardcoded behavior: caching support is a TOML edit, not code.
const CLAUDE_NO_CACHE: &str = "global.anthropic.claude-opus-4-8-20251101-v1:0";

fn caps() -> ConfigModelCapabilities {
    let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load models.toml");
    ConfigModelCapabilities::new(config)
}

/// A test-only capabilities impl with tunable cache params so the gating
/// logic can be exercised without depending on shipped TOML values. The
/// checkpoint ceiling is NOT a trait accessor — it is passed to
/// `decorate_messages` directly — so it is stored separately and read via
/// [`FakeCaps::checkpoints`].
struct FakeCaps {
    min: Option<u32>,
    max: Option<u32>,
    checkpoints: Option<u32>,
}

impl ModelCapabilities for FakeCaps {
    fn has(&self, _model: &str, _cap: crate::domain::Capability) -> bool {
        false
    }
    fn resolve_foundation(&self, m: &str) -> String {
        m.to_string()
    }
    fn budget_ratios(&self, _model: &str) -> Option<crate::domain::BudgetRatios> {
        None
    }
    fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
        self.max
    }
    fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
        self.min
    }
    fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
        self.checkpoints
    }
    fn beta_headers(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }
    fn reasoning_path(&self, _model: &str) -> crate::domain::ReasoningPath {
        crate::domain::ReasoningPath::None
    }
    fn responses_backend(&self, _model: &str) -> crate::domain::ResponsesBackend {
        crate::domain::ResponsesBackend::Converse
    }
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
    }
    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        None
    }
}

impl FakeCaps {
    fn supports(min: Option<u32>, max: Option<u32>, checkpoints: Option<u32>) -> Self {
        Self {
            min,
            max,
            checkpoints,
        }
    }

    fn checkpoints(&self) -> Option<u32> {
        self.checkpoints
    }
}

fn sys(texts: &[&str]) -> Value {
    Value::Array(texts.iter().map(|t| json!({ "text": t })).collect())
}

fn long_text(words: usize) -> String {
    vec!["word"; words].join(" ")
}

// ---- PromptCachingControl: parse + strip ----

#[test]
fn control_extract_strips_field_and_parses_flags() {
    let mut extra = Some(json!({
        "thinking": {"type": "enabled"},
        "prompt_caching": {"system": true, "messages": false}
    }));
    let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
    assert_eq!(ctrl.system, Some(true));
    assert_eq!(ctrl.messages, Some(false));

    // CRITICAL: the control field is removed and never reaches Bedrock.
    let obj = extra.expect("extra_body retained");
    assert!(
        obj.get("prompt_caching").is_none(),
        "prompt_caching control field must be stripped"
    );
    // Non-control fields survive for Bedrock passthrough.
    assert!(obj.get("thinking").is_some());
}

#[test]
fn control_strip_when_only_control_field_present() {
    let mut extra = Some(json!({ "prompt_caching": {"system": true} }));
    let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
    assert_eq!(ctrl.system, Some(true));
    // The object remains (now empty) but prompt_caching is gone.
    let obj = extra.expect("extra retained");
    assert!(obj.get("prompt_caching").is_none());
    assert_eq!(obj.as_object().map(|m| m.len()), Some(0));
}

#[test]
fn control_absent_extra_body_is_default() {
    let mut none: Option<Value> = None;
    assert_eq!(
        PromptCachingControl::extract_and_strip(&mut none),
        PromptCachingControl::default()
    );
    // Non-object extra_body: default, untouched.
    let mut arr = Some(json!([1, 2, 3]));
    assert_eq!(
        PromptCachingControl::extract_and_strip(&mut arr),
        PromptCachingControl::default()
    );
}

#[test]
fn control_parse_is_non_mutating() {
    let extra = json!({ "prompt_caching": {"messages": true} });
    let ctrl = PromptCachingControl::parse(Some(&extra));
    assert_eq!(ctrl.messages, Some(true));
    assert_eq!(ctrl.system, None);
    // parse does not strip; the field is still present on the borrowed value.
    assert!(extra.get("prompt_caching").is_some());
}

#[test]
fn control_precedence_per_request_beats_global() {
    // Per-request explicit wins over global default (bedrock.py:744-748).
    let ctrl = PromptCachingControl {
        system: Some(false),
        messages: Some(true),
        ttl: None,
    };
    // Global true, but request disables system.
    assert!(!ctrl.system_enabled(true));
    // Global false, but request enables messages.
    assert!(ctrl.messages_enabled(false));

    // Absent per-request ⇒ fall back to global.
    let empty = PromptCachingControl::default();
    assert!(empty.system_enabled(true));
    assert!(!empty.system_enabled(false));
    assert!(empty.messages_enabled(true));
}

// ---- supports_caching: config-driven capability gate ----

#[test]
fn supports_caching_from_config_nova_yes_claude_yes() {
    let c = caps();
    // Nova has max_cache_tokens=20000 in config ⇒ supported.
    assert!(supports_caching(NOVA, &c));
    // Claude entries now declare cache_min_tokens=1024 in config ⇒ supported.
    assert!(supports_caching(CLAUDE_NO_CACHE, &c));
    // Unknown model ⇒ not supported.
    assert!(!supports_caching("vendor.unknown-v1:0", &c));
}

#[test]
fn nova_max_cache_tokens_read_from_config() {
    // The Nova 20000 ceiling is config-sourced, not hardcoded here.
    let c = caps();
    assert_eq!(c.max_cache_tokens(NOVA), Some(20_000));
}

#[test]
fn nova_cache_floor_applies_from_config() {
    let c = caps();
    assert_eq!(c.cache_min_tokens(NOVA), Some(1024));

    // Below the 1024-token floor (estimate = words * 1.3) ⇒ no cachePoint.
    let small = decorate_system_blocks(sys(&[&long_text(100)]), NOVA, &c, true, None);
    let small_arr = small.as_array().unwrap();
    assert_eq!(small_arr.len(), 1, "below Nova floor ⇒ no cachePoint");
    assert!(small_arr[0].get("cachePoint").is_none());

    // Above the floor ⇒ cachePoint injected.
    let large = decorate_system_blocks(sys(&[&long_text(2000)]), NOVA, &c, true, None);
    let large_arr = large.as_array().unwrap();
    assert_eq!(large_arr.len(), 2, "above Nova floor ⇒ cachePoint");
    assert_eq!(large_arr[1], cache_point(None));
}

// ---- decorate_system_blocks ----

#[test]
fn system_cachepoint_inserted_when_enabled_supported_over_min() {
    // min=4 tokens; supply ~13 estimated tokens (10 words * 1.3).
    let caps = FakeCaps::supports(Some(4), Some(20_000), Some(4));
    let blocks = sys(&[&long_text(10)]);
    let out = decorate_system_blocks(blocks, "m", &caps, true, None);
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2, "text block + cachePoint");
    assert!(arr[0]["text"].is_string());
    assert_eq!(arr[1], cache_point(None));
}

#[test]
fn system_cachepoint_not_inserted_when_under_min() {
    // min very high; a single short word can't clear it.
    let caps = FakeCaps::supports(Some(100_000), None, None);
    let blocks = sys(&["hi"]);
    let out = decorate_system_blocks(blocks, "m", &caps, true, None);
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 1, "below min ⇒ no cachePoint");
    assert!(arr[0].get("cachePoint").is_none());
}

#[test]
fn system_cachepoint_inserted_over_max_warn_only() {
    // Over max_cache_tokens still inserts (Python parity: warn-only).
    let caps = FakeCaps::supports(Some(1), Some(2), Some(4));
    let blocks = sys(&[&long_text(50)]); // ~65 tokens >> max 2
    let out = decorate_system_blocks(blocks, "m", &caps, true, None);
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[1], cache_point(None));
}

#[test]
fn system_not_inserted_for_unsupported_model() {
    // No cache params ⇒ supports_caching == false ⇒ unchanged.
    let caps = FakeCaps::supports(None, None, None);
    let blocks = sys(&[&long_text(100)]);
    let out = decorate_system_blocks(blocks.clone(), "m", &caps, true, None);
    assert_eq!(out, blocks, "unsupported model: no cachePoint");
}

#[test]
fn system_not_inserted_when_disabled() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let blocks = sys(&[&long_text(100)]);
    let out = decorate_system_blocks(blocks.clone(), "m", &caps, false, None);
    assert_eq!(out, blocks, "disabled: no cachePoint");
}

#[test]
fn system_empty_is_unchanged() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let blocks = Value::Array(vec![]);
    let out = decorate_system_blocks(blocks.clone(), "m", &caps, true, None);
    assert_eq!(out, blocks);
}

#[test]
fn system_no_min_configured_inserts_when_enabled() {
    // cache_min_tokens None but max present ⇒ supported, no floor to clear.
    let caps = FakeCaps::supports(None, Some(20_000), Some(4));
    let blocks = sys(&["short"]);
    let out = decorate_system_blocks(blocks, "m", &caps, true, None);
    assert_eq!(out.as_array().unwrap().len(), 2);
}

// ---- decorate_messages ----

fn user_turn(text: &str) -> Value {
    json!({ "role": "user", "content": [{ "text": text }] })
}
fn assistant_turn(text: &str) -> Value {
    json!({ "role": "assistant", "content": [{ "text": text }] })
}

#[test]
fn messages_cachepoint_appended_to_last_user_turn() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![
        user_turn("first"),
        assistant_turn("reply"),
        user_turn("second"),
    ]);
    let out = decorate_messages(messages, "m", &caps, true, 0, caps.checkpoints(), None);
    let turns = out.as_array().unwrap();
    // Last user turn gets the cachePoint appended.
    let last_content = turns[2]["content"].as_array().unwrap();
    assert_eq!(last_content.len(), 2);
    assert_eq!(last_content[1], cache_point(None));
    // Earlier user turn is untouched.
    assert_eq!(turns[0]["content"].as_array().unwrap().len(), 1);
}

#[test]
fn messages_not_inserted_for_unsupported_model() {
    let caps = FakeCaps::supports(None, None, None);
    let messages = Value::Array(vec![user_turn("hi")]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        true,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages);
}

#[test]
fn messages_not_inserted_when_disabled() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![user_turn("hi")]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        false,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages);
}

#[test]
fn messages_respect_checkpoint_ceiling() {
    // already_used == max_cache_checkpoints ⇒ no room.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(2));
    let messages = Value::Array(vec![user_turn("hi")]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        true,
        2,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages, "no room left under the checkpoint ceiling");

    // One under the ceiling ⇒ inserts.
    let out2 = decorate_messages(messages, "m", &caps, true, 1, caps.checkpoints(), None);
    let turns = out2.as_array().unwrap();
    assert_eq!(turns[0]["content"].as_array().unwrap().len(), 2);
}

#[test]
fn messages_skip_tool_result_only_turn() {
    // A user turn whose content is only toolResult must NOT get a cachePoint
    // (bedrock.py:1018-1019). With no other eligible user turn, unchanged.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![
        assistant_turn("calling tool"),
        json!({
            "role": "user",
            "content": [{ "toolResult": { "toolUseId": "t1", "content": [] } }]
        }),
    ]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        true,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages, "toolResult-only turn must be skipped");
}

#[test]
fn messages_pick_prior_user_when_last_is_tool_result() {
    // Last user turn is toolResult-only ⇒ fall back to the prior normal user.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![
        user_turn("real question"),
        assistant_turn("calling tool"),
        json!({
            "role": "user",
            "content": [{ "toolResult": { "toolUseId": "t1", "content": [] } }]
        }),
    ]);
    let out = decorate_messages(messages, "m", &caps, true, 0, caps.checkpoints(), None);
    let turns = out.as_array().unwrap();
    // The normal user turn (index 0) receives the cachePoint.
    let first = turns[0]["content"].as_array().unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(first[1], cache_point(None));
    // The toolResult-only turn is untouched.
    assert_eq!(turns[2]["content"].as_array().unwrap().len(), 1);
}

#[test]
fn messages_no_double_cachepoint() {
    // A user turn already ending in a cachePoint is not eligible again.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![json!({
        "role": "user",
        "content": [{ "text": "hi" }, cache_point(None)]
    })]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        true,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages, "must not append a second cachePoint");
}

#[test]
fn messages_empty_unchanged() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![]);
    let out = decorate_messages(
        messages.clone(),
        "m",
        &caps,
        true,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(out, messages);
}

// ---- combined: control parse → strip → decorate, asserting absence ----

#[test]
fn end_to_end_control_stripped_and_decoration_applied() {
    // Simulate the gateway flow: parse+strip the control field, then use the
    // resolved decision to decorate. Assert the control field is ABSENT from
    // the (Bedrock-bound) extra_body afterwards.
    let mut extra = Some(json!({
        "anthropic_beta": ["x"],
        "prompt_caching": {"system": true, "messages": true}
    }));
    let ctrl = PromptCachingControl::extract_and_strip(&mut extra);

    // Control field gone — would-be Bedrock args do not carry it.
    let bedrock_extra = extra.unwrap();
    assert!(bedrock_extra.get("prompt_caching").is_none());
    assert!(bedrock_extra.get("anthropic_beta").is_some());

    // Global default false, but per-request enables both.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let system = decorate_system_blocks(
        sys(&[&long_text(10)]),
        "m",
        &caps,
        ctrl.system_enabled(false),
        None,
    );
    assert_eq!(system.as_array().unwrap().len(), 2);

    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &caps,
        ctrl.messages_enabled(false),
        1, // one checkpoint already spent on system
        caps.checkpoints(),
        None,
    );
    assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
}

// ---- T4: master-switch default semantics (provider.rs global_default) ----

fn resolved(ctrl: PromptCachingControl, master_on: bool) -> (bool, bool) {
    (
        ctrl.system_enabled(master_on),
        ctrl.messages_enabled(master_on),
    )
}

#[test]
fn t4_master_on_no_extra_body_auto_injects_system_and_messages() {
    let ctrl = PromptCachingControl::parse(None);
    let (sys_on, msg_on) = resolved(ctrl, true);

    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, sys_on, None);
    assert_eq!(
        system.as_array().unwrap().len(),
        2,
        "master on + no extra_body ⇒ system cachePoint"
    );

    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &caps,
        msg_on,
        count_cp(&system),
        caps.checkpoints(),
        None,
    );
    assert_eq!(
        messages[0]["content"].as_array().unwrap().len(),
        2,
        "master on + no extra_body ⇒ messages cachePoint"
    );
}

#[test]
fn t4_extra_body_messages_false_disables_messages_only() {
    let extra = json!({ "prompt_caching": { "messages": false } });
    let ctrl = PromptCachingControl::parse(Some(&extra));
    let (sys_on, msg_on) = resolved(ctrl, true);

    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, sys_on, None);
    assert_eq!(
        system.as_array().unwrap().len(),
        2,
        "system still cached (override only touches messages)"
    );

    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &caps,
        msg_on,
        count_cp(&system),
        caps.checkpoints(),
        None,
    );
    assert_eq!(
        messages[0]["content"].as_array().unwrap().len(),
        1,
        "messages=false override ⇒ no messages cachePoint"
    );
}

#[test]
fn t4_unsupported_model_zero_cachepoints_even_with_master_on() {
    let ctrl = PromptCachingControl::parse(None);
    let (sys_on, msg_on) = resolved(ctrl, true);

    let caps = FakeCaps::supports(None, None, None);
    let system = decorate_system_blocks(sys(&[&long_text(100)]), "m", &caps, sys_on, None);
    assert_eq!(
        count_cp(&system),
        0,
        "unsupported model ⇒ no system cachePoint"
    );

    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &caps,
        msg_on,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(
        count_cp(&messages),
        0,
        "unsupported model ⇒ no messages cachePoint"
    );
}

#[test]
fn t4_master_off_no_extra_body_zero_cachepoints_supported_model() {
    let ctrl = PromptCachingControl::parse(None);
    let (sys_on, msg_on) = resolved(ctrl, false);

    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let system = decorate_system_blocks(sys(&[&long_text(100)]), "m", &caps, sys_on, None);
    assert_eq!(count_cp(&system), 0, "master off ⇒ no system cachePoint");

    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &caps,
        msg_on,
        0,
        caps.checkpoints(),
        None,
    );
    assert_eq!(
        count_cp(&messages),
        0,
        "master off ⇒ no messages cachePoint"
    );

    let extra = json!({ "prompt_caching": { "system": true } });
    let forced = PromptCachingControl::parse(Some(&extra));
    let system_forced = decorate_system_blocks(
        sys(&[&long_text(100)]),
        "m",
        &caps,
        forced.system_enabled(false),
        None,
    );
    assert_eq!(
        count_cp(&system_forced),
        1,
        "master off but extra_body forces system on"
    );
}

fn count_cp(value: &Value) -> u32 {
    match value {
        Value::Array(arr) => arr
            .iter()
            .map(|item| match item {
                Value::Object(o) if o.contains_key("cachePoint") => 1,
                _ => item.get("content").map(count_cp).unwrap_or(0),
            })
            .sum(),
        _ => 0,
    }
}

// ---- T6: decorate_tools + shared tools→system→messages budget ----

fn tool_specs(names: &[&str]) -> Value {
    Value::Array(
        names
            .iter()
            .map(|n| {
                json!({
                    "toolSpec": {
                        "name": n,
                        "inputSchema": { "json": { "type": "object", "properties": {} } }
                    }
                })
            })
            .collect(),
    )
}

fn tools_tail_is_cache_point(tools: &Value) -> bool {
    tools
        .as_array()
        .and_then(|a| a.last())
        .and_then(Value::as_object)
        .is_some_and(|o| o.contains_key("cachePoint"))
}

/// Drive the full assemble() zone order (tools → system → messages) against
/// the pure decorators with ONE shared running budget, exactly as
/// `provider.rs::assemble` does. Returns each decorated zone plus the grand
/// total of cachePoints placed.
#[allow(clippy::too_many_arguments)]
fn run_three_zones(
    tools: Value,
    system: Value,
    messages: Value,
    model: &str,
    caps: &dyn ModelCapabilities,
    enabled: bool,
    max_cp: Option<u32>,
    ttl: Option<&str>,
) -> (Value, Value, Value, u32) {
    let mut used: u32 = 0;

    let dt = decorate_tools(tools, model, caps, enabled, used, max_cp, ttl);
    used += count_cp(&dt);

    let ds = decorate_system_blocks(system, model, caps, enabled, ttl);
    used += count_cp(&ds);

    let dm = decorate_messages(messages, model, caps, enabled, used, max_cp, ttl);
    let grand = count_cp(&dt) + count_cp(&ds) + count_cp(&dm);
    (dt, ds, dm, grand)
}

#[test]
fn t6_tools_system_history_budget4_tail_cp_and_total_le_4() {
    // tools + system + long history, max_checkpoints=4 → ≤4 total AND tools
    // tail carries a cachePoint.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![
        user_turn("first long turn"),
        assistant_turn("reply"),
        user_turn(&long_text(50)),
    ]);
    let (dt, ds, dm, grand) = run_three_zones(
        tool_specs(&["get_weather", "search"]),
        sys(&[&long_text(10)]),
        messages,
        "m",
        &caps,
        true,
        Some(4),
        None,
    );
    assert!(grand <= 4, "grand total {grand} must be ≤ 4");
    assert!(
        tools_tail_is_cache_point(&dt),
        "tools array tail must carry a cachePoint"
    );
    // tools(1) + system(1) + messages(1) all fit under 4.
    assert_eq!(count_cp(&dt), 1, "one tools cachePoint");
    assert_eq!(count_cp(&ds), 1, "one system cachePoint");
    assert_eq!(count_cp(&dm), 1, "one messages cachePoint");
    assert_eq!(grand, 3);
}

#[test]
fn t6_no_tools_system_messages_behavior_unchanged_regression() {
    // NO tools → no tools cachePoint; system/messages identical to the
    // pre-T6 (system_checkpoints-seeded) behavior.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![user_turn("q")]);

    // T6 path: empty tools array (no toolConfig) + shared budget.
    let (dt, ds_new, dm_new, grand) = run_three_zones(
        Value::Array(vec![]),
        sys(&[&long_text(10)]),
        messages.clone(),
        "m",
        &caps,
        true,
        Some(4),
        None,
    );
    assert_eq!(count_cp(&dt), 0, "no tools ⇒ zero tools cachePoint");

    // Legacy path (pre-T6): system decorated, then messages seeded with the
    // system checkpoint count.
    let ds_old = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, true, None);
    let sys_cp = count_cp(&ds_old);
    let dm_old = decorate_messages(messages, "m", &caps, true, sys_cp, Some(4), None);

    assert_eq!(ds_new, ds_old, "system behavior must be unchanged");
    assert_eq!(dm_new, dm_old, "messages behavior must be unchanged");
    assert_eq!(grand, count_cp(&ds_old) + count_cp(&dm_old));
}

#[test]
fn t6_budget2_places_exactly_two_tools_and_system_messages_skipped() {
    // max_checkpoints=2 → exactly 2 placed (tools + system); messages skipped
    // because the budget is exhausted by the time we reach zone 3.
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(2));
    let messages = Value::Array(vec![user_turn(&long_text(20))]);
    let (dt, ds, dm, grand) = run_three_zones(
        tool_specs(&["t1"]),
        sys(&[&long_text(10)]),
        messages,
        "m",
        &caps,
        true,
        Some(2),
        None,
    );
    assert_eq!(grand, 2, "exactly two cachePoints under a budget of 2");
    assert_eq!(count_cp(&dt), 1, "tools consumes slot 1");
    assert_eq!(count_cp(&ds), 1, "system consumes slot 2");
    assert_eq!(
        count_cp(&dm),
        0,
        "messages skipped: budget exhausted before zone 3"
    );
    assert!(tools_tail_is_cache_point(&dt));
}

#[test]
fn t6_unsupported_or_disabled_zero_cachepoints_anywhere() {
    let messages = Value::Array(vec![user_turn(&long_text(20))]);

    // (a) Unsupported model: zero cache params ⇒ zero cachePoints in all
    // three zones, including tools.
    let unsupported = FakeCaps::supports(None, None, None);
    let (dt, ds, dm, grand) = run_three_zones(
        tool_specs(&["t1", "t2"]),
        sys(&[&long_text(50)]),
        messages.clone(),
        "m",
        &unsupported,
        true,
        Some(4),
        None,
    );
    assert_eq!(grand, 0, "unsupported model ⇒ zero cachePoints");
    assert_eq!(count_cp(&dt), 0);
    assert_eq!(count_cp(&ds), 0);
    assert_eq!(count_cp(&dm), 0);
    assert!(!tools_tail_is_cache_point(&dt));

    // (b) Caching disabled (all zone flags false) on a SUPPORTED model ⇒
    // still zero cachePoints anywhere, including tools.
    let supported = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let (dt2, ds2, dm2, grand2) = run_three_zones(
        tool_specs(&["t1", "t2"]),
        sys(&[&long_text(50)]),
        messages,
        "m",
        &supported,
        false,
        Some(4),
        None,
    );
    assert_eq!(grand2, 0, "disabled ⇒ zero cachePoints");
    assert_eq!(count_cp(&dt2), 0);
    assert_eq!(count_cp(&ds2), 0);
    assert_eq!(count_cp(&dm2), 0);
    assert!(!tools_tail_is_cache_point(&dt2));
}

#[test]
fn t6_decorate_tools_empty_and_double_insert_guard() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    // Empty tools ⇒ unchanged.
    let empty = decorate_tools(Value::Array(vec![]), "m", &caps, true, 0, Some(4), None);
    assert_eq!(empty, Value::Array(vec![]));

    // Already ends with a cachePoint ⇒ no double-insert.
    let already = Value::Array(vec![
        json!({ "toolSpec": { "name": "t", "inputSchema": { "json": {} } } }),
        cache_point(None),
    ]);
    let out = decorate_tools(already.clone(), "m", &caps, true, 0, Some(4), None);
    assert_eq!(out, already, "must not append a second tools cachePoint");

    // Budget exhausted ⇒ unchanged.
    let full = decorate_tools(tool_specs(&["t"]), "m", &caps, true, 4, Some(4), None);
    assert_eq!(count_cp(&full), 0, "no room under the ceiling");
}

// ---- PR-G: cachePoint.ttl (5m/1h) ----

/// A test-only capabilities impl that reports `Capability::CacheTtl1h` for a
/// model, so the 1h gate can be exercised in a `.rs` test WITHOUT naming a
/// model in production code (zero-hardcoding: the flag is the gate).
struct TtlCaps {
    supports_1h: bool,
}

impl ModelCapabilities for TtlCaps {
    fn has(&self, _model: &str, cap: Capability) -> bool {
        matches!(cap, Capability::CacheTtl1h) && self.supports_1h
    }
    fn resolve_foundation(&self, m: &str) -> String {
        m.to_string()
    }
    fn budget_ratios(&self, _model: &str) -> Option<crate::domain::BudgetRatios> {
        None
    }
    fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
        Some(20_000)
    }
    fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
        Some(1)
    }
    fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
        Some(4)
    }
    fn beta_headers(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }
    fn reasoning_path(&self, _model: &str) -> crate::domain::ReasoningPath {
        crate::domain::ReasoningPath::None
    }
    fn responses_backend(&self, _model: &str) -> crate::domain::ResponsesBackend {
        crate::domain::ResponsesBackend::Converse
    }
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
    }
    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        None
    }
}

#[test]
fn cache_point_emits_ttl_when_present() {
    assert_eq!(
        cache_point(Some("1h")),
        json!({ "cachePoint": { "type": "default", "ttl": "1h" } })
    );
    assert_eq!(
        cache_point(Some("5m")),
        json!({ "cachePoint": { "type": "default", "ttl": "5m" } })
    );
    // None ⇒ NO ttl key: byte-identical to the historical output.
    assert_eq!(
        cache_point(None),
        json!({ "cachePoint": { "type": "default" } })
    );
    let none = serde_json::to_string(&cache_point(None)).unwrap();
    assert!(!none.contains("ttl"), "None must not emit a ttl key");
}

#[test]
fn decorate_system_blocks_threads_ttl() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let out = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, true, Some("1h"));
    let arr = out.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[1], cache_point(Some("1h")));
    assert_eq!(arr[1]["cachePoint"]["ttl"], "1h");
}

#[test]
fn decorate_tools_threads_ttl() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let out = decorate_tools(
        tool_specs(&["t1"]),
        "m",
        &caps,
        true,
        0,
        Some(4),
        Some("1h"),
    );
    assert!(tools_tail_is_cache_point(&out));
    let tail = out.as_array().unwrap().last().unwrap();
    assert_eq!(tail, &cache_point(Some("1h")));
    assert_eq!(tail["cachePoint"]["ttl"], "1h");
}

#[test]
fn decorate_messages_threads_ttl() {
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![user_turn("q")]);
    let out = decorate_messages(
        messages,
        "m",
        &caps,
        true,
        0,
        caps.checkpoints(),
        Some("1h"),
    );
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[1], cache_point(Some("1h")));
    assert_eq!(content[1]["cachePoint"]["ttl"], "1h");
}

#[test]
fn prompt_caching_ttl_parsed_from_extra_body() {
    let mut extra = Some(json!({
        "thinking": {"type": "enabled"},
        "prompt_caching": {"system": true, "ttl": "1h"}
    }));
    let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
    assert_eq!(ctrl.ttl.as_deref(), Some("1h"));
    assert_eq!(ctrl.system, Some(true));

    // The control field is stripped; the Bedrock-bound field survives.
    let obj = extra.expect("extra retained");
    assert!(
        obj.get("prompt_caching").is_none(),
        "prompt_caching (incl. ttl) must be stripped from extra_body"
    );
    assert!(obj.get("thinking").is_some());

    // Absent ttl ⇒ None (unchanged behavior).
    let mut no_ttl = Some(json!({ "prompt_caching": {"system": true} }));
    let ctrl2 = PromptCachingControl::extract_and_strip(&mut no_ttl);
    assert_eq!(ctrl2.ttl, None);
}

#[test]
fn one_hour_downgraded_to_5m_on_unsupported_model() {
    // Model LACKS cache_ttl_1h: a requested 1h resolves to 5m + downgrade flag.
    let unsupported = TtlCaps { supports_1h: false };
    let resolved = resolve_cache_ttl(Some("1h"), "5m", "m", &unsupported);
    assert_eq!(resolved.effective, "5m");
    assert_eq!(resolved.requested, "1h");
    assert!(
        resolved.downgraded,
        "1h on unsupported model must downgrade"
    );

    // The emitted cachePoints (all zones) carry 5m — assert via decorated args.
    let ttl = Some(resolved.effective.as_str());
    let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &unsupported, true, ttl);
    assert_eq!(system.as_array().unwrap()[1]["cachePoint"]["ttl"], "5m");
    let messages = decorate_messages(
        Value::Array(vec![user_turn("q")]),
        "m",
        &unsupported,
        true,
        0,
        Some(4),
        ttl,
    );
    assert_eq!(
        messages[0]["content"].as_array().unwrap()[1]["cachePoint"]["ttl"],
        "5m"
    );

    // Model SUPPORTS cache_ttl_1h: 1h is honored, no downgrade.
    let supported = TtlCaps { supports_1h: true };
    let ok = resolve_cache_ttl(Some("1h"), "5m", "m", &supported);
    assert_eq!(ok.effective, "1h");
    assert!(!ok.downgraded);

    // 5m is always allowed on any caching-capable model (no downgrade).
    let five = resolve_cache_ttl(Some("5m"), "5m", "m", &unsupported);
    assert_eq!(five.effective, "5m");
    assert!(!five.downgraded);
}

#[test]
fn resolve_cache_ttl_precedence_per_request_over_default() {
    let supported = TtlCaps { supports_1h: true };
    // Per-request override wins over the settings default.
    let r = resolve_cache_ttl(Some("1h"), "5m", "m", &supported);
    assert_eq!(r.effective, "1h");
    // No per-request ttl ⇒ settings default applies.
    let d = resolve_cache_ttl(None, "5m", "m", &supported);
    assert_eq!(d.effective, "5m");
    let d1h = resolve_cache_ttl(None, "1h", "m", &supported);
    assert_eq!(d1h.effective, "1h");
}

#[test]
fn ttl_is_uniform_across_all_zones_in_one_request() {
    // Every cachePoint placed across tools → system → messages carries the
    // SAME ttl (no longer-before-shorter ordering).
    let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
    let messages = Value::Array(vec![
        user_turn("first"),
        assistant_turn("reply"),
        user_turn(&long_text(50)),
    ]);
    let (dt, ds, dm, grand) = run_three_zones(
        tool_specs(&["get_weather", "search"]),
        sys(&[&long_text(10)]),
        messages,
        "m",
        &caps,
        true,
        Some(4),
        Some("1h"),
    );
    assert_eq!(grand, 3);
    let ttls = collect_ttls(&dt)
        .into_iter()
        .chain(collect_ttls(&ds))
        .chain(collect_ttls(&dm))
        .collect::<Vec<_>>();
    assert_eq!(ttls.len(), 3, "one ttl per zone");
    assert!(
        ttls.iter().all(|t| t == "1h"),
        "all cachePoint ttls must be uniform (1h), got {ttls:?}"
    );
}

/// Recursively gather every `cachePoint.ttl` value present in a decorated
/// zone (arrays of blocks or message turns with nested `content` arrays).
fn collect_ttls(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Value::Array(arr) = value {
        for item in arr {
            if let Some(ttl) = item
                .get("cachePoint")
                .and_then(|cp| cp.get("ttl"))
                .and_then(Value::as_str)
            {
                out.push(ttl.to_string());
            }
            if let Some(content) = item.get("content") {
                out.extend(collect_ttls(content));
            }
        }
    }
    out
}

// ---- Property 3: cache-checkpoint budget invariant (for-all) ----
//
// Feature: test-coverage-codecov, Property 3: 缓存点预算不变量
// (see `.kiro/specs/test-coverage-codecov/design.md`).
//
// These proptests generalize the example-based `t6_*` budget tests above
// into for-all properties. They drive the exact tools → system → messages
// zone order that `provider.rs::assemble` uses (via `run_three_zones`),
// sharing ONE running checkpoint budget. There is NO model-name branching:
// caching support flows solely from the config-gated `cache_min_tokens` /
// `max_cache_tokens` accessors on the fake capabilities impl (Req 1.5).

/// Build a Bedrock `system` array from generated per-block word counts.
/// An empty slice yields an empty (undecorated) system zone.
fn sys_from_words(words: &[usize]) -> Value {
    Value::Array(
        words
            .iter()
            .map(|&w| json!({ "text": long_text(w.max(1)) }))
            .collect(),
    )
}

/// Build a Bedrock `messages` array from generated `(is_user, words)` turns.
fn messages_from(turns: &[(bool, usize)]) -> Value {
    Value::Array(
        turns
            .iter()
            .map(|&(is_user, w)| {
                let text = long_text(w.max(1));
                if is_user {
                    user_turn(&text)
                } else {
                    assistant_turn(&text)
                }
            })
            .collect(),
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Feature: test-coverage-codecov, Property 3: 缓存点预算不变量.
    ///
    /// For ANY model capabilities, tools/system/messages content and cache
    /// toggle, decorating the three zones in `tools → system → messages`
    /// order with one shared budget guarantees:
    /// - each zone carries AT MOST one cachePoint (no double insertion),
    /// - the grand total never exceeds the configured
    ///   `max_cache_checkpoints` ceiling (exercised over the realistic
    ///   `2..=8` domain the provider operates in — it defaults to 4 via
    ///   `unwrap_or` and configs never set 0/1),
    /// - the grand total never exceeds 3 (one per zone).
    #[test]
    fn prop_cache_budget_never_exceeds_ceiling(
        min_opt in prop::option::of(0u32..=1500),
        max_opt in prop::option::of(0u32..=40_000),
        ckpt in 2u32..=8,
        enabled in any::<bool>(),
        tool_names in prop::collection::vec("[a-z]{1,6}", 0..4),
        sys_words in prop::collection::vec(0usize..1000, 0..3),
        msg_turns in prop::collection::vec((any::<bool>(), 1usize..40), 0..5),
    ) {
        let caps = FakeCaps::supports(min_opt, max_opt, Some(ckpt));
        let tool_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();
        let tools = if tool_refs.is_empty() {
            Value::Array(vec![])
        } else {
            tool_specs(&tool_refs)
        };
        let system = sys_from_words(&sys_words);
        let messages = messages_from(&msg_turns);

        let (dt, ds, dm, grand) = run_three_zones(
            tools, system, messages, "m", &caps, enabled, Some(ckpt), None,
        );

        // No zone ever carries more than one cachePoint (no double insertion).
        prop_assert!(count_cp(&dt) <= 1, "tools zone must hold at most one cachePoint");
        prop_assert!(count_cp(&ds) <= 1, "system zone must hold at most one cachePoint");
        prop_assert!(count_cp(&dm) <= 1, "messages zone must hold at most one cachePoint");

        // Grand total never exceeds the configured checkpoint ceiling.
        prop_assert!(grand <= ckpt, "grand total {} must not exceed ceiling {}", grand, ckpt);
        // At most one per zone ⇒ never more than three overall.
        prop_assert!(grand <= 3, "grand total {} must not exceed 3 (one per zone)", grand);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Feature: test-coverage-codecov, Property 3: 缓存点预算不变量
    /// (zero-injection half).
    ///
    /// When the model does NOT support caching (no cache params) OR caching
    /// is disabled, the grand total of cachePoints across all three zones is
    /// exactly 0 — for ANY content and ANY budget (including the degenerate
    /// `0`/`1` ceilings, safe here because the total is 0 regardless).
    #[test]
    fn prop_cache_zero_when_unsupported_or_disabled(
        ckpt_opt in prop::option::of(0u32..=8),
        tool_names in prop::collection::vec("[a-z]{1,6}", 0..4),
        sys_words in prop::collection::vec(1usize..1000, 0..3),
        msg_turns in prop::collection::vec((any::<bool>(), 1usize..40), 0..5),
    ) {
        let tool_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();
        let tools = || if tool_refs.is_empty() {
            Value::Array(vec![])
        } else {
            tool_specs(&tool_refs)
        };
        let system = || sys_from_words(&sys_words);
        let messages = || messages_from(&msg_turns);

        // Case 1: unsupported model (no cache_min_tokens, no max_cache_tokens),
        // caching ENABLED ⇒ still zero cachePoints in every zone.
        let unsupported = FakeCaps::supports(None, None, ckpt_opt);
        let (dt, ds, dm, grand) = run_three_zones(
            tools(), system(), messages(), "m", &unsupported, true, ckpt_opt, None,
        );
        prop_assert_eq!(grand, 0, "unsupported model ⇒ zero cachePoints");
        prop_assert_eq!(count_cp(&dt), 0);
        prop_assert_eq!(count_cp(&ds), 0);
        prop_assert_eq!(count_cp(&dm), 0);

        // Case 2: supported model but caching DISABLED ⇒ also zero.
        let supported = FakeCaps::supports(Some(1), Some(20_000), ckpt_opt);
        let (dt2, ds2, dm2, grand2) = run_three_zones(
            tools(), system(), messages(), "m", &supported, false, ckpt_opt, None,
        );
        prop_assert_eq!(grand2, 0, "caching disabled ⇒ zero cachePoints");
        prop_assert_eq!(count_cp(&dt2), 0);
        prop_assert_eq!(count_cp(&ds2), 0);
        prop_assert_eq!(count_cp(&dm2), 0);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Feature: test-coverage-codecov, Property 3: 缓存点预算不变量
    /// (no-double-insertion guard).
    ///
    /// The tools and messages decorators are idempotent: decorating a zone
    /// that already ends in a cachePoint appends nothing, so re-running the
    /// decorator is a no-op. This proves the `ends_with_cp` guard for ANY
    /// non-empty tools list / user text.
    #[test]
    fn prop_no_double_insert_tools_and_messages(
        tool_names in prop::collection::vec("[a-z]{1,6}", 1..4),
        text in "[a-zA-Z ]{1,64}",
        ckpt in 2u32..=8,
    ) {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(ckpt));
        let tool_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();

        // Tools: first pass inserts exactly one; second pass is a no-op.
        let once = decorate_tools(tool_specs(&tool_refs), "m", &caps, true, 0, Some(ckpt), None);
        prop_assert_eq!(count_cp(&once), 1, "first tools decoration inserts one cachePoint");
        let twice = decorate_tools(once.clone(), "m", &caps, true, 0, Some(ckpt), None);
        prop_assert_eq!(&twice, &once, "tools decoration must not double-insert");

        // Messages: first pass inserts one on the last user turn; second is a no-op.
        let msgs = Value::Array(vec![user_turn(&text)]);
        let m_once = decorate_messages(msgs, "m", &caps, true, 0, Some(ckpt), None);
        prop_assert_eq!(count_cp(&m_once), 1, "first messages decoration inserts one cachePoint");
        let m_twice = decorate_messages(m_once.clone(), "m", &caps, true, 0, Some(ckpt), None);
        prop_assert_eq!(&m_twice, &m_once, "messages decoration must not double-insert");
    }
}

/// Live integration test — skipped unless `BEDROCK_INTEGRATION` is set.
/// Proves (C1/A3) that Nova ACCEPTS an explicit cachePoint and reports a
/// real cache hit: a large system prefix sent twice yields
/// cacheRead > 0 on the second identical call, with no error.
/// Bearer auth via `BEDROCK_API_KEY`, region us-east-2,
/// `us.amazon.nova-lite-v1:0`.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn nova_live_cache_read_on_second_call() {
    use aws_sdk_bedrockruntime::types::{
        CachePointBlock, CachePointType, ContentBlock, ConversationRole, Message,
        SystemContentBlock,
    };

    if std::env::var("BEDROCK_INTEGRATION").is_err() {
        return;
    }

    let settings = crate::config::AppSettings::load().expect("settings load");
    let clients = crate::bedrock::client::BedrockClients::from_settings(&settings).await;

    // A large, byte-stable prefix that comfortably clears the Nova floor.
    let prefix = vec!["The quick brown fox jumps over the lazy dog."; 400].join(" ");
    let system = vec![
        SystemContentBlock::Text(prefix),
        SystemContentBlock::CachePoint(
            CachePointBlock::builder()
                .r#type(CachePointType::Default)
                .build()
                .expect("cache point"),
        ),
    ];
    let user = Message::builder()
        .role(ConversationRole::User)
        .content(ContentBlock::Text(
            "Reply with the single word: ok".to_string(),
        ))
        .build()
        .expect("user message");

    let call = || async {
        clients
            .runtime
            .converse()
            .model_id("us.amazon.nova-lite-v1:0")
            .set_system(Some(system.clone()))
            .messages(user.clone())
            .customize()
            .config_override(crate::bedrock::client::region_config_override("us-east-2"))
            .send()
            .await
    };

    let first = call().await.expect("first Nova converse should succeed");
    let _ = first
        .usage()
        .map(|u| u.cache_write_input_tokens())
        .expect("usage present on first call");

    let second = call().await.expect("second Nova converse should succeed");
    let cache_read = second
        .usage()
        .and_then(|u| u.cache_read_input_tokens())
        .expect("usage present on second call");

    assert!(
        cache_read > 0,
        "second identical Nova call must report cacheRead > 0, got {cache_read}"
    );
}
