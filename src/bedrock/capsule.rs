use std::collections::HashMap;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Sha256;

use crate::config::AppSettings;
use crate::error::AppError;

const CAPSULE_PREFIX: &str = "brtc_v1.";
const RESPONSES_CAPSULE_PREFIX: &str = "rsc_v1.";
const MAX_CAPSULE_WIRE_BYTES: usize = 65_536;

pub struct CapsuleKeyring {
    keys: HashMap<String, Vec<u8>>,
    active_kid: Option<String>,
}

impl CapsuleKeyring {
    pub fn new(keys: HashMap<String, Vec<u8>>, active_kid: Option<String>) -> Self {
        Self { keys, active_kid }
    }

    pub fn active_kid(&self) -> Option<&str> {
        self.active_kid.as_deref()
    }

    pub fn key_for(&self, kid: &str) -> Option<&[u8]> {
        self.keys.get(kid).map(Vec::as_slice)
    }
}

pub struct CapsuleRuntime {
    pub keyring: CapsuleKeyring,
    pub encoder_enabled: bool,
}

pub fn resolve_capsule_runtime(settings: &AppSettings) -> Result<CapsuleRuntime, AppError> {
    let mut keys = HashMap::new();
    if let Some(serialized) = settings
        .chat_reasoning_capsule_keys
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        for entry in serialized.split(',') {
            let (kid, encoded_key) = entry.split_once(':').ok_or_else(|| {
                AppError::Internal(
                    "CHAT_REASONING_CAPSULE_KEYS contains a malformed entry".to_string(),
                )
            })?;
            let kid = kid.trim();
            let encoded_key = encoded_key.trim();
            if kid.is_empty() || encoded_key.is_empty() {
                return Err(AppError::Internal(
                    "CHAT_REASONING_CAPSULE_KEYS contains a malformed entry".to_string(),
                ));
            }
            let key = URL_SAFE_NO_PAD.decode(encoded_key).map_err(|_| {
                AppError::Internal(
                    "CHAT_REASONING_CAPSULE_KEYS contains invalid base64url key data".to_string(),
                )
            })?;
            keys.insert(kid.to_string(), key);
        }
    }

    let keyring = CapsuleKeyring::new(keys, settings.chat_reasoning_capsule_active_kid.clone());
    if settings.chat_reasoning_capsule_enabled {
        let active_kid = keyring
            .active_kid()
            .filter(|kid| !kid.is_empty())
            .ok_or_else(|| {
                AppError::Internal(
                    "Chat reasoning capsule encoder is enabled without an active kid".to_string(),
                )
            })?;
        if keyring.key_for(active_kid).is_none() {
            return Err(AppError::Internal(
                "Chat reasoning capsule encoder active kid is missing from the keyring".to_string(),
            ));
        }
    }

    Ok(CapsuleRuntime {
        keyring,
        encoder_enabled: settings.chat_reasoning_capsule_enabled,
    })
}

#[derive(Debug, PartialEq)]
pub struct DecodedCapsule {
    pub tool_use_id: String,
    pub reasoning_blocks: Vec<Value>,
}

#[derive(Debug, PartialEq)]
pub struct DecodedResponsesCapsule {
    pub call_id: String,
    pub reasoning_items: Vec<Value>,
}

#[derive(Serialize)]
struct CapsulePayloadRef<'a> {
    version: u8,
    kid: &'a str,
    tool_use_id: &'a str,
    reasoning: ReasoningPayloadRef<'a>,
}

#[derive(Serialize)]
struct ReasoningPayloadRef<'a> {
    version: u8,
    blocks: &'a [Value],
}

#[derive(Deserialize)]
struct CapsulePayload {
    version: u8,
    kid: String,
    tool_use_id: String,
    reasoning: ReasoningPayload,
}

#[derive(Deserialize)]
struct ReasoningPayload {
    version: u8,
    blocks: Vec<Value>,
}

#[derive(Serialize)]
struct ResponsesCapsulePayloadRef<'a> {
    version: u8,
    kid: &'a str,
    call_id: &'a str,
    reasoning_items: &'a [Value],
}

#[derive(Deserialize)]
struct ResponsesCapsulePayload {
    version: u8,
    kid: String,
    call_id: String,
    reasoning_items: Vec<Value>,
}

pub fn is_capsule(id: &str) -> bool {
    id.starts_with(CAPSULE_PREFIX)
}

pub fn is_responses_capsule(id: &str) -> bool {
    id.starts_with(RESPONSES_CAPSULE_PREFIX)
}

pub fn encode_capsule(
    tool_use_id: &str,
    reasoning_blocks: &[Value],
    keyring: &CapsuleKeyring,
) -> Result<String, AppError> {
    let kid = keyring
        .active_kid()
        .filter(|kid| !kid.is_empty())
        .ok_or_else(|| AppError::Internal("capsule encoder has no active kid".to_string()))?;
    let key = keyring.key_for(kid).ok_or_else(|| {
        AppError::Internal("capsule encoder active kid is not in the keyring".to_string())
    })?;
    let payload = CapsulePayloadRef {
        version: 1,
        kid,
        tool_use_id,
        reasoning: ReasoningPayloadRef {
            version: 1,
            blocks: reasoning_blocks,
        },
    };
    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|_| AppError::Internal("failed to serialize reasoning capsule".to_string()))?;
    let payload_segment = URL_SAFE_NO_PAD.encode(payload_bytes);
    let signed = format!("{CAPSULE_PREFIX}{payload_segment}");
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|_| AppError::Internal("invalid capsule signing key".to_string()))?;
    mac.update(signed.as_bytes());
    let tag = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());
    let wire = format!("{signed}.{tag}");
    if wire.len() > MAX_CAPSULE_WIRE_BYTES {
        return Err(AppError::Internal(
            "encoded reasoning capsule exceeds the wire size limit".to_string(),
        ));
    }
    Ok(wire)
}

pub fn decode_capsule(
    candidate: &str,
    keyring: &CapsuleKeyring,
) -> Result<DecodedCapsule, AppError> {
    if candidate.len() > MAX_CAPSULE_WIRE_BYTES {
        return Err(AppError::BadRequest(
            "reasoning capsule exceeds the wire size limit".to_string(),
        ));
    }
    let encoded = candidate
        .strip_prefix(CAPSULE_PREFIX)
        .ok_or_else(|| AppError::BadRequest("malformed capsule".to_string()))?;
    let mut segments = encoded.split('.');
    let payload_segment = segments
        .next()
        .filter(|segment| !segment.is_empty())
        .ok_or_else(|| AppError::BadRequest("malformed capsule".to_string()))?;
    let tag_segment = segments
        .next()
        .filter(|segment| !segment.is_empty())
        .ok_or_else(|| AppError::BadRequest("malformed capsule".to_string()))?;
    if segments.next().is_some() {
        return Err(AppError::BadRequest("malformed capsule".to_string()));
    }
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_segment)
        .map_err(|_| AppError::BadRequest("malformed capsule".to_string()))?;
    let payload: CapsulePayload = serde_json::from_slice(&payload_bytes)
        .map_err(|_| AppError::BadRequest("malformed capsule".to_string()))?;
    if payload.version != 1 || payload.reasoning.version != 1 {
        return Err(AppError::BadRequest(
            "unsupported reasoning capsule version".to_string(),
        ));
    }
    if payload.kid.is_empty() {
        return Err(AppError::BadRequest(
            "reasoning capsule has no kid".to_string(),
        ));
    }
    let key = keyring
        .key_for(&payload.kid)
        .ok_or_else(|| AppError::BadRequest("unknown reasoning capsule kid".to_string()))?;
    let tag = URL_SAFE_NO_PAD
        .decode(tag_segment)
        .map_err(|_| AppError::BadRequest("malformed capsule".to_string()))?;
    let signed = format!("{CAPSULE_PREFIX}{payload_segment}");
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|_| AppError::BadRequest("invalid reasoning capsule key".to_string()))?;
    mac.update(signed.as_bytes());
    mac.verify_slice(&tag)
        .map_err(|_| AppError::BadRequest("reasoning capsule authentication failed".to_string()))?;
    if !payload
        .reasoning
        .blocks
        .iter()
        .all(reasoning_block_is_valid)
    {
        return Err(AppError::BadRequest(
            "reasoning capsule contains an invalid block".to_string(),
        ));
    }
    Ok(DecodedCapsule {
        tool_use_id: payload.tool_use_id,
        reasoning_blocks: payload.reasoning.blocks,
    })
}

pub fn encode_responses_capsule(
    call_id: &str,
    reasoning_items: &[Value],
    keyring: &CapsuleKeyring,
) -> Result<String, AppError> {
    if call_id.is_empty()
        || !reasoning_items
            .iter()
            .all(responses_reasoning_item_is_valid)
    {
        return Err(AppError::Internal(
            "cannot encode an invalid Responses reasoning capsule".to_string(),
        ));
    }
    let kid = keyring
        .active_kid()
        .filter(|kid| !kid.is_empty())
        .ok_or_else(|| AppError::Internal("capsule encoder has no active kid".to_string()))?;
    let key = keyring.key_for(kid).ok_or_else(|| {
        AppError::Internal("capsule encoder active kid is not in the keyring".to_string())
    })?;
    let payload = ResponsesCapsulePayloadRef {
        version: 1,
        kid,
        call_id,
        reasoning_items,
    };
    let payload_bytes = serde_json::to_vec(&payload).map_err(|_| {
        AppError::Internal("failed to serialize Responses reasoning capsule".to_string())
    })?;
    let payload_segment = URL_SAFE_NO_PAD.encode(payload_bytes);
    let signed = format!("{RESPONSES_CAPSULE_PREFIX}{payload_segment}");
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|_| AppError::Internal("invalid capsule signing key".to_string()))?;
    mac.update(signed.as_bytes());
    let tag = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());
    let wire = format!("{signed}.{tag}");
    if wire.len() > MAX_CAPSULE_WIRE_BYTES {
        return Err(AppError::Internal(
            "encoded Responses reasoning capsule exceeds the wire size limit".to_string(),
        ));
    }
    Ok(wire)
}

pub fn decode_responses_capsule(
    candidate: &str,
    keyring: &CapsuleKeyring,
) -> Result<DecodedResponsesCapsule, AppError> {
    if candidate.len() > MAX_CAPSULE_WIRE_BYTES {
        return Err(AppError::BadRequest(
            "Responses reasoning capsule exceeds the wire size limit".to_string(),
        ));
    }
    let encoded = candidate
        .strip_prefix(RESPONSES_CAPSULE_PREFIX)
        .ok_or_else(|| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    let mut segments = encoded.split('.');
    let payload_segment = segments
        .next()
        .filter(|segment| !segment.is_empty())
        .ok_or_else(|| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    let tag_segment = segments
        .next()
        .filter(|segment| !segment.is_empty())
        .ok_or_else(|| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    if segments.next().is_some() {
        return Err(AppError::BadRequest(
            "malformed Responses reasoning capsule".to_string(),
        ));
    }
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_segment)
        .map_err(|_| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    let payload: ResponsesCapsulePayload = serde_json::from_slice(&payload_bytes)
        .map_err(|_| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    if payload.version != 1 {
        return Err(AppError::BadRequest(
            "unsupported Responses reasoning capsule version".to_string(),
        ));
    }
    if payload.kid.is_empty() {
        return Err(AppError::BadRequest(
            "Responses reasoning capsule has no kid".to_string(),
        ));
    }
    let key = keyring.key_for(&payload.kid).ok_or_else(|| {
        AppError::BadRequest("unknown Responses reasoning capsule kid".to_string())
    })?;
    let tag = URL_SAFE_NO_PAD
        .decode(tag_segment)
        .map_err(|_| AppError::BadRequest("malformed Responses reasoning capsule".to_string()))?;
    let signed = format!("{RESPONSES_CAPSULE_PREFIX}{payload_segment}");
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .map_err(|_| AppError::BadRequest("invalid Responses reasoning capsule key".to_string()))?;
    mac.update(signed.as_bytes());
    mac.verify_slice(&tag).map_err(|_| {
        AppError::BadRequest("Responses reasoning capsule authentication failed".to_string())
    })?;
    if payload.call_id.is_empty()
        || !payload
            .reasoning_items
            .iter()
            .all(responses_reasoning_item_is_valid)
    {
        return Err(AppError::BadRequest(
            "Responses reasoning capsule contains invalid data".to_string(),
        ));
    }
    Ok(DecodedResponsesCapsule {
        call_id: payload.call_id,
        reasoning_items: payload.reasoning_items,
    })
}

pub(crate) fn reasoning_block_is_valid(block: &Value) -> bool {
    let valid_text = block
        .get("reasoningText")
        .and_then(Value::as_object)
        .is_some_and(|text| {
            text.get("text").and_then(Value::as_str).is_some()
                && text.get("signature").and_then(Value::as_str).is_some()
        });
    let valid_redacted = block
        .get("redactedContent")
        .and_then(Value::as_str)
        .is_some();
    valid_text || valid_redacted
}

pub(crate) fn responses_reasoning_item_is_valid(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && item
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty())
        && item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.is_empty())
}

#[cfg(test)]
#[path = "capsule_tests.rs"]
mod tests;
