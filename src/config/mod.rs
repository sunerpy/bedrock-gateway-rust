//! Configuration module
pub mod capabilities;
pub mod embeddings;
pub mod regions;
pub mod settings;

pub use capabilities::{BudgetRatios, Capability, ModelCapabilityConfig, ReasoningPath};
pub use embeddings::EmbeddingRegistry;
pub use regions::{RegionRoutingConfig, RouteOverride};
pub use settings::AppSettings;
