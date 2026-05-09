//! Cache of compiled pipelines keyed by drawable kind.
//!
//! Stub for Task 3: see plan. Plan 3 Task 3 fills this in with real
//! storage for the per-drawable pipelines (point, line, plane, ...). For
//! Task 2 the [`Scene`](crate::scene::Scene) only needs an owned
//! `PipelineCache` field; the cache is otherwise unused.

/// Cache of GPU pipelines, looked up by drawable kind.
///
/// Stub for Task 3: see plan. In Plan 3 Task 3 this becomes a real
/// cache that lazily compiles and stores per-kind pipelines (point,
/// line, plane, ...). For Task 2 it stores nothing and exists only so
/// [`Scene`](crate::scene::Scene) can name an owned field.
#[derive(Default)]
pub struct PipelineCache {
    /// Reserved private state for Task 3. The empty struct keeps the
    /// public surface small and lets us evolve the representation freely
    /// when real pipeline storage lands.
    _private: (),
}

impl PipelineCache {
    /// Construct an empty pipeline cache.
    ///
    /// Equivalent to `PipelineCache::default()`. Provided so call sites
    /// can stay readable without importing the `Default` trait.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default_construct() {
        let _ = PipelineCache::new();
        let _ = PipelineCache::default();
    }
}
