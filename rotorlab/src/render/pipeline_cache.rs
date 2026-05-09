//! Cache of compiled pipelines keyed by drawable kind.
//!
//! Plan 3 Task 3 fills in the lazy-construction store. A
//! [`PipelineCache`] holds at most one [`Arc<dyn Pipeline>`] per
//! [`PipelineId`]; calling [`PipelineCache::get_or_create`] for a fresh
//! id invokes the supplied factory and caches the result, while a
//! repeat lookup returns the previously stored handle without invoking
//! the factory again.
//!
//! # Concurrency
//!
//! The cache is internally synchronised with a [`Mutex`] so it is
//! `Send + Sync`. Frame recording in v0.0.x is single-threaded, but the
//! [`Drawable`](crate::object::Drawable) trait already requires `Send`
//! and a multi-threaded recording path would only need this type to be
//! shared. A per-id [`OnceCell`](std::cell::OnceCell)-style storage is
//! a possible later optimisation; for now `Mutex<HashMap>` keeps the
//! API simple and the contention low (one lock per drawable, not per
//! draw).
//!
//! # Lifetime
//!
//! Each cached pipeline is reference-counted via `Arc`. Dropping the
//! [`PipelineCache`] drops every stored `Arc`; the underlying Vulkan
//! resources are released when the last `Arc` to a given pipeline goes
//! away, which is normally also when the cache is dropped.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::RenderError;
use crate::render::pipelines::Pipeline;

/// Identifier for a cached pipeline.
///
/// Plan 3 ships only [`PipelineId::Point`] and [`PipelineId::Line`].
/// Future drawables (filled triangle, shaded mesh, textured glyph)
/// extend this enum as they land.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum PipelineId {
    /// Pipeline used by [`crate::object::Drawable`] implementations
    /// that render points (instanced billboard quads in Plan 3).
    Point,
    /// Pipeline used by drawables that render thick lines
    /// (screen-space-thickness rectangles in Plan 3).
    Line,
}

/// Cache of GPU pipelines, looked up by drawable kind.
///
/// See the [module docs](self) for the cache contract. Construct with
/// [`PipelineCache::new`]; populate lazily via
/// [`PipelineCache::get_or_create`].
#[derive(Default)]
pub struct PipelineCache {
    /// One slot per [`PipelineId`]. The lock is held only across the
    /// short critical sections that read or insert; pipeline
    /// construction itself runs outside the lock so a slow factory does
    /// not block other slots.
    slots: Mutex<HashMap<PipelineId, Arc<dyn Pipeline>>>,
}

impl PipelineCache {
    /// Construct an empty pipeline cache.
    ///
    /// Equivalent to `PipelineCache::default()`. Provided so call sites
    /// can stay readable without importing the `Default` trait.
    pub fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }

    /// Get or lazily build the pipeline keyed by `id`.
    ///
    /// `factory` is invoked at most once per `id` per
    /// [`PipelineCache`] lifetime; subsequent lookups for the same id
    /// return the cached `Arc` without invoking it. The factory takes
    /// no arguments; callers close over whatever device, render pass,
    /// or viewport state they need at the call site, which keeps the
    /// cache itself ignorant of pipeline-construction signatures.
    ///
    /// # Errors
    ///
    /// Returns whatever [`RenderError`] the factory returns on a fresh
    /// id; an error is *not* cached, so a subsequent call with the
    /// same id and a different factory will retry.
    ///
    /// # Concurrency
    ///
    /// Two threads racing on the same fresh id may both observe the
    /// slot empty and both invoke their factories; the first insert
    /// wins and the loser's pipeline is dropped immediately. This is
    /// acceptable because pipeline construction is idempotent and
    /// expensive only on cold start. If this ever needs tightening,
    /// switch the inner type to `OnceCell<Arc<dyn Pipeline>>` per id.
    pub fn get_or_create<F>(
        &self,
        id: PipelineId,
        factory: F,
    ) -> Result<Arc<dyn Pipeline>, RenderError>
    where
        F: FnOnce() -> Result<Arc<dyn Pipeline>, RenderError>,
    {
        // Fast path: already populated.
        if let Some(existing) = self
            .slots
            .lock()
            .expect("pipeline cache mutex poisoned")
            .get(&id)
            .cloned()
        {
            return Ok(existing);
        }

        // Slow path: build outside the lock.
        let built = factory()?;

        let mut slots = self.slots.lock().expect("pipeline cache mutex poisoned");
        // If another thread populated the slot while we were building,
        // their entry wins; drop our freshly-built pipeline by leaving
        // it as a local that goes out of scope at function return.
        let entry = slots.entry(id).or_insert_with(|| built.clone());
        Ok(entry.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Trivial Vulkan-free [`Pipeline`] used by the cache tests.
    struct FakePipeline {
        #[allow(dead_code)]
        marker: usize,
    }
    impl Pipeline for FakePipeline {
        fn raw(&self) -> vk::Pipeline {
            vk::Pipeline::null()
        }
        fn layout(&self) -> vk::PipelineLayout {
            vk::PipelineLayout::null()
        }
    }

    #[test]
    fn cache_populates_lazily() {
        let cache = PipelineCache::new();
        let calls = AtomicUsize::new(0);
        let factory = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakePipeline { marker: 1 }) as Arc<dyn Pipeline>)
        };
        let _p1 = cache.get_or_create(PipelineId::Point, factory).unwrap();
        // Same id, second lookup must NOT invoke the factory again.
        let factory2 = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakePipeline { marker: 2 }) as Arc<dyn Pipeline>)
        };
        let _p2 = cache.get_or_create(PipelineId::Point, factory2).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn second_lookup_returns_same_handle() {
        let cache = PipelineCache::new();
        let p1 = cache
            .get_or_create(PipelineId::Line, || {
                Ok(Arc::new(FakePipeline { marker: 42 }) as Arc<dyn Pipeline>)
            })
            .unwrap();
        let p2 = cache
            .get_or_create(PipelineId::Line, || {
                unreachable!("factory must not be called for cached id");
            })
            .unwrap();
        // Pointer identity.
        assert!(Arc::ptr_eq(&p1, &p2));
    }

    #[test]
    fn drop_order_is_correct() {
        // Use a Pipeline whose Drop bumps a counter; populate the cache
        // with two ids; drop the cache; assert both Pipelines were dropped.
        struct CountingPipeline {
            dropper: Arc<AtomicUsize>,
        }
        impl Pipeline for CountingPipeline {
            fn raw(&self) -> vk::Pipeline {
                vk::Pipeline::null()
            }
            fn layout(&self) -> vk::PipelineLayout {
                vk::PipelineLayout::null()
            }
        }
        impl Drop for CountingPipeline {
            fn drop(&mut self) {
                self.dropper.fetch_add(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let cache = PipelineCache::new();
        let p1 = cache
            .get_or_create(PipelineId::Point, {
                let c = counter.clone();
                move || Ok(Arc::new(CountingPipeline { dropper: c }) as Arc<dyn Pipeline>)
            })
            .unwrap();
        let p2 = cache
            .get_or_create(PipelineId::Line, {
                let c = counter.clone();
                move || Ok(Arc::new(CountingPipeline { dropper: c }) as Arc<dyn Pipeline>)
            })
            .unwrap();
        // Drop our outstanding Arc handles, then the cache.
        drop(p1);
        drop(p2);
        drop(cache);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
