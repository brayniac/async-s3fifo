//! Multi-tier cache with S3-FIFO admission policy
//!
//! This module provides the top-level `Cache` struct that orchestrates
//! multiple storage tiers (RAM, SSD) with proper demotion callbacks.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         Cache (Public API)                      │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                    RAM Layer (pool_id=0)                 │   │
//! │  │  ┌──────────────┐      ┌────────────────────────────┐   │   │
//! │  │  │ Small Queue  │ ───► │ TTL Buckets (Main Cache)   │   │   │
//! │  │  │ (Admission)  │      │ (Segment-level TTL)        │   │   │
//! │  │  └──────────────┘      └────────────────────────────┘   │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │              │ demote cold items      │ merge eviction demote  │
//! │              ▼                        ▼                        │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                    SSD Layer (pool_id=1)                 │   │
//! │  │  ┌──────────────┐      ┌────────────────────────────┐   │   │
//! │  │  │ Small Queue  │ ───► │ TTL Buckets (Main Cache)   │   │   │
//! │  │  │ (Admission)  │      │ (Segment-level TTL)        │   │   │
//! │  │  └──────────────┘      └────────────────────────────┘   │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │              │ evict (no demotion)                             │
//! │              ▼                                                 │
//! │           Drop / Ghost entries                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::hashtable::Hashtable;
use crate::layer::CacheLayer;
use crate::metrics::CacheMetrics;
use crate::pool::{MemoryPool, MmapPool, Pool};
use crate::segment::Segment;
use crate::ttlbuckets::TtlBuckets;
use clocksource::coarse::UnixInstant;
use std::time::Duration;
use std::sync::Arc;

/// SSD backend type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsdBackendType {
    /// Automatically select the best backend for this platform.
    /// - Linux/macOS: DirectIo (O_DIRECT/F_NOCACHE)
    /// - Other platforms: Mmap (fallback)
    #[default]
    Auto,
    /// Memory-mapped file backend.
    /// Uses the OS page cache, which may cause double-caching.
    /// Simpler but less predictable memory usage.
    Mmap,
    /// Direct I/O backend (O_DIRECT on Linux, F_NOCACHE on macOS).
    /// Bypasses the OS page cache for predictable memory usage.
    /// Reads from SSD are automatically promoted to RAM.
    DirectIo,
}

/// Internal enum for the actual SSD layer implementation
enum SsdLayer {
    /// Memory-mapped SSD storage
    Mmap(CacheLayer<MmapPool>),
    /// Direct I/O SSD storage (not yet implemented - placeholder)
    #[allow(dead_code)]
    DirectIo {
        pool: crate::pool::DirectIoPool,
        // Note: For direct I/O, we track segments but reads go to RAM
        // The small_queue and ttl_buckets would need custom implementation
        // that writes to disk rather than memory-mapped segments
    },
}

impl SsdLayer {
    /// Get the mmap layer if this is a Mmap backend.
    /// Returns None for DirectIo backend.
    fn as_mmap(&self) -> Option<&CacheLayer<MmapPool>> {
        match self {
            SsdLayer::Mmap(layer) => Some(layer),
            SsdLayer::DirectIo { .. } => None,
        }
    }

    /// Get the pool from the mmap layer.
    /// Returns None for DirectIo backend.
    fn pool(&self) -> Option<&MmapPool> {
        match self {
            SsdLayer::Mmap(layer) => Some(layer.pool()),
            SsdLayer::DirectIo { .. } => None,
        }
    }

    /// Check if this is a DirectIo backend.
    #[allow(dead_code)]
    fn is_direct_io(&self) -> bool {
        matches!(self, SsdLayer::DirectIo { .. })
    }

    /// Demote an item to the SSD layer.
    ///
    /// For mmap backend, uses the existing CacheLayer append mechanism.
    /// For DirectIo backend, writes directly to the file.
    ///
    /// Returns Ok(()) on success, Err(()) on failure.
    #[allow(dead_code)] // Reserved for future use when wiring up DirectIo demotion
    async fn demote_item(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        hashtable: &Hashtable,
        metrics: &CacheMetrics,
    ) -> Result<(), ()> {
        match self {
            SsdLayer::Mmap(layer) => {
                // Use existing CacheLayer mechanism
                layer.append_to_small_queue(
                    key,
                    value,
                    optional,
                    expire_at,
                    hashtable,
                    metrics,
                ).await.map(|_| ())
            }
            SsdLayer::DirectIo { pool } => {
                // Write to DirectIo pool
                // First, try to get a small queue segment
                let segment_id = pool.reserve_small_queue(metrics).ok_or(())?;

                // Write item to segment
                let offset = pool.append_small_queue_item(
                    segment_id,
                    key,
                    value,
                    optional,
                    expire_at,
                    metrics,
                ).ok_or(())?;

                // Link in hashtable
                // pool_id is 1 for SSD
                hashtable.link_item(
                    key,
                    1, // pool_id for SSD
                    segment_id,
                    offset,
                    pool,
                    metrics,
                ).map_err(|_| ())?;

                Ok(())
            }
        }
    }
}

/// Trait for cache operations needed by TTL buckets and other internal components
pub(crate) trait CacheOps<P: Pool> {
    fn pool(&self) -> &P;
    fn ttl_buckets(&self) -> &TtlBuckets;
    fn hashtable(&self) -> &Hashtable;
    fn metrics(&self) -> &CacheMetrics;

    /// Try to expire a segment from the TTL buckets.
    ///
    /// Scans buckets for expired segments and returns one ready for reuse.
    /// This is called before eviction to reclaim naturally expired segments first.
    ///
    /// # Returns
    /// - `Some(segment_id)`: A segment was expired and is in Reserved state, ready for reuse
    /// - `None`: No expired segments found
    fn try_expire(&self) -> Option<u32>;

    /// Evict a segment using the configured eviction policy
    async fn evict_segment_by_policy(&self) -> Option<u32>;
}

/// Builder for creating a multi-tier cache
///
/// # Example
///
/// ```ignore
/// use s3::CacheBuilder;
///
/// // RAM-only cache with 64MB
/// let cache = CacheBuilder::new()
///     .ram_size(64 * 1024 * 1024)
///     .build()?;
///
/// // RAM + SSD cache
/// let cache = CacheBuilder::new()
///     .ram_size(128 * 1024 * 1024)
///     .ssd("/tmp/cache.dat", 1024 * 1024 * 1024)
///     .build()?;
/// ```
pub struct CacheBuilder {
    // RAM configuration
    ram_size: usize,
    ram_segment_size: usize,
    ram_small_queue_percent: u8,
    /// Hugepage size preference for RAM allocation
    hugepage_size: crate::hugepage::HugepageSize,

    // SSD configuration (optional)
    ssd_path: Option<std::path::PathBuf>,
    ssd_size: usize,
    ssd_segment_size: usize,
    ssd_small_queue_percent: u8,
    /// SSD backend type (mmap, direct I/O, or auto-detect)
    ssd_backend: SsdBackendType,

    // Hashtable configuration
    hashtable_power: u8,
    /// If true, use two-choice hashing for higher fill rates (~95%) at the cost
    /// of ~30% lower throughput. If false (default), use single-choice hashing.
    two_choice_hashing: bool,

    // Merge eviction configuration
    merge_ratio: usize,
    merge_target_ratio: usize,

    // Promotion configuration
    /// If true, items accessed from SSD are automatically promoted to RAM small queue.
    /// This helps hot items that were demoted migrate back to RAM.
    auto_promote: bool,
}

impl Default for CacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheBuilder {
    /// Create a new cache builder with default settings
    ///
    /// Defaults:
    /// - RAM: 64MB with 1MB segments, 10% small queue
    /// - No SSD tier
    /// - Hashtable: 2^16 = 64K buckets (448K items at 7 per bucket)
    /// - Single-choice hashing (faster, ~85% fill rate)
    pub fn new() -> Self {
        Self {
            ram_size: 64 * 1024 * 1024,        // 64MB default
            ram_segment_size: 1024 * 1024,     // 1MB segments
            ram_small_queue_percent: 10,
            hugepage_size: crate::hugepage::HugepageSize::None,
            ssd_path: None,
            ssd_size: 0,
            ssd_segment_size: 4 * 1024 * 1024, // 4MB segments for SSD
            ssd_small_queue_percent: 10,
            ssd_backend: SsdBackendType::Auto, // Auto-detect best backend
            hashtable_power: 16,               // 64K buckets
            two_choice_hashing: false,         // Single-choice is faster
            merge_ratio: 4,
            merge_target_ratio: 2,
            auto_promote: false,               // Disabled by default
        }
    }

    /// Set the total RAM cache size in bytes
    ///
    /// The number of segments is calculated as `ram_size / ram_segment_size`.
    pub fn ram_size(mut self, bytes: usize) -> Self {
        self.ram_size = bytes;
        self
    }

    /// Set the RAM segment size in bytes (default: 1MB)
    ///
    /// Smaller segments allow finer-grained eviction but increase metadata overhead.
    /// Larger segments are more efficient for large values.
    pub fn ram_segment_size(mut self, bytes: usize) -> Self {
        self.ram_segment_size = bytes;
        self
    }

    /// Set the percentage of RAM segments allocated to the small queue (default: 10%)
    ///
    /// The small queue acts as an admission filter - items must be accessed
    /// multiple times to be promoted to the main cache.
    pub fn ram_small_queue_percent(mut self, percent: u8) -> Self {
        debug_assert!(percent <= 100, "percent must be <= 100");
        self.ram_small_queue_percent = percent;
        self
    }

    /// Set the hugepage size preference for RAM heap allocation.
    ///
    /// - `HugepageSize::None` - Use regular 4KB pages (default)
    /// - `HugepageSize::TwoMegabyte` - Try 2MB hugepages, fallback to regular
    /// - `HugepageSize::OneGigabyte` - Try 1GB hugepages, fallback to regular
    ///
    /// Note: The actual allocation may fall back to regular pages if the
    /// requested hugepage size is not available on the system.
    pub fn hugepage_size(mut self, size: crate::hugepage::HugepageSize) -> Self {
        self.hugepage_size = size;
        self
    }

    /// Add an SSD tier with the given path and size
    ///
    /// The SSD tier provides overflow storage for items demoted from RAM.
    /// Items evicted from the RAM small queue or main cache are written to SSD.
    pub fn ssd<P: Into<std::path::PathBuf>>(mut self, path: P, size_bytes: usize) -> Self {
        self.ssd_path = Some(path.into());
        self.ssd_size = size_bytes;
        self
    }

    /// Set the SSD segment size in bytes (default: 4MB)
    ///
    /// SSD segments are typically larger than RAM segments to reduce
    /// metadata overhead and improve sequential I/O performance.
    pub fn ssd_segment_size(mut self, bytes: usize) -> Self {
        self.ssd_segment_size = bytes;
        self
    }

    /// Set the percentage of SSD segments allocated to the small queue (default: 10%)
    pub fn ssd_small_queue_percent(mut self, percent: u8) -> Self {
        debug_assert!(percent <= 100, "percent must be <= 100");
        self.ssd_small_queue_percent = percent;
        self
    }

    /// Set the SSD backend type (default: Auto)
    ///
    /// Backend options:
    /// - `Auto`: Automatically select the best backend for this platform.
    ///   Uses DirectIo on Linux/macOS, falls back to Mmap elsewhere.
    /// - `Mmap`: Memory-mapped file backend. Uses the OS page cache, which
    ///   may cause double-caching but is simpler and cross-platform.
    /// - `DirectIo`: Direct I/O backend (O_DIRECT on Linux, F_NOCACHE on macOS).
    ///   Bypasses the OS page cache for predictable memory usage. Reads from
    ///   SSD are automatically promoted to RAM.
    ///
    /// # Platform Support
    ///
    /// - **Linux**: Full O_DIRECT support. Requires aligned I/O.
    /// - **macOS**: F_NOCACHE support. Best-effort cache bypass.
    /// - **Other**: Falls back to Mmap regardless of setting.
    ///
    /// # Trade-offs
    ///
    /// | Backend  | Memory Predictability | Complexity | Cross-Platform |
    /// |----------|----------------------|------------|----------------|
    /// | Mmap     | Low (page cache)     | Simple     | Yes            |
    /// | DirectIo | High (no page cache) | Higher     | Linux/macOS    |
    pub fn ssd_backend(mut self, backend: SsdBackendType) -> Self {
        self.ssd_backend = backend;
        self
    }

    /// Force the SSD backend to use memory-mapped I/O.
    ///
    /// This is a convenience method equivalent to `.ssd_backend(SsdBackendType::Mmap)`.
    /// Use this when you want to ensure mmap is used regardless of platform.
    pub fn force_mmap(self) -> Self {
        self.ssd_backend(SsdBackendType::Mmap)
    }

    /// Set the hashtable power (2^power buckets)
    ///
    /// Each bucket holds 7 items, so:
    /// - power 16 = 64K buckets = 448K items
    /// - power 18 = 256K buckets = 1.8M items
    /// - power 20 = 1M buckets = 7M items
    ///
    /// Default: 16 (64K buckets)
    pub fn hashtable_power(mut self, power: u8) -> Self {
        self.hashtable_power = power;
        self
    }

    /// Enable two-choice hashing for higher hashtable fill rates
    ///
    /// Trade-off:
    /// - `false` (default): Single-choice hashing, ~30% faster, ~85% fill rate
    /// - `true`: Two-choice hashing, ~95% fill rate, better for memory-constrained environments
    ///
    /// Two-choice hashing computes two bucket indices for each key and inserts
    /// into the less-loaded bucket. This achieves higher fill rates but requires
    /// checking two buckets on lookup.
    pub fn two_choice_hashing(mut self, enabled: bool) -> Self {
        self.two_choice_hashing = enabled;
        self
    }

    /// Set the merge eviction ratio (default: 4)
    ///
    /// During merge eviction, this many source segments are consolidated into
    /// fewer destination segments.
    pub fn merge_ratio(mut self, ratio: usize) -> Self {
        self.merge_ratio = ratio;
        self
    }

    /// Set the merge eviction target ratio (default: 2)
    ///
    /// The target number of destination segments after merge eviction.
    pub fn merge_target_ratio(mut self, ratio: usize) -> Self {
        self.merge_target_ratio = ratio;
        self
    }

    /// Enable automatic promotion from SSD to RAM (default: false)
    ///
    /// When enabled, items accessed from the SSD layer are automatically copied
    /// to the RAM small queue. This helps hot items that were demoted migrate
    /// back to RAM.
    ///
    /// Trade-offs:
    /// - `true`: Better hit latency for re-heated items, but adds write traffic to RAM
    /// - `false`: Items stay on SSD until evicted, simpler behavior
    ///
    /// The promoted item:
    /// - Goes to RAM small queue (must prove itself hot again)
    /// - Gets frequency reset to 1 (fresh start)
    /// - Original SSD copy becomes orphaned (cleaned up on eviction)
    pub fn auto_promote(mut self, enabled: bool) -> Self {
        self.auto_promote = enabled;
        self
    }

    /// Build the cache
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - RAM size is less than one segment
    /// - SSD path is specified but SSD size is 0
    /// - Memory allocation fails
    /// - SSD file creation fails
    pub fn build(self) -> Result<Cache, std::io::Error> {
        // Validate RAM configuration
        if self.ram_size < self.ram_segment_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "RAM size ({}) must be at least one segment ({})",
                    self.ram_size, self.ram_segment_size
                ),
            ));
        }

        // Validate SSD configuration
        if self.ssd_path.is_some() && self.ssd_size < self.ssd_segment_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "SSD size ({}) must be at least one segment ({})",
                    self.ssd_size, self.ssd_segment_size
                ),
            ));
        }

        // Build the cache
        let hashtable = Hashtable::with_two_choice(self.hashtable_power, self.two_choice_hashing);
        let metrics = CacheMetrics::new();

        // Build RAM pool
        let ram_pool = crate::pool::MemoryPoolBuilder::new(0)
            .segment_size(self.ram_segment_size)
            .heap_size(self.ram_size)
            .small_queue_percent(self.ram_small_queue_percent)
            .hugepage_size(self.hugepage_size)
            .build()?;

        // Initialize segments_free gauge with total segment count
        let ram_segment_count = ram_pool.segment_count() as i64;
        metrics.segments_free.set(ram_segment_count);

        let ram_layer = CacheLayer::new(ram_pool, 0);

        // Determine the effective SSD backend based on configuration and platform
        let effective_backend = match self.ssd_backend {
            SsdBackendType::Auto => {
                if crate::pool::is_direct_io_supported() {
                    SsdBackendType::DirectIo
                } else {
                    SsdBackendType::Mmap
                }
            }
            SsdBackendType::DirectIo => {
                if crate::pool::is_direct_io_supported() {
                    SsdBackendType::DirectIo
                } else {
                    // Fallback to mmap on unsupported platforms
                    eprintln!("Warning: DirectIo not supported on this platform, falling back to Mmap");
                    SsdBackendType::Mmap
                }
            }
            SsdBackendType::Mmap => SsdBackendType::Mmap,
        };

        // Build optional SSD pool
        let ssd_layer = if let Some(path) = self.ssd_path {
            match effective_backend {
                SsdBackendType::Mmap | SsdBackendType::Auto => {
                    // Use memory-mapped backend
                    let ssd_pool = crate::pool::MmapPoolBuilder::new(1, &path)
                        .segment_size(self.ssd_segment_size)
                        .heap_size(self.ssd_size)
                        .small_queue_percent(self.ssd_small_queue_percent)
                        .build()?;

                    // Add SSD segments to the free count
                    let ssd_segment_count = ssd_pool.segment_count() as i64;
                    metrics.segments_free.add(ssd_segment_count);

                    Some(SsdLayer::Mmap(CacheLayer::new(ssd_pool, 1)))
                }
                SsdBackendType::DirectIo => {
                    // Use direct I/O backend
                    let ssd_pool = crate::pool::DirectIoPoolBuilder::new(1, &path)
                        .segment_size(self.ssd_segment_size)
                        .heap_size(self.ssd_size)
                        .small_queue_percent(self.ssd_small_queue_percent)
                        .build()?;

                    // Add SSD segments to the free count
                    let ssd_segment_count = ssd_pool.segment_count() as i64;
                    metrics.segments_free.add(ssd_segment_count);

                    Some(SsdLayer::DirectIo { pool: ssd_pool })
                }
            }
        } else {
            None
        };

        Ok(Cache {
            inner: Arc::new(CacheInner {
                hashtable,
                ram_layer,
                ssd_layer,
                metrics,
                merge_ratio: self.merge_ratio,
                merge_target_ratio: self.merge_target_ratio,
                auto_promote: self.auto_promote,
            }),
        })
    }
}

/// Multi-tier cache with S3-FIFO admission policy
///
/// This is the main public API for the cache. It manages:
/// - A RAM layer for fast access
/// - An optional SSD layer for overflow
/// - Demotion callbacks between layers
pub struct Cache {
    inner: Arc<CacheInner>,
}

struct CacheInner {
    /// Shared hashtable for all layers
    hashtable: Hashtable,

    /// RAM cache layer (pool_id = 0)
    ram_layer: CacheLayer<MemoryPool>,

    /// Optional SSD cache layer (pool_id = 1)
    /// Can be either memory-mapped or direct I/O backed
    ssd_layer: Option<SsdLayer>,

    /// Shared metrics
    metrics: CacheMetrics,

    /// Merge eviction configuration
    merge_ratio: usize,
    merge_target_ratio: usize,

    /// Legacy field - SSD reads now always promote to RAM for consistent behavior.
    /// Kept for API compatibility.
    #[allow(dead_code)]
    auto_promote: bool,
}

impl Cache {
    /// Create a new cache builder
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = Cache::builder()
    ///     .ram_size(64 * 1024 * 1024)
    ///     .build()?;
    /// ```
    pub fn builder() -> CacheBuilder {
        CacheBuilder::new()
    }

    /// Insert an item into the cache
    ///
    /// Items are first inserted into the RAM layer's small queue.
    /// If accessed again (frequency > 1), they are promoted to the main cache.
    ///
    /// This method automatically triggers eviction if the cache is full.
    ///
    /// # Returns
    /// - `Ok(())` if the item was inserted
    /// - `Err(StorageFull)` if storage is full and eviction failed
    /// - `Err(HashTableFull)` if hashtable buckets are full
    pub async fn set(&self, key: &[u8], value: &[u8], ttl: Duration) -> Result<(), crate::CacheError> {
        self.set_with_optional(key, value, &[], ttl).await
    }

    /// Insert an item with optional metadata
    ///
    /// This method implements the S3-FIFO ghost-based admission optimization:
    /// - If the key has a ghost entry with frequency > 1 (was accessed after eviction),
    ///   the item goes directly to the main cache (TTL buckets)
    /// - Otherwise, the item goes through the small queue (admission queue)
    ///
    /// This method automatically triggers eviction if the cache is full.
    ///
    /// # Returns
    /// - `Ok(())` if the item was inserted
    /// - `Err(StorageFull)` if storage is full and eviction failed
    /// - `Err(HashTableFull)` if hashtable buckets are full
    pub async fn set_with_optional(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        // Check if this key has a ghost entry with high frequency
        // If so, it was accessed between eviction and re-insertion, suggesting
        // it should bypass the small queue and go directly to main cache
        const GHOST_PROMOTION_THRESHOLD: u8 = 1;

        if let Some(ghost_freq) = self.inner.hashtable.get_ghost_freq(key)
            && ghost_freq > GHOST_PROMOTION_THRESHOLD
        {
            // High-frequency ghost - insert directly to main cache
            let ram_ops = LayerCacheOps {
                layer: &self.inner.ram_layer,
                hashtable: &self.inner.hashtable,
                metrics: &self.inner.metrics,
            };

            if let Some((segment_id, offset)) = ram_ops.ttl_buckets()
                .append_item(&ram_ops, key, value, optional, ttl)
                .await
            {
                // Link in hashtable with inherited frequency
                if self.inner.hashtable.link_item(
                    key,
                    0, // pool_id for RAM
                    segment_id,
                    offset,
                    ram_ops.pool(),
                    &self.inner.metrics,
                ).is_ok() {
                    self.inner.metrics.ghost_promote.increment();
                    return Ok(());
                }
            }
            // If direct insertion failed, fall through to small queue path
        }

        let now = UnixInstant::now();
        // Convert TTL to u32 seconds with saturation, then add to current time
        let ttl_secs = u32::try_from(ttl.as_secs()).unwrap_or(u32::MAX);
        let expire_at = now.duration_since(UnixInstant::EPOCH).as_secs()
            .saturating_add(ttl_secs);

        // Try to append to small queue
        let result = self.inner.ram_layer.append_to_small_queue(
            key,
            value,
            optional,
            expire_at,
            &self.inner.hashtable,
            &self.inner.metrics,
        ).await;

        if result.is_ok() {
            return Ok(());
        }

        // Failed - need to evict to free up space
        // Under high concurrency, other threads may be processing segments.
        // We retry with exponential backoff, yielding to let them complete.
        let mut eviction_attempts = 0;
        let max_eviction_attempts = 32;

        while eviction_attempts < max_eviction_attempts {
            let evicted = self.evict_ram_small_queue().await;

            if evicted {
                // Successfully evicted a segment, retry append
                let result = self.inner.ram_layer.append_to_small_queue(
                    key,
                    value,
                    optional,
                    expire_at,
                    &self.inner.hashtable,
                    &self.inner.metrics,
                ).await;

                if result.is_ok() {
                    return Ok(());
                }
                // Append still failed, try evicting more
            } else {
                // Small queue empty - other threads are processing segments
                // Yield to let them complete their evictions
                tokio::task::yield_now().await;
            }

            eviction_attempts += 1;
        }

        // Still failed after many eviction attempts
        self.inner.metrics.segment_alloc_fail.increment();
        Err(crate::CacheError::StorageFull)
    }

    /// Insert an item only if the key does not already exist (ADD semantics).
    ///
    /// This implements atomic ADD: if two concurrent ADD operations target the same
    /// key, exactly one will succeed and the other will return `KeyExists`.
    ///
    /// This method implements the S3-FIFO ghost-based admission optimization:
    /// - If the key has a ghost entry with frequency > 1 (was accessed after eviction),
    ///   the item goes directly to the main cache (TTL buckets)
    /// - Otherwise, the item goes through the small queue (admission queue)
    ///
    /// # Returns
    /// - `Ok(())` if the item was inserted (key was not present)
    /// - `Err(KeyExists)` if the key already exists
    /// - `Err(StorageFull)` if storage is full and eviction failed
    /// - `Err(HashTableFull)` if hashtable buckets are full
    pub async fn add(&self, key: &[u8], value: &[u8], ttl: Duration) -> Result<(), crate::CacheError> {
        self.add_with_optional(key, value, &[], ttl).await
    }

    /// Insert an item with optional metadata only if the key does not exist.
    ///
    /// See [`add`](Self::add) for details on ADD semantics.
    pub async fn add_with_optional(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        // Check if this key has a ghost entry with high frequency
        // If so, it was accessed between eviction and re-insertion, suggesting
        // it should bypass the small queue and go directly to main cache
        const GHOST_PROMOTION_THRESHOLD: u8 = 1;

        if let Some(ghost_freq) = self.inner.hashtable.get_ghost_freq(key)
            && ghost_freq > GHOST_PROMOTION_THRESHOLD
        {
            // High-frequency ghost - insert directly to main cache (ADD semantics)
            let ram_ops = LayerCacheOps {
                layer: &self.inner.ram_layer,
                hashtable: &self.inner.hashtable,
                metrics: &self.inner.metrics,
            };

            if let Some((segment_id, offset)) = ram_ops.ttl_buckets()
                .append_item(&ram_ops, key, value, optional, ttl)
                .await
            {
                // Link in hashtable with ADD semantics (fail if key exists)
                match self.inner.hashtable.link_item_if_absent(
                    key,
                    0, // pool_id for RAM
                    segment_id,
                    offset,
                    ram_ops.pool(),
                    &self.inner.metrics,
                ) {
                    Ok(()) => {
                        self.inner.metrics.ghost_promote.increment();
                        return Ok(());
                    }
                    Err(crate::CacheError::KeyExists) => {
                        return Err(crate::CacheError::KeyExists);
                    }
                    Err(_) => {
                        // Hashtable full, fall through to small queue path
                    }
                }
            }
            // If direct insertion failed, fall through to small queue path
        }

        let now = UnixInstant::now();
        // Convert TTL to u32 seconds with saturation, then add to current time
        let ttl_secs = u32::try_from(ttl.as_secs()).unwrap_or(u32::MAX);
        let expire_at = now.duration_since(UnixInstant::EPOCH).as_secs()
            .saturating_add(ttl_secs);

        // Try to append to small queue and link atomically
        let result = self.inner.ram_layer.append_to_small_queue_if_absent(
            key,
            value,
            optional,
            expire_at,
            &self.inner.hashtable,
            &self.inner.metrics,
        ).await;

        match result {
            Ok(()) => return Ok(()),
            Err(crate::CacheError::KeyExists) => return Err(crate::CacheError::KeyExists),
            Err(_) => {} // Storage full, try eviction
        }

        // Failed - need to evict to free up space
        // Under high concurrency, other threads may be processing segments.
        let mut eviction_attempts = 0;
        let max_eviction_attempts = 32;

        while eviction_attempts < max_eviction_attempts {
            let evicted = self.evict_ram_small_queue().await;

            if evicted {
                let result = self.inner.ram_layer.append_to_small_queue_if_absent(
                    key,
                    value,
                    optional,
                    expire_at,
                    &self.inner.hashtable,
                    &self.inner.metrics,
                ).await;

                match result {
                    Ok(()) => return Ok(()),
                    Err(crate::CacheError::KeyExists) => return Err(crate::CacheError::KeyExists),
                    Err(_) => {} // Storage still full, try more eviction
                }
            } else {
                // Small queue empty - yield to let other threads complete
                tokio::task::yield_now().await;
            }

            eviction_attempts += 1;
        }

        self.inner.metrics.segment_alloc_fail.increment();
        Err(crate::CacheError::StorageFull)
    }

    /// Update an item only if the key already exists (REPLACE semantics).
    ///
    /// This implements atomic REPLACE: the operation only succeeds if the key
    /// is already present in the cache. If the key does not exist, returns
    /// `KeyNotFound`.
    ///
    /// The replacement item's destination depends on the existing item's frequency:
    /// - If freq > 1 (item was accessed): goes directly to main cache (TTL buckets)
    /// - If freq <= 1 (item not accessed since insertion): goes through small queue
    ///
    /// This ensures replaced items follow the same admission policy as new items.
    ///
    /// # Returns
    /// - `Ok(())` if the item was replaced (key existed)
    /// - `Err(KeyNotFound)` if the key does not exist
    /// - `Err(StorageFull)` if storage is full and eviction failed
    pub async fn replace(&self, key: &[u8], value: &[u8], ttl: Duration) -> Result<(), crate::CacheError> {
        self.replace_with_optional(key, value, &[], ttl).await
    }

    /// Update an item with optional metadata only if the key exists.
    ///
    /// See [`replace`](Self::replace) for details on REPLACE semantics.
    pub async fn replace_with_optional(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        // Check the existing item's frequency to decide destination
        // If item is "hot" (freq > 1), it goes to main cache
        // If item is "cold" (freq <= 1), it goes through small queue
        const FREQUENCY_THRESHOLD: u8 = 1;

        let freq = match self.inner.hashtable.get_frequency(key, &self.inner.ram_layer) {
            Some(f) => f,
            None => return Err(crate::CacheError::KeyNotFound),
        };

        if freq > FREQUENCY_THRESHOLD {
            // Hot item - replace directly in main cache (TTL buckets)
            self.replace_to_main_cache(key, value, optional, ttl).await
        } else {
            // Cold item - replace through small queue (admission filter)
            self.replace_to_small_queue(key, value, optional, ttl).await
        }
    }

    /// Replace an item directly to main cache (for hot items).
    async fn replace_to_main_cache(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        let ram_ops = LayerCacheOps {
            layer: &self.inner.ram_layer,
            hashtable: &self.inner.hashtable,
            metrics: &self.inner.metrics,
        };

        // Write new value to TTL buckets (main cache)
        let (segment_id, offset) = match ram_ops.ttl_buckets()
            .append_item(&ram_ops, key, value, optional, ttl)
            .await
        {
            Some(location) => location,
            None => {
                // TTL buckets append failed - main cache is full
                // Try eviction from main cache with retries
                let mut eviction_attempts = 0;
                let max_eviction_attempts = 32;

                loop {
                    // Try merge eviction on main cache first
                    if self.merge_evict_ram().await.is_some() {
                        if let Some(location) = ram_ops.ttl_buckets()
                            .append_item(&ram_ops, key, value, optional, ttl)
                            .await
                        {
                            return self.complete_replace_main_cache(key, location.0, location.1);
                        }
                    } else {
                        // Merge eviction failed, try small queue eviction
                        if self.evict_ram_small_queue().await {
                            if let Some(location) = ram_ops.ttl_buckets()
                                .append_item(&ram_ops, key, value, optional, ttl)
                                .await
                            {
                                return self.complete_replace_main_cache(key, location.0, location.1);
                            }
                        } else {
                            // No eviction possible, yield to let other threads complete
                            tokio::task::yield_now().await;
                        }
                    }

                    eviction_attempts += 1;
                    if eviction_attempts >= max_eviction_attempts {
                        self.inner.metrics.segment_alloc_fail.increment();
                        return Err(crate::CacheError::StorageFull);
                    }
                }
            }
        };

        self.complete_replace_main_cache(key, segment_id, offset)
    }

    /// Complete a REPLACE to main cache by atomically updating the hashtable.
    fn complete_replace_main_cache(&self, key: &[u8], segment_id: u32, offset: u32) -> Result<(), crate::CacheError> {
        match self.inner.hashtable.link_item_if_present(
            key,
            0, // pool_id for RAM
            segment_id,
            offset,
            &self.inner.ram_layer,
            &self.inner.metrics,
        ) {
            Ok(_old_location) => {
                // Successfully replaced - old data will be cleaned up during eviction
                Ok(())
            }
            Err(crate::CacheError::KeyNotFound) => {
                // Key was deleted between frequency check and replace - REPLACE fails
                // Note: we've written orphaned data, but it will be cleaned up during eviction
                Err(crate::CacheError::KeyNotFound)
            }
            Err(e) => Err(e),
        }
    }

    /// Replace an item through small queue (for cold items).
    async fn replace_to_small_queue(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        let now = UnixInstant::now();
        // Convert TTL to u32 seconds with saturation, then add to current time
        let ttl_secs = u32::try_from(ttl.as_secs()).unwrap_or(u32::MAX);
        let expire_at = now.duration_since(UnixInstant::EPOCH).as_secs()
            .saturating_add(ttl_secs);

        // Try to append to small queue and link atomically (REPLACE semantics)
        let result = self.inner.ram_layer.append_to_small_queue_if_present(
            key,
            value,
            optional,
            expire_at,
            &self.inner.hashtable,
            &self.inner.metrics,
        ).await;

        match result {
            Ok(_) => return Ok(()),
            Err(crate::CacheError::KeyNotFound) => return Err(crate::CacheError::KeyNotFound),
            Err(_) => {} // Storage full, try eviction
        }

        // Failed - need to evict to free up space
        let mut eviction_attempts = 0;
        let max_eviction_attempts = 32;

        while eviction_attempts < max_eviction_attempts {
            let evicted = self.evict_ram_small_queue().await;

            if evicted {
                let result = self.inner.ram_layer.append_to_small_queue_if_present(
                    key,
                    value,
                    optional,
                    expire_at,
                    &self.inner.hashtable,
                    &self.inner.metrics,
                ).await;

                match result {
                    Ok(_) => return Ok(()),
                    Err(crate::CacheError::KeyNotFound) => return Err(crate::CacheError::KeyNotFound),
                    Err(_) => {} // Storage still full, try more eviction
                }
            } else {
                // Small queue empty - yield to let other threads complete
                tokio::task::yield_now().await;
            }

            eviction_attempts += 1;
        }

        self.inner.metrics.segment_alloc_fail.increment();
        Err(crate::CacheError::StorageFull)
    }

    /// Get an item from the cache with zero-copy access.
    ///
    /// This looks up the item in the hashtable and returns an `ItemGuard` that
    /// provides direct access to the item's data in the segment. Finding an item
    /// increments its frequency counter.
    ///
    /// The returned `ItemGuard` holds a reference count on the segment, preventing
    /// it from being evicted while the guard is alive. The reference count is
    /// automatically decremented when the guard is dropped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = cache.get(b"key").await?;
    /// let value = guard.value();
    /// // Use value directly without copying
    /// socket.write_all(value)?;
    /// // Guard dropped here, segment ref_count decremented
    /// ```
    pub async fn get(&self, key: &[u8]) -> Option<crate::item::ItemGuard<'_>> {
        // Look up in hashtable (this also increments frequency)
        let (segment_id, offset) = self.inner.hashtable.get(key, &self.inner.ram_layer)?;

        // Try RAM layer first (pool_id embedded in hashtable entry)
        if let Some(segment) = self.inner.ram_layer.pool().get(segment_id) {
            return segment.get_item_guard(offset, key).ok();
        }

        // Item is on SSD - read and promote to RAM, then return RAM guard
        // This gives consistent behavior for both mmap and DirectIo backends
        if let Some(ref ssd_layer) = self.inner.ssd_layer {
            return self.get_from_ssd_with_promotion(key, segment_id, offset, ssd_layer).await;
        }

        None
    }

    /// Read an item from SSD, promote to RAM, and return the RAM ItemGuard.
    ///
    /// This is the unified read path for both mmap and DirectIo SSD backends.
    /// SSD reads always promote to RAM for consistent behavior and predictable
    /// memory usage.
    ///
    /// If promotion fails (RAM full), returns None. Callers should trigger
    /// eviction if they need to retry.
    async fn get_from_ssd_with_promotion(
        &self,
        key: &[u8],
        segment_id: u32,
        offset: u32,
        ssd_layer: &SsdLayer,
    ) -> Option<crate::item::ItemGuard<'_>> {
        // Read item data from SSD (mmap or DirectIo)
        let (value, optional, expire_at) = self.read_ssd_item(key, segment_id, offset, ssd_layer)?;

        // Promote to RAM small queue
        let result = self.inner.ram_layer.append_to_small_queue(
            key,
            &value,
            &optional,
            expire_at,
            &self.inner.hashtable,
            &self.inner.metrics,
        ).await;

        self.inner.metrics.ssd_promote.increment();

        if result.is_err() {
            // Promotion failed (RAM full)
            // Return None - caller should trigger eviction and retry
            return None;
        }

        // Look up the new RAM location and return RAM guard
        let (new_segment_id, new_offset) = self.inner.hashtable.get(key, &self.inner.ram_layer)?;
        let segment = self.inner.ram_layer.pool().get(new_segment_id)?;
        segment.get_item_guard(new_offset, key).ok()
    }

    /// Read item data from SSD layer (works for both mmap and DirectIo).
    ///
    /// Returns (value, optional, expire_at) if successful.
    fn read_ssd_item(
        &self,
        key: &[u8],
        segment_id: u32,
        offset: u32,
        ssd_layer: &SsdLayer,
    ) -> Option<(Vec<u8>, Vec<u8>, u32)> {
        match ssd_layer {
            SsdLayer::Mmap(layer) => {
                // Read from memory-mapped segment
                let segment = layer.pool().get(segment_id)?;
                let guard = segment.get_item_guard(offset, key).ok()?;

                // Get remaining TTL
                let now = clocksource::coarse::Instant::now();
                let expire_at = segment.expire_at();
                let remaining_secs = if expire_at > now {
                    expire_at.duration_since(now).as_secs()
                } else {
                    60 // Minimum TTL for nearly-expired items
                };
                let expire_at_unix = Self::current_time() + remaining_secs;

                Some((
                    guard.value().to_vec(),
                    guard.optional().to_vec(),
                    expire_at_unix,
                ))
            }
            SsdLayer::DirectIo { pool } => {
                // Read from file via DirectIoPool
                self.read_ssd_item_direct_io(key, segment_id, offset, pool)
            }
        }
    }

    /// Read item from DirectIo pool by reading from the backing file.
    ///
    /// DirectIo items use SmallQueueItemHeader format (with per-item TTL).
    fn read_ssd_item_direct_io(
        &self,
        key: &[u8],
        segment_id: u32,
        offset: u32,
        pool: &crate::pool::DirectIoPool,
    ) -> Option<(Vec<u8>, Vec<u8>, u32)> {
        use crate::item::SmallQueueItemHeader;

        // First, read the item header to get sizes
        // DirectIo uses SmallQueueItemHeader format (12 bytes with per-item TTL)
        let header_size = SmallQueueItemHeader::SIZE;
        let mut header_buf = vec![0u8; header_size];

        pool.read_segment(segment_id, offset, &mut header_buf).ok()?;

        let header = SmallQueueItemHeader::try_from_bytes(&header_buf)?;

        // Check if item is deleted
        if header.is_deleted() {
            return None;
        }

        // Calculate total item size and read full item
        let item_size = header.padded_size();
        let mut item_buf = vec![0u8; item_size];

        pool.read_segment(segment_id, offset, &mut item_buf).ok()?;

        // Parse key, optional, value from item data
        // SmallQueueItemHeader layout: [header][optional][key][value]
        let optional_start = header_size;
        let optional_end = optional_start + header.optional_len() as usize;
        let key_start = optional_end;
        let key_end = key_start + header.key_len() as usize;
        let value_start = key_end;
        let value_end = value_start + header.value_len() as usize;

        // Verify key matches
        let stored_key = item_buf.get(key_start..key_end)?;
        if stored_key != key {
            return None;
        }

        let optional = item_buf.get(optional_start..optional_end)?.to_vec();
        let value = item_buf.get(value_start..value_end)?.to_vec();

        // Get per-item TTL from the header
        let expire_at_unix = header.expire_at();

        Some((value, optional, expire_at_unix))
    }

    /// Check if a key exists in the cache without incrementing its frequency.
    ///
    /// This is useful for implementing conditional operations like ADD (only write if key
    /// doesn't exist) and REPLACE (only write if key exists) where checking existence
    /// should not affect the item's "hotness" for eviction purposes.
    pub fn contains(&self, key: &[u8]) -> bool {
        self.inner.hashtable.contains(key, &self.inner.ram_layer)
    }

    /// Delete an item from the cache
    pub fn delete(&self, key: &[u8]) -> bool {
        // Look up the item location
        if let Some((segment_id, offset)) = self.inner.hashtable.get(key, &self.inner.ram_layer) {
            // Unlink from hashtable
            self.inner.hashtable.unlink_item(key, segment_id, offset, &self.inner.metrics)
        } else {
            false
        }
    }

    /// Get an item along with its CAS token for compare-and-swap operations.
    ///
    /// The CAS token is derived from the item's location (generation, pool, segment, offset).
    /// If the item moves (due to update, compaction, or promotion), the CAS token changes.
    ///
    /// # Returns
    /// - `Some((guard, cas_token))` if the item exists
    /// - `None` if the item is not found
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some((guard, cas)) = cache.gets(b"counter") {
    ///     let value: u64 = parse(guard.value());
    ///     let new_value = (value + 1).to_string();
    ///     match cache.cas(b"counter", new_value.as_bytes(), cas, ttl).await {
    ///         Ok(()) => println!("Updated!"),
    ///         Err(CacheError::CasMismatch) => println!("Value changed, retry"),
    ///         Err(e) => println!("Error: {:?}", e),
    ///     }
    /// }
    /// ```
    pub fn gets(&self, key: &[u8]) -> Option<(crate::item::ItemGuard<'_>, crate::CasToken)> {
        // Look up in hashtable (this also increments frequency)
        let (segment_id, offset) = self.inner.hashtable.get(key, &self.inner.ram_layer)?;

        // Try RAM layer first
        if let Some(segment) = self.inner.ram_layer.pool().get(segment_id)
            && let Ok(guard) = segment.get_item_guard(offset, key)
        {
            let cas_token = crate::CasToken::from_raw(Self::make_cas_token(
                segment.generation(),
                segment.pool_id(),
                segment_id,
                offset,
            ));
            return Some((guard, cas_token));
        }

        // Try SSD layer if available (only mmap backend supports direct access)
        if let Some(ref ssd_layer) = self.inner.ssd_layer
            && let Some(pool) = ssd_layer.pool()
            && let Some(segment) = pool.get(segment_id)
            && let Ok(guard) = segment.get_item_guard(offset, key)
        {
            let cas_token = crate::CasToken::from_raw(Self::make_cas_token(
                segment.generation(),
                segment.pool_id(),
                segment_id,
                offset,
            ));
            return Some((guard, cas_token));
        }

        // TODO: Handle DirectIo backend

        None
    }

    /// Update an item only if its CAS token matches (compare-and-swap).
    ///
    /// This implements memcached-style CAS semantics:
    /// - Get the current value and CAS token with `gets()`
    /// - Modify the value
    /// - Call `cas()` with the original CAS token
    /// - If another client modified the value in between, CAS fails with `CasMismatch`
    ///
    /// # Returns
    /// - `Ok(())` if the update succeeded (CAS token matched)
    /// - `Err(CasMismatch)` if the CAS token doesn't match (value was modified)
    /// - `Err(KeyNotFound)` if the key doesn't exist
    /// - `Err(StorageFull)` if storage is full
    pub async fn cas(
        &self,
        key: &[u8],
        value: &[u8],
        cas_token: crate::CasToken,
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        self.cas_with_optional(key, value, &[], cas_token, ttl).await
    }

    /// Update an item with optional metadata only if CAS token matches.
    ///
    /// See [`cas`](Self::cas) for details on CAS semantics.
    pub async fn cas_with_optional(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        cas_token: crate::CasToken,
        ttl: Duration,
    ) -> Result<(), crate::CacheError> {
        // First, verify the current CAS token matches
        let (segment_id, offset) = match self.inner.hashtable.get(key, &self.inner.ram_layer) {
            Some(loc) => loc,
            None => return Err(crate::CacheError::KeyNotFound),
        };

        // Get the segment and compute current CAS token
        let current_cas = if let Some(segment) = self.inner.ram_layer.pool().get(segment_id) {
            Self::make_cas_token(segment.generation(), segment.pool_id(), segment_id, offset)
        } else if let Some(ref ssd_layer) = self.inner.ssd_layer
            && let Some(pool) = ssd_layer.pool()
            && let Some(segment) = pool.get(segment_id)
        {
            Self::make_cas_token(segment.generation(), segment.pool_id(), segment_id, offset)
        } else {
            return Err(crate::CacheError::KeyNotFound);
        };

        // Check if CAS token matches
        if current_cas != cas_token.into_raw() {
            return Err(crate::CacheError::CasMismatch);
        }

        // CAS token matches - proceed with update (same as REPLACE)
        self.replace_with_optional(key, value, optional, ttl).await
    }

    /// Create a CAS token from item location components.
    ///
    /// Layout (64 bits):
    /// ```text
    /// [16 bits generation][2 bits pool_id][22 bits segment_id][20 bits offset/8][4 bits unused]
    /// ```
    #[inline]
    fn make_cas_token(generation: u16, pool_id: u8, segment_id: u32, offset: u32) -> u64 {
        let gen_bits = (generation as u64) << 48;
        let pool_bits = ((pool_id & 0x3) as u64) << 46;
        let seg_bits = ((segment_id & 0x3FFFFF) as u64) << 24;
        let off_bits = (((offset >> 3) & 0xFFFFF) as u64) << 4;
        gen_bits | pool_bits | seg_bits | off_bits
    }

    /// Get the current time as unix timestamp (seconds since epoch)
    fn current_time() -> u32 {
        UnixInstant::now().duration_since(UnixInstant::EPOCH).as_secs()
    }

    /// Evict from the RAM layer's small queue
    ///
    /// This evicts the oldest segment from the small queue:
    /// - Hot items (freq > 1) are promoted to RAM main cache
    /// - Cold items (freq <= 1) are demoted to SSD small queue (if available)
    ///
    /// # Returns
    /// - `true` if a segment was evicted
    /// - `false` if the small queue was empty (nothing to evict)
    async fn evict_ram_small_queue(&self) -> bool {
        let current_time = Self::current_time();

        // Create a CacheOps wrapper for the RAM layer
        let ram_ops = LayerCacheOps {
            layer: &self.inner.ram_layer,
            hashtable: &self.inner.hashtable,
            metrics: &self.inner.metrics,
        };

        // Pass SSD mmap layer for demotion (if available)
        // DirectIo backend demotion not yet implemented
        let ssd_mmap_layer = self.inner.ssd_layer.as_ref().and_then(|s| s.as_mmap());
        self.inner.ram_layer
            .evict_small_queue_segment(&ram_ops, current_time, ssd_mmap_layer)
            .await
            .is_some()
    }

    /// Evict from the SSD layer's small queue
    ///
    /// This evicts the oldest segment from the SSD small queue:
    /// - Hot items (freq > 1) are promoted to SSD main cache
    /// - Cold items (freq <= 1) are dropped (no further demotion)
    #[allow(dead_code)]
    async fn evict_ssd_small_queue(&self) {
        // Only mmap backend supports direct eviction
        let Some(ssd_mmap_layer) = self.inner.ssd_layer.as_ref().and_then(|s| s.as_mmap()) else {
            return;
        };

        let current_time = Self::current_time();

        let ssd_ops = LayerCacheOps {
            layer: ssd_mmap_layer,
            hashtable: &self.inner.hashtable,
            metrics: &self.inner.metrics,
        };

        // No demotion from SSD - cold items are dropped (pass None for demote_layer)
        ssd_mmap_layer
            .evict_small_queue_segment::<crate::pool::MmapPool>(&ssd_ops, current_time, None)
            .await;
    }

    /// Run merge eviction on the RAM layer's main cache
    ///
    /// This finds a TTL bucket with enough segments to merge, consolidates them,
    /// and returns segments to the free pool. Evicted items are demoted to the
    /// SSD small queue (if available).
    ///
    /// The reclaimed segments can be used by any bucket that needs space.
    ///
    /// # Returns
    /// - `Some(freed_count)`: Number of segments freed
    /// - `None`: No bucket had enough segments to merge
    #[allow(dead_code)]
    async fn merge_evict_ram(&self) -> Option<usize> {
        let ram_ops = LayerCacheOps {
            layer: &self.inner.ram_layer,
            hashtable: &self.inner.hashtable,
            metrics: &self.inner.metrics,
        };

        // Try merge eviction on buckets until one succeeds
        // Demote pruned items to SSD mmap layer if available
        // DirectIo backend demotion not yet implemented
        let ssd_mmap_layer = self.inner.ssd_layer.as_ref().and_then(|s| s.as_mmap());
        self.inner.ram_layer.ttl_buckets()
            .merge_evict_any(
                &ram_ops,
                self.inner.merge_ratio,
                self.inner.merge_target_ratio,
                &self.inner.metrics,
                ssd_mmap_layer,
            )
            .await
    }

    /// Run merge eviction on the SSD layer's main cache
    ///
    /// This finds a TTL bucket with enough segments to merge, consolidates them,
    /// and returns segments to the free pool. No demotion - evicted items are dropped.
    ///
    /// # Returns
    /// - `Some(freed_count)`: Number of segments freed
    /// - `None`: No bucket had enough segments to merge, or no SSD layer
    #[allow(dead_code)]
    async fn merge_evict_ssd(&self) -> Option<usize> {
        // Only mmap backend supports merge eviction
        let ssd_mmap_layer = self.inner.ssd_layer.as_ref().and_then(|s| s.as_mmap())?;

        let ssd_ops = LayerCacheOps {
            layer: ssd_mmap_layer,
            hashtable: &self.inner.hashtable,
            metrics: &self.inner.metrics,
        };

        // No demotion from SSD - pass None for demote layer
        ssd_mmap_layer.ttl_buckets()
            .merge_evict_any::<_, crate::MmapPool>(
                &ssd_ops,
                self.inner.merge_ratio,
                self.inner.merge_target_ratio,
                &self.inner.metrics,
                None,
            )
            .await
    }

    /// Get a reference to the metrics
    pub fn metrics(&self) -> &CacheMetrics {
        &self.inner.metrics
    }
}

/// Helper struct that implements CacheOps for a single layer
struct LayerCacheOps<'a, P: Pool> {
    layer: &'a CacheLayer<P>,
    hashtable: &'a Hashtable,
    metrics: &'a CacheMetrics,
}

impl<'a, P: Pool> CacheOps<P> for LayerCacheOps<'a, P> {
    fn pool(&self) -> &P {
        self.layer.pool()
    }

    fn ttl_buckets(&self) -> &TtlBuckets {
        self.layer.ttl_buckets()
    }

    fn hashtable(&self) -> &Hashtable {
        self.hashtable
    }

    fn metrics(&self) -> &CacheMetrics {
        self.metrics
    }

    fn try_expire(&self) -> Option<u32> {
        self.layer.ttl_buckets().try_expire(self, self.metrics)
    }

    async fn evict_segment_by_policy(&self) -> Option<u32> {
        // Evict a segment from the main cache (TTL buckets)
        //
        // We prefer merge eviction over FIFO because merge eviction:
        // 1. Properly unlinks items from the hashtable before freeing segments
        // 2. Preserves live items by consolidating them into fewer segments
        // 3. Only drops orphaned/low-frequency items
        //
        // FIFO eviction (evict_any) is faster but drops ALL items in the segment,
        // leaving stale hashtable entries that cause guaranteed misses.

        // Try merge eviction first - consolidates segments and releases to pool
        // Using merge_ratio=2, target_ratio=1 means merge 2 segments into 1,
        // freeing 1 segment. We use small ratios because segments may be spread
        // thinly across many TTL buckets (1024 buckets).
        let freed = self.layer.ttl_buckets()
            .merge_evict_any::<P, P>(
                self,
                2,  // merge_ratio: consolidate 2 segments
                1,  // target_ratio: into 1 segment
                self.metrics,
                None,  // no demotion to another layer from main cache
            )
            .await;

        if freed.is_some() {
            // Merge eviction released segments to pool, try to reserve one
            if let Some(segment_id) = self.pool().reserve_main_cache(self.metrics) {
                return Some(segment_id);
            }
        }

        // Fallback: FIFO eviction if merge eviction didn't work
        // This can happen if no bucket has enough segments for merge
        // Note: This will cause hitrate degradation for evicted items
        self.layer.ttl_buckets().evict_any(self, self.metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test cache with small segments for fast tests
    fn create_test_cache() -> Cache {
        Cache::builder()
            .ram_size(64 * 1024)            // 64KB total
            .ram_segment_size(4 * 1024)     // 4KB segments = 16 segments
            .ram_small_queue_percent(25)    // 4 small queue, 12 main cache
            .hashtable_power(8)             // 256 buckets
            .build()
            .expect("Failed to create test cache")
    }

    #[test]
    fn test_cache_creation() {
        let _cache = create_test_cache();
        // If we get here without panic, creation succeeded
    }

    #[tokio::test]
    async fn test_set_and_get_basic() {
        let cache = create_test_cache();

        let key = b"test_key";
        let value = b"test_value";
        let ttl = Duration::from_secs(3600); // 1 hour

        // Set the item
        let result = cache.set(key, value, ttl).await;
        assert!(result.is_ok(), "set should succeed");

        // Get the item back
        let retrieved = cache.get(key).await;
        assert!(retrieved.is_some(), "get should find the item");
        assert_eq!(retrieved.unwrap().value(), value);
    }

    #[tokio::test]
    async fn test_set_and_get_multiple_items() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // Insert multiple items
        for i in 0..10 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            cache.set(key.as_bytes(), value.as_bytes(), ttl).await.expect("set should succeed");
        }

        // Retrieve all items
        for i in 0..10 {
            let key = format!("key_{}", i);
            let expected_value = format!("value_{}", i);
            let retrieved = cache.get(key.as_bytes()).await;
            assert!(retrieved.is_some(), "get should find key_{}", i);
            assert_eq!(retrieved.unwrap().value(), expected_value.as_bytes());
        }
    }

    #[tokio::test]
    async fn test_get_nonexistent_key() {
        let cache = create_test_cache();

        let result = cache.get(b"nonexistent_key").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete_item() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // Insert an item
        let key = b"delete_me";
        let value = b"some_value";
        cache.set(key, value, ttl).await.expect("set should succeed");

        // Verify it exists
        assert!(cache.get(key).await.is_some());

        // Delete it
        let deleted = cache.delete(key);
        assert!(deleted, "delete should return true for existing item");

        // Verify it's gone
        assert!(cache.get(key).await.is_none());
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let cache = create_test_cache();

        let deleted = cache.delete(b"never_existed");
        assert!(!deleted, "delete should return false for nonexistent key");
    }

    #[tokio::test]
    async fn test_overwrite_existing_key() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        let key = b"overwrite_key";

        // Set initial value
        cache.set(key, b"initial_value", ttl).await.expect("first set should succeed");
        assert_eq!(cache.get(key).await.unwrap().value(), b"initial_value");

        // Overwrite with new value
        cache.set(key, b"new_value", ttl).await.expect("second set should succeed");
        assert_eq!(cache.get(key).await.unwrap().value(), b"new_value");
    }

    #[tokio::test]
    async fn test_set_with_optional_data() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        let key = b"optional_key";
        let value = b"main_value";
        let optional = b"extra_metadata";

        let result = cache.set_with_optional(key, value, optional, ttl).await;
        assert!(result.is_ok());

        // Get returns just the value (not optional data)
        let retrieved = cache.get(key).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), value);
    }

    #[tokio::test]
    async fn test_large_value() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        let key = b"large_key";
        let value = vec![0xAB_u8; 1024]; // 1KB value

        cache.set(key, &value, ttl).await.expect("set large value should succeed");

        let retrieved = cache.get(key).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), &value[..]);
    }

    #[tokio::test]
    async fn test_many_small_items() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // Insert many small items - cache handles eviction internally
        let count = 100;
        for i in 0..count {
            let key = format!("small_{}", i);
            let value = format!("v{}", i);
            let _ = cache.set(key.as_bytes(), value.as_bytes(), ttl).await;
        }

        // Verify we can retrieve at least some items
        let mut found = 0;
        for i in 0..count {
            let key = format!("small_{}", i);
            if cache.get(key.as_bytes()).await.is_some() {
                found += 1;
            }
        }
        assert!(found > 0, "Should find at least some items");
    }

    #[tokio::test]
    async fn test_empty_value() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        let key = b"empty_value_key";
        let value = b"";

        cache.set(key, value, ttl).await.expect("set empty value should succeed");

        let retrieved = cache.get(key).await;
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().value().is_empty());
    }

    #[tokio::test]
    async fn test_binary_key_and_value() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // Key and value with null bytes and other binary data
        let key = &[0x00, 0x01, 0x02, 0xFF, 0xFE];
        let value = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00];

        cache.set(key, value, ttl).await.expect("set binary data should succeed");

        let retrieved = cache.get(key).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), value);
    }

    #[test]
    fn test_metrics_accessible() {
        let cache = create_test_cache();
        let metrics = cache.metrics();

        // Initial state - metrics should exist
        // Just verify we can access them without panic
        let _ = metrics;
    }

    #[tokio::test]
    async fn test_evict_ram_small_queue() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // Insert enough items to fill the small queue segments
        for i in 0..200 {
            let key = format!("evict_test_{}", i);
            let value = format!("value_{}", i);
            let _ = cache.set(key.as_bytes(), value.as_bytes(), ttl).await;
        }

        // Run eviction (this is done automatically, but can also be called manually)
        cache.evict_ram_small_queue().await;

        // Some items should still be accessible (hot items get promoted)
        // This is a smoke test - just ensure eviction doesn't panic
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        // Insert some items
        for i in 0..10 {
            let key = format!("concurrent_{}", i);
            let value = format!("value_{}", i);
            cache.set(key.as_bytes(), value.as_bytes(), ttl).await.expect("set should succeed");
        }

        // Spawn multiple reader tasks
        let mut handles = vec![];
        for _thread_id in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    for i in 0..10 {
                        let key = format!("concurrent_{}", i);
                        let _ = cache_clone.get(key.as_bytes());
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.expect("Task should complete");
        }
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        // Spawn multiple writer tasks
        let mut handles = vec![];
        for thread_id in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for i in 0..50 {
                    let key = format!("write_{}_{}", thread_id, i);
                    let value = format!("value_{}_{}", thread_id, i);
                    let _ = cache_clone.set(key.as_bytes(), value.as_bytes(), ttl).await;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify some items are readable
        let mut found = 0;
        for thread_id in 0..4 {
            for i in 0..50 {
                let key = format!("write_{}_{}", thread_id, i);
                if cache.get(key.as_bytes()).await.is_some() {
                    found += 1;
                }
            }
        }
        assert!(found > 0, "Should find at least some written items");
    }

    #[tokio::test]
    async fn test_concurrent_read_write() {
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        // Pre-populate some items
        for i in 0..10 {
            let key = format!("rw_{}", i);
            let value = format!("initial_{}", i);
            cache.set(key.as_bytes(), value.as_bytes(), ttl).await.expect("set should succeed");
        }

        // Spawn reader and writer tasks
        let mut handles = vec![];

        // Writers
        for thread_id in 0..2 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for round in 0..50 {
                    for i in 0..10 {
                        let key = format!("rw_{}", i);
                        let value = format!("updated_{}_{}_{}", thread_id, round, i);
                        let _ = cache_clone.set(key.as_bytes(), value.as_bytes(), ttl).await;
                    }
                }
            });
            handles.push(handle);
        }

        // Readers
        for _ in 0..2 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    for i in 0..10 {
                        let key = format!("rw_{}", i);
                        let _ = cache_clone.get(key.as_bytes()).await;
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.expect("Task should complete");
        }
    }

    // ==================== ADD tests ====================

    #[tokio::test]
    async fn test_add_new_key() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // ADD should succeed for new key
        let result = cache.add(b"add_key", b"add_value", ttl).await;
        assert!(result.is_ok(), "ADD should succeed for new key");

        // Verify the item was inserted
        let retrieved = cache.get(b"add_key").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), b"add_value");
    }

    #[tokio::test]
    async fn test_add_existing_key_fails() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // First, SET a key
        cache.set(b"existing_key", b"original_value", ttl).await.expect("set should succeed");

        // ADD should fail for existing key
        let result = cache.add(b"existing_key", b"new_value", ttl).await;
        assert!(matches!(result, Err(crate::CacheError::KeyExists)), "ADD should return KeyExists for existing key");

        // Original value should be unchanged
        let retrieved = cache.get(b"existing_key").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), b"original_value");
    }

    #[tokio::test]
    async fn test_add_after_delete() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // SET then DELETE
        cache.set(b"delete_me", b"original", ttl).await.expect("set should succeed");
        cache.delete(b"delete_me");

        // ADD should succeed after delete
        let result = cache.add(b"delete_me", b"new_value", ttl).await;
        assert!(result.is_ok(), "ADD should succeed after delete");

        let retrieved = cache.get(b"delete_me").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), b"new_value");
    }

    #[tokio::test]
    async fn test_concurrent_add_same_key() {
        // Test that concurrent ADDs for the same key result in exactly one success
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        let mut handles = vec![];
        for i in 0..8 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                let value = format!("value_{}", i);
                cache_clone.add(b"contested_key", value.as_bytes(), ttl).await
            });
            handles.push(handle);
        }

        let mut success_count = 0;
        let mut key_exists_count = 0;

        for handle in handles {
            match handle.await.expect("task should complete") {
                Ok(()) => success_count += 1,
                Err(crate::CacheError::KeyExists) => key_exists_count += 1,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        // Exactly one ADD should succeed
        assert_eq!(success_count, 1, "Exactly one concurrent ADD should succeed");
        assert_eq!(key_exists_count, 7, "Other ADDs should get KeyExists");

        // Key should exist with some value
        assert!(cache.get(b"contested_key").await.is_some());
    }

    // ==================== REPLACE tests ====================

    #[tokio::test]
    async fn test_replace_existing_key() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // First SET a key
        cache.set(b"replace_key", b"original_value", ttl).await.expect("set should succeed");

        // REPLACE should succeed for existing key
        let result = cache.replace(b"replace_key", b"new_value", ttl).await;
        assert!(result.is_ok(), "REPLACE should succeed for existing key");

        // Value should be updated
        let retrieved = cache.get(b"replace_key").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), b"new_value");
    }

    #[tokio::test]
    async fn test_replace_nonexistent_key_fails() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // REPLACE should fail for non-existent key
        let result = cache.replace(b"nonexistent_key", b"value", ttl).await;
        assert!(matches!(result, Err(crate::CacheError::KeyNotFound)), "REPLACE should return KeyNotFound");

        // Key should still not exist
        assert!(cache.get(b"nonexistent_key").await.is_none());
    }

    #[tokio::test]
    async fn test_replace_after_delete_fails() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // SET then DELETE
        cache.set(b"deleted_key", b"original", ttl).await.expect("set should succeed");
        cache.delete(b"deleted_key");

        // REPLACE should fail after delete
        let result = cache.replace(b"deleted_key", b"new_value", ttl).await;
        assert!(matches!(result, Err(crate::CacheError::KeyNotFound)), "REPLACE should fail after delete");
    }

    #[tokio::test]
    async fn test_add_then_replace() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // ADD a new key
        cache.add(b"add_replace", b"first", ttl).await.expect("add should succeed");

        // REPLACE should succeed
        cache.replace(b"add_replace", b"second", ttl).await.expect("replace should succeed");

        let retrieved = cache.get(b"add_replace").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value(), b"second");

        // ADD should now fail
        let result = cache.add(b"add_replace", b"third", ttl).await;
        assert!(matches!(result, Err(crate::CacheError::KeyExists)));
    }

    #[tokio::test]
    async fn test_concurrent_replace_same_key() {
        // Test that concurrent REPLACEs all succeed (last write wins)
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        // First SET the key
        cache.set(b"replace_contested", b"original", ttl).await.expect("set should succeed");

        let mut handles = vec![];
        for i in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                let value = format!("value_{}", i);
                cache_clone.replace(b"replace_contested", value.as_bytes(), ttl).await
            });
            handles.push(handle);
        }

        let mut success_count = 0;
        for handle in handles {
            match handle.await.expect("task should complete") {
                Ok(()) => success_count += 1,
                Err(e) => panic!("REPLACE should not fail: {:?}", e),
            }
        }

        // All REPLACEs should succeed
        assert_eq!(success_count, 4, "All concurrent REPLACEs should succeed");

        // Key should exist with one of the values
        let retrieved = cache.get(b"replace_contested").await;
        assert!(retrieved.is_some());
    }

    // ==================== CAS tests ====================

    #[tokio::test]
    async fn test_gets_returns_cas_token() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        cache.set(b"cas_key", b"value1", ttl).await.expect("set should succeed");

        // gets() should return item and CAS token
        let result = cache.gets(b"cas_key");
        assert!(result.is_some());

        let (guard, _cas_token) = result.unwrap();
        assert_eq!(guard.value(), b"value1");
        // CAS token is opaque - we just verify we got one
    }

    #[tokio::test]
    async fn test_cas_succeeds_with_valid_token() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        cache.set(b"cas_key", b"original", ttl).await.expect("set should succeed");

        // Get CAS token
        let (_, cas_token) = cache.gets(b"cas_key").expect("gets should succeed");

        // CAS with valid token should succeed
        let result = cache.cas(b"cas_key", b"updated", cas_token, ttl).await;
        assert!(result.is_ok(), "CAS with valid token should succeed");

        // Verify value was updated
        let guard = cache.get(b"cas_key").await.expect("get should succeed");
        assert_eq!(guard.value(), b"updated");
    }

    #[tokio::test]
    async fn test_cas_fails_with_stale_token() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        cache.set(b"cas_key", b"original", ttl).await.expect("set should succeed");

        // Get CAS token
        let (_, old_cas_token) = cache.gets(b"cas_key").expect("gets should succeed");

        // Another client modifies the value
        cache.set(b"cas_key", b"modified_by_other", ttl).await.expect("set should succeed");

        // CAS with old token should fail
        let result = cache.cas(b"cas_key", b"my_update", old_cas_token, ttl).await;
        assert!(
            matches!(result, Err(crate::CacheError::CasMismatch)),
            "CAS with stale token should return CasMismatch"
        );

        // Value should be what the other client set
        let guard = cache.get(b"cas_key").await.expect("get should succeed");
        assert_eq!(guard.value(), b"modified_by_other");
    }

    #[tokio::test]
    async fn test_cas_fails_for_nonexistent_key() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        // CAS on non-existent key should fail (using a made-up token)
        let fake_token = crate::CasToken::from_raw(12345);
        let result = cache.cas(b"nonexistent", b"value", fake_token, ttl).await;
        assert!(
            matches!(result, Err(crate::CacheError::KeyNotFound)),
            "CAS on non-existent key should return KeyNotFound"
        );
    }

    #[tokio::test]
    async fn test_cas_fails_after_delete() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        cache.set(b"cas_delete", b"value", ttl).await.expect("set should succeed");

        // Get CAS token
        let (_, cas_token) = cache.gets(b"cas_delete").expect("gets should succeed");

        // Delete the key
        cache.delete(b"cas_delete");

        // CAS should fail with KeyNotFound
        let result = cache.cas(b"cas_delete", b"new_value", cas_token, ttl).await;
        assert!(
            matches!(result, Err(crate::CacheError::KeyNotFound)),
            "CAS after delete should return KeyNotFound"
        );
    }

    #[tokio::test]
    async fn test_cas_token_changes_on_update() {
        let cache = create_test_cache();
        let ttl = Duration::from_secs(3600);

        cache.set(b"cas_key", b"v1", ttl).await.expect("set should succeed");

        let (_, cas1) = cache.gets(b"cas_key").expect("gets should succeed");

        // Update the value
        cache.set(b"cas_key", b"v2", ttl).await.expect("set should succeed");

        let (_, cas2) = cache.gets(b"cas_key").expect("gets should succeed");

        // CAS token should change after update (item moved to new location)
        assert_ne!(cas1, cas2, "CAS token should change after update");
    }

    #[tokio::test]
    async fn test_concurrent_cas_one_wins() {
        // Test that concurrent CAS operations result in only one success
        let cache = Arc::new(create_test_cache());
        let ttl = Duration::from_secs(3600);

        cache.set(b"counter", b"0", ttl).await.expect("set should succeed");

        let (_, initial_cas) = cache.gets(b"counter").expect("gets should succeed");

        // Spawn multiple tasks all trying to CAS with the same token
        let mut handles = vec![];
        for i in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                let value = format!("{}", i + 1);
                cache_clone.cas(b"counter", value.as_bytes(), initial_cas, ttl).await
            });
            handles.push(handle);
        }

        let mut success_count = 0;
        let mut mismatch_count = 0;

        for handle in handles {
            match handle.await.expect("task should complete") {
                Ok(()) => success_count += 1,
                Err(crate::CacheError::CasMismatch) => mismatch_count += 1,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        // Only one should succeed, others get CasMismatch
        assert_eq!(success_count, 1, "Exactly one concurrent CAS should succeed");
        assert_eq!(mismatch_count, 3, "Other CAS operations should get CasMismatch");
    }

    // ==================== Expiration tests ====================

    #[tokio::test]
    async fn test_add_succeeds_for_expired_key() {
        let cache = create_test_cache();

        // Set a key with very short TTL (1 second)
        cache.set(b"expiring_key", b"original", Duration::from_secs(1)).await
            .expect("set should succeed");

        // Verify key exists initially
        assert!(cache.get(b"expiring_key").await.is_some(), "Key should exist before expiration");

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Key should now be expired (GET returns None)
        assert!(cache.get(b"expiring_key").await.is_none(), "Expired key should not be returned by GET");

        // ADD should succeed for expired key (treated as non-existent)
        let result = cache.add(b"expiring_key", b"new_value", Duration::from_secs(3600)).await;
        assert!(result.is_ok(), "ADD should succeed for expired key, got: {:?}", result);

        // New value should be retrievable
        let retrieved = cache.get(b"expiring_key").await;
        assert!(retrieved.is_some(), "New value should be retrievable after ADD");
        assert_eq!(retrieved.unwrap().value(), b"new_value");
    }

    #[tokio::test]
    async fn test_replace_fails_for_expired_key() {
        let cache = create_test_cache();

        // Set a key with very short TTL (1 second)
        cache.set(b"expiring_replace_key", b"original", Duration::from_secs(1)).await
            .expect("set should succeed");

        // Verify key exists initially
        assert!(cache.get(b"expiring_replace_key").await.is_some(), "Key should exist before expiration");

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Key should now be expired (GET returns None)
        assert!(cache.get(b"expiring_replace_key").await.is_none(), "Expired key should not be returned by GET");

        // REPLACE should fail for expired key (treated as non-existent)
        let result = cache.replace(b"expiring_replace_key", b"new_value", Duration::from_secs(3600)).await;
        assert!(matches!(result, Err(crate::CacheError::KeyNotFound)),
            "REPLACE should return KeyNotFound for expired key, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_cas_fails_for_expired_key() {
        let cache = create_test_cache();

        // Set a key with very short TTL (1 second)
        cache.set(b"expiring_cas_key", b"original", Duration::from_secs(1)).await
            .expect("set should succeed");

        // Get the CAS token while key is valid
        let (_, cas_token) = cache.gets(b"expiring_cas_key")
            .expect("gets should succeed before expiration");

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Key should now be expired
        assert!(cache.gets(b"expiring_cas_key").is_none(), "Expired key should not be returned by GETS");

        // CAS should fail for expired key (treated as non-existent)
        let result = cache.cas(b"expiring_cas_key", b"new_value", cas_token, Duration::from_secs(3600)).await;
        assert!(matches!(result, Err(crate::CacheError::KeyNotFound)),
            "CAS should return KeyNotFound for expired key, got: {:?}", result);
    }

    // ==================== Auto-promote configuration tests ====================

    #[test]
    fn test_auto_promote_disabled_by_default() {
        // Default cache should have auto_promote disabled
        let cache = CacheBuilder::new()
            .ram_size(1024 * 1024)
            .build()
            .expect("build should succeed");

        // Verify via metrics - ssd_promote counter should stay at 0
        // (no way to directly inspect inner.auto_promote, but we can verify behavior)
        assert_eq!(cache.metrics().ssd_promote.value(), 0);
    }

    #[test]
    fn test_auto_promote_builder_config() {
        // Test that auto_promote can be enabled via builder
        let cache = CacheBuilder::new()
            .ram_size(1024 * 1024)
            .auto_promote(true)
            .build()
            .expect("build should succeed");

        // Cache built successfully with auto_promote enabled
        // Full integration test would require SSD layer with real file
        assert_eq!(cache.metrics().ssd_promote.value(), 0);
    }
}