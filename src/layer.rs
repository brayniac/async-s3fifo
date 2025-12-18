//! Cache Layer - composes Pool + SmallQueue + TtlBuckets
//!
//! A cache layer represents a single storage tier (e.g., RAM or SSD) with:
//! - A segment pool for storage
//! - A small queue (admission queue) for filtering one-hit-wonders
//! - TTL buckets for the main cache
//!
//! # Data Flow
//!
//! ```text
//! New items → SmallQueue (FIFO)
//!                ↓ (on eviction)
//!           freq > 1? → Promote to TtlBuckets (main cache)
//!           freq <= 1? → Demote to next layer or drop
//! ```
//!
//! # Frequency Threshold
//!
//! Items are linked with initial frequency = 1. If an item is accessed again,
//! its frequency increases. Therefore:
//! - freq == 1: Never accessed after insertion (one-hit-wonder)
//! - freq > 1: Accessed at least once → "hot", promote to main cache
//!
//! # Multi-Layer Architecture
//!
//! Layers can be chained for tiered caching:
//! ```text
//! RAM Layer (fast, small)
//!     ↓ eviction
//! SSD Layer (slower, larger)
//!     ↓ eviction
//! Drop / Ghost entries
//! ```

use crate::hashtable::Hashtable;
use crate::item::SmallQueueItemHeader;
use crate::pool::Pool;
use crate::segment::Segment;
use crate::smallqueue::SmallQueue;
use crate::ttlbuckets::TtlBuckets;
use std::sync::atomic::{fence, Ordering};
use std::time::Duration;

/// Frequency threshold for promotion from small queue to main cache.
/// Items with frequency > this value are considered "hot" and promoted.
/// Items are inserted with freq=1, so freq>1 means accessed at least once.
pub const PROMOTION_THRESHOLD: u8 = 1;

/// Result of evicting a segment from the small queue
#[derive(Debug, Default)]
pub struct SmallQueueEvictResult {
    /// Number of items promoted to main cache (freq > threshold)
    pub items_promoted: u32,
    /// Number of items dropped (freq <= threshold, one-hit-wonders)
    pub items_dropped: u32,
    /// Number of items that were already deleted
    pub items_already_deleted: u32,
    /// Number of items that had expired TTL
    pub items_expired: u32,
    /// Bytes promoted to main cache
    pub bytes_promoted: u32,
    /// Bytes dropped
    pub bytes_dropped: u32,
}

/// A cache layer combining storage pool with admission and main cache queues
pub struct CacheLayer<P: Pool> {
    /// Storage pool (owns segment memory)
    pool: P,

    /// Pool ID for this layer (0-3)
    pool_id: u8,

    /// Small queue (admission filter) - FIFO, per-item TTL
    small_queue: SmallQueue,

    /// TTL buckets (main cache) - TTL-indexed, segment-level TTL
    ttl_buckets: TtlBuckets,
}

impl<P: Pool> CacheLayer<P> {
    /// Create a new cache layer with the given pool
    pub fn new(pool: P, pool_id: u8) -> Self {
        debug_assert!(pool_id <= 3, "pool_id must be 0-3");
        Self {
            pool,
            pool_id,
            small_queue: SmallQueue::new(),
            ttl_buckets: TtlBuckets::new(),
        }
    }

    /// Get a reference to the underlying pool
    pub fn pool(&self) -> &P {
        &self.pool
    }

    /// Get a reference to the TTL buckets
    pub fn ttl_buckets(&self) -> &TtlBuckets {
        &self.ttl_buckets
    }

    /// Append an item to the small queue (admission queue).
    ///
    /// New items enter the cache through the small queue. They must be accessed
    /// again (frequency > 1) to be promoted to the main cache.
    ///
    /// # Parameters
    /// - `key`: Item key
    /// - `value`: Item value
    /// - `optional`: Optional metadata
    /// - `expire_at`: Absolute expiration time (coarse seconds)
    /// - `hashtable`: Hashtable for linking the item
    /// - `metrics`: Cache metrics
    ///
    /// # Returns
    /// - `Ok((segment_id, offset))` on success
    /// - `Err(())` if insertion failed (no segments available, etc.)
    pub async fn append_to_small_queue(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        hashtable: &Hashtable,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(u32, u32), ()> {
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > 16 {
                return Err(()); // Too many retries
            }

            // Try to append to current tail segment
            if let Some(tail_id) = self.small_queue.tail()
                && let Some(segment) = self.pool.get(tail_id)
                && segment.state() == crate::segment::State::Live
                && let Some(offset) = segment.append_small_queue_item(
                    key, value, optional, expire_at, metrics
                )
            {
                // Successfully appended, now link in hashtable
                match hashtable.link_item(
                    key,
                    self.pool_id,
                    tail_id,
                    offset,
                    &self.pool,
                    metrics,
                ) {
                    Ok(_) => return Ok((tail_id, offset)),
                    Err(()) => {
                        // Link failed (hashtable full or collision)
                        // Item is written but not accessible - wasted space but safe
                        return Err(());
                    }
                }
            }
            // Segment full or not available, fall through to allocate new segment

            // Need a new segment - try to reserve one
            let new_segment_id = match self.reserve_small_queue_segment(metrics) {
                Some(id) => id,
                None => {
                    // No small queue segments available
                    // Could try to evict here, but for now just fail
                    return Err(());
                }
            };

            // Append the new segment to the small queue (async with mutex)
            match self.small_queue.append_segment(new_segment_id, &self.pool, metrics).await {
                Ok(()) => {
                    // Successfully added segment, loop back to try appending item
                    continue;
                }
                Err(()) => {
                    // Failed to append segment
                    // Release it and retry
                    self.pool.release(new_segment_id, metrics);
                    continue;
                }
            }
        }
    }

    /// Append an item to the small queue only if the key does not already exist.
    ///
    /// This implements atomic ADD semantics for the small queue. The key existence
    /// check and insertion are atomic - if two concurrent ADDs target the same key,
    /// exactly one will succeed.
    ///
    /// # Returns
    /// - `Ok(())` on success (key was not present)
    /// - `Err(KeyExists)` if the key already exists
    /// - `Err(StorageFull)` if no segments available
    /// - `Err(HashTableFull)` if hashtable buckets are full
    pub async fn append_to_small_queue_if_absent(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        hashtable: &Hashtable,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), crate::CacheError> {
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > 16 {
                return Err(crate::CacheError::StorageFull);
            }

            // Try to append to current tail segment
            if let Some(tail_id) = self.small_queue.tail()
                && let Some(segment) = self.pool.get(tail_id)
                && segment.state() == crate::segment::State::Live
                && let Some(offset) = segment.append_small_queue_item(
                    key, value, optional, expire_at, metrics
                )
            {
                // Successfully appended, now link atomically (ADD semantics)
                match hashtable.link_item_if_absent(
                    key,
                    self.pool_id,
                    tail_id,
                    offset,
                    &self.pool,
                    metrics,
                ) {
                    Ok(()) => return Ok(()),
                    Err(crate::CacheError::KeyExists) => {
                        // Key already exists - return immediately
                        // Note: we've written orphaned data, but it will be cleaned up
                        return Err(crate::CacheError::KeyExists);
                    }
                    Err(e) => return Err(e),
                }
            }
            // Segment full or not available, fall through to allocate new segment

            // Need a new segment - try to reserve one
            let new_segment_id = match self.reserve_small_queue_segment(metrics) {
                Some(id) => id,
                None => return Err(crate::CacheError::StorageFull),
            };

            // Append the new segment to the small queue (async with mutex)
            match self.small_queue.append_segment(new_segment_id, &self.pool, metrics).await {
                Ok(()) => continue,
                Err(()) => {
                    self.pool.release(new_segment_id, metrics);
                    continue;
                }
            }
        }
    }

    /// Reserve a segment from the pool for the small queue
    ///
    /// This reserves a segment marked as `is_small_queue = true`.
    /// If no small queue segments are available, returns None.
    pub fn reserve_small_queue_segment(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        self.pool.reserve_small_queue(metrics)
    }

    /// Append an item to the small queue only if the key already exists (REPLACE semantics).
    ///
    /// This is used when replacing an item that has low frequency - it goes back through
    /// the small queue admission filter rather than directly to main cache.
    ///
    /// # Returns
    /// - `Ok((segment_id, offset))` on success (key existed and was replaced)
    /// - `Err(KeyNotFound)` if the key does not exist
    /// - `Err(StorageFull)` if no segments available
    pub async fn append_to_small_queue_if_present(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        hashtable: &Hashtable,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(u32, u32), crate::CacheError> {
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > 16 {
                return Err(crate::CacheError::StorageFull);
            }

            // Try to append to current tail segment
            if let Some(tail_id) = self.small_queue.tail()
                && let Some(segment) = self.pool.get(tail_id)
                && segment.state() == crate::segment::State::Live
                && let Some(offset) = segment.append_small_queue_item(
                    key, value, optional, expire_at, metrics
                )
            {
                // Successfully appended, now link atomically (REPLACE semantics)
                match hashtable.link_item_if_present(
                    key,
                    self.pool_id,
                    tail_id,
                    offset,
                    &self.pool,
                    metrics,
                ) {
                    Ok(_old_location) => return Ok((tail_id, offset)),
                    Err(crate::CacheError::KeyNotFound) => {
                        // Key doesn't exist - return immediately
                        // Note: we've written orphaned data, but it will be cleaned up
                        return Err(crate::CacheError::KeyNotFound);
                    }
                    Err(e) => return Err(e),
                }
            }
            // Segment full or not available, fall through to allocate new segment

            // Need a new segment - try to reserve one
            let new_segment_id = match self.reserve_small_queue_segment(metrics) {
                Some(id) => id,
                None => return Err(crate::CacheError::StorageFull),
            };

            // Append the new segment to the small queue (async with mutex)
            match self.small_queue.append_segment(new_segment_id, &self.pool, metrics).await {
                Ok(()) => continue,
                Err(()) => {
                    self.pool.release(new_segment_id, metrics);
                    continue;
                }
            }
        }
    }

    /// Evict the oldest segment from the small queue.
    ///
    /// This implements the S3-FIFO admission policy:
    /// 1. Remove the head (oldest) segment from the small queue
    /// 2. For each item in the segment:
    ///    - If freq > 1 (accessed since insertion): promote to main cache
    ///    - If freq <= 1 (one-hit-wonder): demote to SSD layer or drop
    /// 3. Release the segment back to the pool
    ///
    /// # Parameters
    /// - `cache`: CacheOps implementation for accessing pool, hashtable, and metrics
    /// - `current_time`: Current coarse time for TTL checking (seconds)
    /// - `demote_layer`: Optional SSD layer to demote cold items to. If None, cold items are dropped.
    ///
    /// # Returns
    /// - `Some(SmallQueueEvictResult)` with eviction statistics
    /// - `None` if the small queue is empty
    pub async fn evict_small_queue_segment<P2: Pool>(
        &self,
        cache: &impl crate::cache::CacheOps<P>,
        current_time: u32,
        demote_layer: Option<&CacheLayer<P2>>,
    ) -> Option<SmallQueueEvictResult>
    {
        let hashtable = cache.hashtable();
        let metrics = cache.metrics();
        // Evict the head segment from the small queue (async with mutex)
        let segment_id = self.small_queue.evict_head(&self.pool, metrics).await?;

        let segment = self.pool.get(segment_id)?;
        let mut result = SmallQueueEvictResult::default();

        // Get the write offset to know how much data to scan
        let write_offset = segment.offset();

        // Synchronize with append operations
        fence(Ordering::Acquire);

        let mut current_offset = 0u32;

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + SmallQueueItemHeader::SIZE as u32 > segment.data_len() as u32 {
                break;
            }

            // Read the item header
            // SAFETY: We've validated the offset is within bounds
            let header = match self.read_small_queue_header(segment_id, current_offset) {
                Some(h) => h,
                None => break,
            };

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip already deleted items
            if header.is_deleted() {
                result.items_already_deleted += 1;
                current_offset += item_size;
                continue;
            }

            // Get item data slice once for this item (zero allocation)
            let item_data = match segment.data_slice(current_offset, header.padded_size()) {
                Some(data) => data,
                None => {
                    current_offset += item_size;
                    continue;
                }
            };

            // Check if item has expired
            if header.is_expired(current_time) {
                result.items_expired += 1;
                result.bytes_dropped += item_size;
                metrics.small_queue_expire.increment();
                // Unlink from hashtable - only need the key (zero allocation)
                if let Some(key) = Self::read_small_queue_key_slice(item_data, &header) {
                    hashtable.unlink_item(key, segment_id, current_offset, metrics);
                }
                current_offset += item_size;
                continue;
            }

            // Extract item data as slices (zero allocation)
            let (key, value, optional) = match Self::read_small_queue_item_slices(item_data, &header) {
                Some(data) => data,
                None => {
                    current_offset += item_size;
                    continue;
                }
            };

            // Look up item frequency from hashtable
            // Returns None if item is orphaned (key exists but at different location)
            let freq = match hashtable.get_item_frequency(key, segment_id, current_offset) {
                Some(f) => f,
                None => {
                    // Item is orphaned (was overwritten with new version at different location)
                    // Skip silently - don't count as drop, don't create ghost
                    // The current version will be handled when its segment is evicted
                    current_offset += item_size;
                    continue;
                }
            };

            if freq > PROMOTION_THRESHOLD {
                // Hot item: promote to main cache (TTL buckets)
                // Calculate remaining TTL
                let ttl_remaining = header.expire_at().saturating_sub(current_time);

                // Skip items with no remaining TTL
                if ttl_remaining == 0 {
                    result.items_expired += 1;
                    result.bytes_dropped += item_size;
                    hashtable.unlink_item(key, segment_id, current_offset, metrics);
                    current_offset += item_size;
                    continue;
                }

                // Try to promote to main cache
                match self.promote_to_main_cache(
                    cache,
                    key,
                    value,
                    optional,
                    ttl_remaining,
                    segment_id,
                    current_offset,
                    hashtable,
                ).await {
                    Ok(_) => {
                        result.items_promoted += 1;
                        result.bytes_promoted += item_size;
                        metrics.small_queue_promote.increment();
                    }
                    Err(()) => {
                        // Promotion failed - either:
                        // 1. append_item failed (data not written)
                        // 2. relink_item failed (data written, hashtable already changed)
                        // In case 2, we must not touch the hashtable.
                        // We can't distinguish the cases, so don't modify hashtable at all.
                        result.items_dropped += 1;
                        result.bytes_dropped += item_size;
                        metrics.small_queue_promote_fail.increment();
                    }
                }
            } else {
                // Cold item: one-hit-wonder, demote or drop
                let ttl_remaining = header.expire_at().saturating_sub(current_time);

                if let Some(ssd_layer) = demote_layer {
                    if ttl_remaining > 0 {
                        let expire_at = current_time + ttl_remaining;
                        // Demote to SSD small queue (best effort)
                        let _ = ssd_layer.append_to_small_queue(
                            key,
                            value,
                            optional,
                            expire_at,
                            hashtable,
                            metrics,
                        ).await;
                    }
                }

                result.items_dropped += 1;
                result.bytes_dropped += item_size;
                metrics.small_queue_drop.increment();

                // Create ghost entry - helps detect if "cold" items are actually accessed repeatedly
                hashtable.unlink_item_to_ghost(key, segment_id, current_offset, metrics);
            }

            current_offset += item_size;
        }

        // Release the segment back to the pool
        self.pool.release(segment_id, metrics);

        Some(result)
    }

    /// Read a SmallQueueItemHeader from a segment at the given offset.
    fn read_small_queue_header(&self, segment_id: u32, offset: u32) -> Option<SmallQueueItemHeader> {
        let segment = self.pool.get(segment_id)?;

        // Read header bytes from segment
        let header_bytes = segment.data_slice(offset, SmallQueueItemHeader::SIZE)?;

        // Parse the header
        SmallQueueItemHeader::try_from_bytes(header_bytes)
    }

    /// Read item data (key, value, optional) as slices from segment data.
    ///
    /// This is the zero-allocation version that returns references directly into
    /// segment memory. Use this in hot paths like eviction.
    fn read_small_queue_item_slices<'a>(
        item_data: &'a [u8],
        header: &SmallQueueItemHeader,
    ) -> Option<(&'a [u8], &'a [u8], &'a [u8])> {
        // Calculate ranges within the item
        let optional_start = SmallQueueItemHeader::SIZE;
        let optional_end = optional_start + header.optional_len() as usize;
        let key_start = optional_end;
        let key_end = key_start + header.key_len() as usize;
        let value_start = key_end;
        let value_end = value_start + header.value_len() as usize;

        // Extract each component as slices (no allocation)
        let optional = item_data.get(optional_start..optional_end)?;
        let key = item_data.get(key_start..key_end)?;
        let value = item_data.get(value_start..value_end)?;

        Some((key, value, optional))
    }

    /// Read just the key as a slice from segment data.
    ///
    /// Use this when only the key is needed (e.g., for unlink operations).
    fn read_small_queue_key_slice<'a>(
        item_data: &'a [u8],
        header: &SmallQueueItemHeader,
    ) -> Option<&'a [u8]> {
        let optional_end = SmallQueueItemHeader::SIZE + header.optional_len() as usize;
        let key_start = optional_end;
        let key_end = key_start + header.key_len() as usize;
        item_data.get(key_start..key_end)
    }

    /// Try to promote an item from small queue to main cache.
    ///
    /// This appends the item to the appropriate TTL bucket based on remaining TTL,
    /// then relinks the hashtable entry to point to the new location.
    ///
    /// # Parameters
    /// - `key`: Item key
    /// - `value`: Item value
    /// - `optional`: Optional metadata
    /// - `ttl_remaining_secs`: Remaining TTL in seconds
    /// - `old_segment_id`: Current segment ID in small queue
    /// - `old_offset`: Current offset in small queue segment
    /// - `hashtable`: Hashtable for relinking
    /// - `metrics`: Cache metrics
    ///
    /// # Returns
    /// - `Ok((new_segment_id, new_offset))` on success
    /// - `Err(())` if promotion failed (segment full, no segments available, etc.)
    #[allow(clippy::too_many_arguments)]
    async fn promote_to_main_cache(
        &self,
        cache: &impl crate::cache::CacheOps<P>,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        ttl_remaining_secs: u32,
        old_segment_id: u32,
        old_offset: u32,
        hashtable: &Hashtable,
    ) -> Result<(u32, u32), ()> {
        // Get the appropriate TTL bucket based on remaining TTL
        let ttl = Duration::from_secs(ttl_remaining_secs.into());

        // Append to TTL bucket - this handles segment allocation and retries
        let (new_segment_id, new_offset) = self.ttl_buckets
            .append_item(cache, key, value, optional, ttl)
            .await
            .ok_or(())?;

        // Relink the hashtable entry to point to the new location
        // Note: We reset frequency to 1 since this is a fresh entry in main cache
        // The item has proven itself by being accessed in the small queue
        let old_pool_id = self.pool_id;
        let new_pool_id = self.pool_id;

        if hashtable.relink_item(
            key,
            old_pool_id,
            old_segment_id,
            old_offset,
            new_pool_id,
            new_segment_id,
            new_offset,
            false, // Don't preserve freq, reset to 1
        ) {
            Ok((new_segment_id, new_offset))
        } else {
            // Relink failed - item may have been deleted or modified
            // The data is already written to the segment but won't be accessible
            // This is a waste of space but safe
            Err(())
        }
    }
}

// Implement Pool trait delegation so CacheLayer can be used where Pool is expected
impl<P: Pool> Pool for CacheLayer<P> {
    type Segment = P::Segment;

    fn get(&self, id: u32) -> Option<&Self::Segment> {
        self.pool.get(id)
    }

    fn segment_count(&self) -> usize {
        self.pool.segment_count()
    }

    fn reserve_small_queue(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        self.pool.reserve_small_queue(metrics)
    }

    fn reserve_main_cache(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        self.pool.reserve_main_cache(metrics)
    }

    fn release(&self, id: u32, metrics: &crate::metrics::CacheMetrics) {
        self.pool.release(id, metrics)
    }
}
