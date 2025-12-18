use crate::segment::Segment;
use crate::pool::Pool;
use crate::cache::CacheOps;
use crate::layer::CacheLayer;
use crate::segment::INVALID_SEGMENT_ID;
use crate::segment::State;
use crate::sync::*;
use std::time::Duration;

/// Error type for try_append_segment operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendSegmentError {
    InvalidSegmentId,
    SegmentNotReserved,
    FailedToTransitionToLinking,
    BucketModified,
    InvalidTailSegmentId,
    TailInUnexpectedState,
}

/// Error type for try_append_item operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppendItemError {
    NoTailSegment,
    InvalidTailSegmentId,
    TailNotLive,
    TailSegmentFull,
}

/// Error type for try_remove_segment operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RemoveSegmentError {
    InvalidSegmentId,
    FailedToTransitionToDraining,
    #[allow(dead_code)]
    SegmentHasActiveReaders,
    FailedToTransitionToLocked,
    InvalidPrevSegmentId,
    InvalidNextSegmentId,
    FailedToLockPrevSegment,
    FailedToLockNextSegment,
    BucketModified,
}

// TTL bucket configuration constants
const N_BUCKET_PER_STEP_N_BIT: usize = 8;
const N_BUCKET_PER_STEP: usize = 1 << N_BUCKET_PER_STEP_N_BIT; // 256 buckets per step

const TTL_BUCKET_INTERVAL_N_BIT_1: usize = 3; // 8 second intervals
const TTL_BUCKET_INTERVAL_N_BIT_2: usize = 7; // 128 second intervals
const TTL_BUCKET_INTERVAL_N_BIT_3: usize = 11; // 2048 second intervals
const TTL_BUCKET_INTERVAL_N_BIT_4: usize = 15; // 32768 second intervals

const TTL_BUCKET_INTERVAL_1: usize = 1 << TTL_BUCKET_INTERVAL_N_BIT_1; // 8
const TTL_BUCKET_INTERVAL_2: usize = 1 << TTL_BUCKET_INTERVAL_N_BIT_2; // 128
const TTL_BUCKET_INTERVAL_3: usize = 1 << TTL_BUCKET_INTERVAL_N_BIT_3; // 2048
const TTL_BUCKET_INTERVAL_4: usize = 1 << TTL_BUCKET_INTERVAL_N_BIT_4; // 32768

const TTL_BOUNDARY_1: i32 = 1 << (TTL_BUCKET_INTERVAL_N_BIT_1 + N_BUCKET_PER_STEP_N_BIT); // 2048
const TTL_BOUNDARY_2: i32 = 1 << (TTL_BUCKET_INTERVAL_N_BIT_2 + N_BUCKET_PER_STEP_N_BIT); // 32768
const TTL_BOUNDARY_3: i32 = 1 << (TTL_BUCKET_INTERVAL_N_BIT_3 + N_BUCKET_PER_STEP_N_BIT); // 524288

const MAX_N_TTL_BUCKET: usize = N_BUCKET_PER_STEP * 4; // 1024 total buckets
const MAX_TTL_BUCKET_IDX: usize = MAX_N_TTL_BUCKET - 1;

/// TTL buckets for organizing segments by expiration time
pub struct TtlBuckets {
    buckets: Box<[TtlBucket]>,
    #[allow(dead_code)]
    last_expired: clocksource::coarse::AtomicInstant,
}

impl TtlBuckets {
    /// Create a new set of `TtlBuckets` which cover the full range of TTLs.
    /// Uses logarithmic bucketing for efficient coverage of wide TTL ranges.
    pub fn new() -> Self {
        let intervals = [
            TTL_BUCKET_INTERVAL_1,
            TTL_BUCKET_INTERVAL_2,
            TTL_BUCKET_INTERVAL_3,
            TTL_BUCKET_INTERVAL_4,
        ];

        let mut buckets = Vec::with_capacity(intervals.len() * N_BUCKET_PER_STEP);

        for (i, interval) in intervals.iter().enumerate() {
            for j in 0..N_BUCKET_PER_STEP {
                // Buckets use the minimum TTL of their range to ensure we expire early, not late
                // Shift all ranges by 1 to avoid TTL=0:
                // Bucket 0 (range 1-8s): TTL=1s
                // Bucket 1 (range 9-16s): TTL=9s
                // Bucket 2 (range 17-24s): TTL=17s
                let ttl_secs = (interval * j + 1) as u64;
                let ttl = Duration::from_secs(ttl_secs);
                let index = (i * N_BUCKET_PER_STEP + j) as u16;
                let bucket = TtlBucket::new(ttl, index);
                buckets.push(bucket);
            }
        }

        let buckets = buckets.into_boxed_slice();
        let last_expired = clocksource::coarse::AtomicInstant::now();

        Self {
            buckets,
            last_expired,
        }
    }

    /// Get the index of the `TtlBucket` for the given TTL.
    /// Uses branchless operations for better performance.
    #[allow(clippy::manual_range_contains)]
    fn get_bucket_index(&self, ttl: Duration) -> usize {
        let ttl_secs = ttl.as_secs() as i32;

        // Handle negative/zero TTL - this is a bug condition
        if ttl_secs <= 0 {
            panic!("TTL must be positive, got {} seconds", ttl_secs);
        }

        // Branchless bucket index calculation using bit manipulation
        // We compute all possible indices and select the correct one
        let idx1 = (ttl_secs >> TTL_BUCKET_INTERVAL_N_BIT_1) as usize;
        let idx2 = (ttl_secs >> TTL_BUCKET_INTERVAL_N_BIT_2) as usize + N_BUCKET_PER_STEP;
        let idx3 = (ttl_secs >> TTL_BUCKET_INTERVAL_N_BIT_3) as usize + N_BUCKET_PER_STEP * 2;
        let idx4 = (ttl_secs >> TTL_BUCKET_INTERVAL_N_BIT_4) as usize + N_BUCKET_PER_STEP * 3;

        // Create masks for each range (1 if in range, 0 otherwise)
        let mask1 = ((ttl_secs < TTL_BOUNDARY_1) as usize).wrapping_neg();
        let mask2 =
            (((ttl_secs >= TTL_BOUNDARY_1) & (ttl_secs < TTL_BOUNDARY_2)) as usize).wrapping_neg();
        let mask3 =
            (((ttl_secs >= TTL_BOUNDARY_2) & (ttl_secs < TTL_BOUNDARY_3)) as usize).wrapping_neg();
        let mask4 = ((ttl_secs >= TTL_BOUNDARY_3) as usize).wrapping_neg();

        // Select the appropriate index using bitwise operations
        let bucket_idx = (idx1 & mask1) | (idx2 & mask2) | (idx3 & mask3) | (idx4 & mask4);

        // Clamp to maximum bucket index
        std::cmp::min(bucket_idx, MAX_TTL_BUCKET_IDX)
    }

    /// Try to expire a segment from any TTL bucket.
    ///
    /// Scans buckets starting from low TTL (most likely to have expired segments)
    /// and attempts to expire the head segment. Returns immediately when one is found.
    ///
    /// # Returns
    /// - `Some(segment_id)`: A segment was expired and is ready for reuse (in Reserved state)
    /// - `None`: No expired segments found
    ///
    /// # Note
    /// This is a best-effort scan - it stops at the first successfully expired segment.
    /// Multiple calls may be needed to expire all available expired segments.
    pub fn try_expire<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<u32> {
        let now = clocksource::coarse::Instant::now();

        // Scan buckets from low TTL to high (low TTL buckets expire sooner)
        for bucket in self.buckets.iter() {
            // Quick check: skip empty buckets
            if bucket.head().is_none() {
                continue;
            }

            // Try to expire the head segment of this bucket
            if let Ok(segment_id) = bucket.try_expire_head_segment(cache, metrics, now) {
                return Some(segment_id);
            }
            // If expiration failed (not expired, readers active, etc.), try next bucket
        }

        None
    }

    /// Append an item to the appropriate TTL bucket based on duration
    ///
    /// # Loom Test Coverage
    /// - `concurrent_insert_same_segment` (ignored) - Full end-to-end test through this path
    pub async fn append_item<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        duration: Duration,
    ) -> Option<(u32, u32)> {
        let target_idx = self.get_bucket_index(duration);
        let bucket = &self.buckets[target_idx];

        // Try the correct TTL bucket first
        if let Some(result) = bucket.append_item(cache, key, value, optional, cache.metrics()).await {
            return Some(result);
        }

        // Fallback: try shorter-TTL buckets (lower indices = earlier expiration)
        // This is safe because TTL is a maximum lifetime guarantee, not exact.
        // Under memory pressure, items may expire earlier than requested.
        for idx in (0..target_idx).rev() {
            let fallback_bucket = &self.buckets[idx];
            if let Some(result) = fallback_bucket.append_item(cache, key, value, optional, cache.metrics()).await {
                cache.metrics().ttl_bucket_borrow.increment();
                return Some(result);
            }
        }

        None
    }

    /// Evict a single segment from any TTL bucket
    ///
    /// Evicts the head segment from the bucket whose head is closest to expiry.
    /// This prioritizes evicting items that are about to expire anyway, preserving
    /// items with more remaining useful life.
    ///
    /// Requires at least 2 segments in the bucket (need to keep tail live).
    ///
    /// # Returns
    /// - `Some(segment_id)`: A segment was evicted and is ready for reuse
    /// - `None`: No bucket had enough segments to evict
    pub fn evict_any<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<u32> {
        // Find the bucket with the earliest expiring head segment
        let mut earliest_bucket_idx: Option<usize> = None;
        let mut earliest_expire_at: Option<clocksource::coarse::Instant> = None;

        for (idx, bucket) in self.buckets.iter().enumerate() {
            // Need at least 2 segments to evict (can't evict from single-segment chain)
            if bucket.chain_len(cache, 2) < 2 {
                continue;
            }

            if let Some(expire_at) = bucket.head_expire_at(cache) {
                let is_earlier = match earliest_expire_at {
                    None => true,
                    Some(current_earliest) => expire_at < current_earliest,
                };
                if is_earlier {
                    earliest_bucket_idx = Some(idx);
                    earliest_expire_at = Some(expire_at);
                }
            }
        }

        // Evict from the bucket with earliest expiring head
        if let Some(idx) = earliest_bucket_idx
            && let Some(segment_id) = self.buckets[idx].evict_head_segment(cache, metrics)
        {
            return Some(segment_id);
        }

        // Fallback: try any bucket with enough segments (in case of race)
        for bucket in self.buckets.iter() {
            if bucket.chain_len(cache, 2) >= 2
                && let Some(segment_id) = bucket.evict_head_segment(cache, metrics)
            {
                return Some(segment_id);
            }
        }

        None
    }

    /// Run merge eviction on the bucket whose head is closest to expiry.
    ///
    /// This prioritizes evicting items that are about to expire anyway, preserving
    /// items with more remaining useful life. Only buckets with enough segments
    /// for merge are considered.
    ///
    /// # Parameters
    /// - `cache`: CacheOps implementation (provides hashtable for demotion)
    /// - `merge_ratio`: Number of segments to merge
    /// - `target_ratio`: Target segments after merge
    /// - `metrics`: Cache metrics
    /// - `demote_layer`: Optional layer to demote pruned items to
    ///
    /// # Returns
    /// - `Some(freed_count)`: Number of segments freed
    /// - `None`: No bucket had enough segments to merge
    pub async fn merge_evict_any<P: Pool, P2: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        merge_ratio: usize,
        target_ratio: usize,
        metrics: &crate::metrics::CacheMetrics,
        demote_layer: Option<&CacheLayer<P2>>,
    ) -> Option<usize> {
        // Minimum segments needed for merge (destination + sources + keep tail live)
        let min_segments = merge_ratio + 1;

        // Find the bucket with the earliest expiring head segment (among those with enough segments)
        let mut earliest_bucket_idx: Option<usize> = None;
        let mut earliest_expire_at: Option<clocksource::coarse::Instant> = None;
        let mut candidates: Vec<usize> = Vec::new();

        for (idx, bucket) in self.buckets.iter().enumerate() {
            let count = bucket.chain_len(cache, min_segments + 1);
            if count >= min_segments {
                candidates.push(idx);

                if let Some(expire_at) = bucket.head_expire_at(cache) {
                    let is_earlier = match earliest_expire_at {
                        None => true,
                        Some(current_earliest) => expire_at < current_earliest,
                    };
                    if is_earlier {
                        earliest_bucket_idx = Some(idx);
                        earliest_expire_at = Some(expire_at);
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Try the bucket with earliest expiring head first
        if let Some(selected_idx) = earliest_bucket_idx {
            if let Some(freed) = self.buckets[selected_idx]
                .merge_evict(cache, merge_ratio, target_ratio, metrics, demote_layer)
                .await
            {
                return Some(freed);
            }

            // If selected bucket failed, try others in order of earliest expiry
            for idx in candidates {
                if idx == selected_idx {
                    continue; // Already tried
                }
                if let Some(freed) = self.buckets[idx]
                    .merge_evict(cache, merge_ratio, target_ratio, metrics, demote_layer)
                    .await
                {
                    return Some(freed);
                }
            }
        }

        None
    }
}

/// A single TTL bucket containing a doubly-linked list of segments
pub struct TtlBucket {
    // Use AtomicU64 to pack head and tail for atomic updates
    // Layout: [16 bits unused][24 bits tail][24 bits head]
    head_tail: AtomicU64,

    // Async mutex for serializing segment append/remove operations
    // This eliminates CAS retry loops and allows other tasks to run while waiting
    write_lock: tokio::sync::Mutex<()>,

    ttl: Duration,
    index: u16, // Bucket index in the TtlBuckets array
}

impl TtlBucket {
    const INVALID_ID: u32 = 0xFFFFFF;

    pub fn new(ttl: Duration, index: u16) -> Self {
        // Initially empty - both head and tail are invalid
        let initial = Self::pack_head_tail(Self::INVALID_ID, Self::INVALID_ID);
        Self {
            head_tail: AtomicU64::new(initial),
            write_lock: tokio::sync::Mutex::new(()),
            ttl,
            index,
        }
    }

    fn pack_head_tail(head: u32, tail: u32) -> u64 {
        let head_24 = (head & 0xFFFFFF) as u64;
        let tail_24 = (tail & 0xFFFFFF) as u64;
        (tail_24 << 24) | head_24
    }

    fn unpack_head_tail(packed: u64) -> (u32, u32) {
        let head = (packed & 0xFFFFFF) as u32;
        let tail = ((packed >> 24) & 0xFFFFFF) as u32;
        (head, tail)
    }

    /// Get the head segment ID of this bucket
    pub fn head(&self) -> Option<u32> {
        let packed = self.head_tail.load(Ordering::Acquire);
        let (head, _) = Self::unpack_head_tail(packed);
        if head == Self::INVALID_ID {
            None
        } else {
            Some(head)
        }
    }

    /// Get the expiration time of the head segment
    ///
    /// Returns the `expire_at` instant of the head segment, or None if bucket is empty.
    pub fn head_expire_at<P: Pool>(&self, cache: &impl CacheOps<P>) -> Option<clocksource::coarse::Instant> {
        let head_id = self.head()?;
        let segment = cache.pool().get(head_id)?;
        Some(segment.expire_at())
    }

    /// Test-only helper to set bucket head and tail pointers directly
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn test_set_head_tail(&self, head: u32, tail: u32) {
        let packed = Self::pack_head_tail(head, tail);
        self.head_tail.store(packed, Ordering::Release);
    }

    /// Try to append a segment to the end of this TTL bucket chain (single attempt, no retry)
    ///
    /// This transitions the segment from Reserved to Live state and adds it to the bucket's
    /// linked list. If the bucket is empty, the segment becomes both head and tail.
    ///
    /// Returns Ok(()) on success, Err with reason on failure.
    ///
    /// # Loom Test Coverage
    /// - `try_append_segment` - Low-level single-attempt append without retry loops
    pub(crate) fn try_append_segment<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        segment_id: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), AppendSegmentError> {
        let segment = cache
            .pool()
            .get(segment_id)
            .ok_or(AppendSegmentError::InvalidSegmentId)?;

        // Verify segment is in Reserved state
        if segment.state() != State::Reserved {
            return Err(AppendSegmentError::SegmentNotReserved);
        }

        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (current_head, current_tail) = Self::unpack_head_tail(current_packed);

        if current_head == Self::INVALID_ID {
            // List is empty - try to make this segment both head and tail
            let new_packed = Self::pack_head_tail(segment_id, segment_id);

            // First, transition segment to Linking state
            if !segment.cas_metadata(
                State::Reserved,
                State::Linking,
                None, // next stays invalid
                None, // prev stays invalid
                metrics,
            ) {
                return Err(AppendSegmentError::FailedToTransitionToLinking);
            }

            // Set bucket ID now that we own the segment in Linking state
            segment.set_bucket_id(self.index);

            // Try to update head and tail atomically
            match self.head_tail.compare_exchange(
                current_packed,
                new_packed,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Successfully updated bucket, now transition segment to Live
                    // We own this segment in Linking state, this CAS must succeed
                    assert!(
                        segment.cas_metadata(
                            State::Linking,
                            State::Live,
                            None, // next stays invalid
                            None, // prev stays invalid
                            metrics,
                        ),
                        "Failed to transition owned segment to Live in empty bucket - data structure corruption detected"
                    );
                    metrics.ttl_append_segment.increment();
                    Ok(())
                }
                Err(_) => {
                    // Someone else modified the bucket
                    // Clear bucket ID and revert from Linking to Reserved first
                    segment.clear_bucket_id();
                    segment.cas_metadata(
                        State::Linking,
                        State::Reserved,
                        None,
                        None,
                        metrics,
                    );
                    // Now return the segment to the free pool
                    cache.pool().release(segment_id, metrics);
                    metrics.ttl_append_segment_error.increment();
                    Err(AppendSegmentError::BucketModified)
                }
            }
        } else {
            // List is not empty - append to tail
            let tail_segment = cache
                .pool()
                .get(current_tail)
                .ok_or(AppendSegmentError::InvalidTailSegmentId)?;

            // First, transition to Linking state and set prev pointer
            if !segment.cas_metadata(
                State::Reserved,
                State::Linking,
                None,               // next stays invalid (it's the new tail)
                Some(current_tail), // prev points to old tail
                metrics,
            ) {
                return Err(AppendSegmentError::FailedToTransitionToLinking);
            }

            // Set bucket ID now that we own the segment in Linking state
            segment.set_bucket_id(self.index);

            // Seal the old tail and update it to point forward to new segment
            // Try Live first (common case), then Sealed (if already sealed)
            // Pass None for prev to preserve current value (race-free)
            let sealed = tail_segment.cas_metadata(
                State::Live,
                State::Sealed,
                Some(segment_id), // next now points to new segment
                None,             // Preserve current prev
                metrics,
            ) || tail_segment.cas_metadata(
                State::Sealed,
                State::Sealed,
                Some(segment_id), // next now points to new segment
                None,             // Preserve current prev
                metrics,
            );

            if !sealed {
                // Tail is in unexpected state, abort
                segment.clear_bucket_id();
                segment.cas_metadata(State::Linking, State::Reserved, None, None, metrics);
                metrics.ttl_append_segment_error.increment();
                return Err(AppendSegmentError::TailInUnexpectedState);
            }

            // Finally, update the bucket's tail pointer
            let new_packed = Self::pack_head_tail(current_head, segment_id);
            match self.head_tail.compare_exchange(
                current_packed,
                new_packed,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Successfully updated bucket tail, now transition to Live
                    // We own this segment in Linking state, this CAS must succeed
                    assert!(
                        segment.cas_metadata(
                            State::Linking,
                            State::Live,
                            None,               // next stays invalid
                            Some(current_tail), // prev stays set
                            metrics,
                        ),
                        "Failed to transition owned segment to Live after linking - data structure corruption detected"
                    );
                    metrics.ttl_append_segment.increment();
                    Ok(())
                }
                Err(_) => {
                    // Someone else modified the bucket - they won the race
                    // We speculatively updated the old tail's next pointer, but another thread
                    // may have also done so and won the CAS. We can't reliably unlink because
                    // the old tail's next pointer may have been overwritten multiple times.

                    // Try to unlink: restore old tail's next to invalid, but don't panic on failure
                    // Failure just means another thread won and has properly linked the chain
                    let _unlink_succeeded = tail_segment.cas_metadata(
                        State::Sealed,
                        State::Sealed,
                        Some(INVALID_SEGMENT_ID), // Unlink from new segment
                        None,                      // Preserve current prev
                        metrics,
                    );

                    // If unlink failed, the segment we tried to link is orphaned but that's OK -
                    // it will be cleaned up when the segment pool runs low or during eviction

                    // Clear bucket ID and revert new segment from Linking back to Reserved
                    segment.clear_bucket_id();
                    segment.cas_metadata(
                        State::Linking,
                        State::Reserved,
                        None,
                        None,
                        metrics,
                    );

                    // Return segment to free pool
                    cache.pool().release(segment_id, metrics);
                    metrics.ttl_append_segment_error.increment();
                    Err(AppendSegmentError::BucketModified)
                }
            }
        }
    }

    /// Append a segment to the end of this TTL bucket chain
    /// The segment must be in Reserved state
    ///
    /// # Loom Test Coverage
    /// - `concurrent_ttl_bucket_append` - Two threads appending different segments to same bucket
    /// - `concurrent_insert_same_segment` (ignored) - Full end-to-end insertion path
    pub async fn append_segment<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        segment_id: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), &'static str> {
        let segment = cache
            .pool()
            .get(segment_id)
            .ok_or("Invalid segment ID")?;

        // Set the expiration time for this segment (only once)
        let expire_at = clocksource::coarse::Instant::now() + self.ttl;
        segment.set_expire_at(expire_at);

        // Acquire async mutex to serialize append operations
        // This eliminates CAS retry loops and allows other tasks to run while waiting
        let _guard = self.write_lock.lock().await;

        // With mutex held, try_append_segment should succeed on first attempt
        // (unless there's a genuine error like invalid segment ID)
        match self.try_append_segment(cache, segment_id, metrics) {
            Ok(()) => Ok(()),
            Err(error) => {
                match error {
                    AppendSegmentError::InvalidSegmentId => Err("Invalid segment ID"),
                    AppendSegmentError::SegmentNotReserved => Err("Segment must be in Reserved state"),
                    AppendSegmentError::BucketModified => Err("Bucket was concurrently modified"),
                    AppendSegmentError::InvalidTailSegmentId => Err("Invalid tail segment ID"),
                    AppendSegmentError::TailInUnexpectedState => Err("Tail segment in unexpected state"),
                    AppendSegmentError::FailedToTransitionToLinking => Err("Failed to transition segment to Linking state"),
                }
            }
        }
    }

    /// Try to evict the head segment from this TTL bucket (single attempt, no retry)
    /// Returns Ok(segment_id) if successful, Err with reason if not
    ///
    /// # Loom Test Coverage
    /// - `try_evict_head_segment` - Low-level single-attempt eviction without retry loops
    pub(crate) fn try_evict_head_segment<P: Pool>(&self, cache: &impl CacheOps<P>, metrics: &crate::metrics::CacheMetrics) -> Result<u32, &'static str> {
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (head, tail) = Self::unpack_head_tail(current_packed);

        if head == Self::INVALID_ID || head == tail {
            // Empty chain or only one segment, can't evict
            return Err("Cannot evict: empty or single segment");
        }

        let segment = cache.pool().get(head).ok_or("Invalid head segment ID")?;
        let next_id = segment.next();

        // First transition to Draining state
        if !segment.cas_metadata(
            State::Sealed,
            State::Draining,
            None,
            None,
            metrics,
        ) {
            // Segment might still be Live or in another state
            let state = segment.state();
            match state {
                State::Live => return Err("Cannot evict Live segment"),
                State::Draining | State::Locked => return Err("Segment already being evicted"),
                State::Linking => return Err("Segment in Linking state"),
                State::Reserved | State::Free => {
                    // Segment is already freed - this can happen due to races in append_segment
                    // where orphaned segments create transient inconsistencies. Another thread
                    // likely already completed the eviction. Return error to retry.
                    return Err("Segment already freed");
                }
                _ => return Err("Segment in unexpected state"),
            }
        }

        // Wait for readers to finish - bounded spinning for async compatibility
        #[cfg(not(feature = "loom"))]
        {
            const MAX_SPINS: u32 = 100_000;
            let mut spin_count = 0;

            while segment.ref_count() > 0 {
                if spin_count >= MAX_SPINS {
                    // Readers taking too long - abort removal
                    segment.cas_metadata(
                        State::Draining,
                        State::Sealed, // Restore to Sealed
                        None,
                        None,
                        metrics,
                    );
                    return Err("Segment has active readers after timeout");
                }

                // Exponential backoff: 1 spin for first 10k, then 2 spins
                if spin_count < 10_000 {
                    spin_loop();
                } else {
                    spin_loop();
                    spin_loop();
                }
                spin_count += 1;
            }
        }

        // Transition to Locked state
        if !segment.cas_metadata(
            State::Draining,
            State::Locked,
            None,
            None,
            metrics,
        ) {
            return Err("Failed to lock segment");
        }

        // Update the bucket's head pointer
        let new_head = next_id.unwrap_or(Self::INVALID_ID);
        let new_packed = Self::pack_head_tail(new_head, tail);

        if self
            .head_tail
            .compare_exchange(
                current_packed,
                new_packed,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_err()
        {
            // Someone else modified the bucket - restore state
            segment.cas_metadata(State::Locked, State::Sealed, next_id, None, metrics);
            return Err("Bucket head pointer was modified");
        }

        // Clear the evicted segment's chain links
        // Must do this before evict_and_clear since it checks for no links
        segment.cas_metadata(
            State::Locked,
            State::Locked,
            Some(INVALID_SEGMENT_ID), // Clear next pointer
            Some(INVALID_SEGMENT_ID), // Clear prev pointer
            metrics,
        );

        // AFTER successfully updating bucket head, update the next segment's prev pointer
        if let Some(next_id) = next_id
            && let Some(next_segment) = cache.pool().get(next_id)
        {
            // Try once to update prev pointer (retry loop moved to caller if needed)
            let next_state = next_segment.state();
            if next_segment.prev() == Some(head) {
                next_segment.cas_metadata(
                    next_state,
                    next_state,
                    None,
                    Some(INVALID_SEGMENT_ID),
                    metrics,
                );
            }
        }

        // Unlink all items from hashtable before freeing the segment.
        // This prevents stale hashtable entries that would cause guaranteed misses
        // when the segment is reused with new data.
        // Create ghost entries to preserve frequency history for evicted items.
        segment.unlink_all_items(cache.hashtable(), metrics, true);

        // Transition Locked -> Free, then reserve to reset stats
        // We have exclusive access (segment is Locked and unlinked from all chains)
        segment.cas_metadata(
            State::Locked,
            State::Free,
            Some(INVALID_SEGMENT_ID),
            Some(INVALID_SEGMENT_ID),
            metrics,
        );

        // try_reserve transitions Free -> Reserved and resets all stats
        segment.try_reserve();

        // Update metrics
        metrics.ttl_evict_head.increment();
        metrics.segment_evict.increment();

        Ok(head)
    }

    /// Evict the head segment from this TTL bucket with retry logic
    /// Returns the segment ID if successful, None if unable to evict
    ///
    /// # Implementation
    /// Calls `try_evict_head_segment` with retry logic for transient failures.
    ///
    /// # Loom Test Coverage
    /// - `try_evict_head_segment` (ignored) - Tests the underlying single-attempt logic
    pub(crate) fn evict_head_segment<P: Pool>(&self, cache: &impl CacheOps<P>, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        for attempt in 1..=16 {
            if attempt > 1 {
                metrics.ttl_evict_head_retry.increment();
            }

            match self.try_evict_head_segment(cache, metrics) {
                Ok(segment_id) => return Some(segment_id),
                Err(reason) => {
                    // Give up immediately for certain errors
                    match reason {
                        "Cannot evict: empty or single segment" => return None,
                        "Cannot evict Live segment" => return None,
                        "Bucket head {} points to segment in {:?} state - data structure corruption" => return None,
                        _ => {
                            // Retry for transient failures - no yield for async compatibility
                            // Just continue with next attempt
                            continue;
                        }
                    }
                }
            }
        }

        metrics.ttl_evict_head_give_up.increment();
        None
    }

    /// Expire the head segment from this TTL bucket (single attempt, no retry).
    /// This is used for proactive TTL expiration.
    ///
    /// Unlike evict_head_segment which returns the segment for immediate reuse,
    /// this function releases the segment back to the free pool.
    ///
    /// # Process
    /// 1. Get head segment and verify it exists
    /// 2. Transition head to Draining state
    /// 3. Wait for readers to finish
    /// 4. Transition to Locked state
    /// 5. Update bucket's head pointer to next segment
    /// 6. Clear the segment's chain links
    /// 7. Update next segment's prev pointer
    /// 8. Reset segment for reuse (transition to Reserved state)
    ///
    /// # Returns
    /// - `Ok(segment_id)` if segment was successfully expired and is ready for reuse
    /// - `Err(&str)` describing why expiration failed
    ///
    /// # Note
    /// The returned segment is in Reserved state with reset statistics, ready to be
    /// linked into a new chain. Stale hashtable entries pointing to this segment
    /// will be handled lazily on lookup (the items are expired).
    pub(crate) fn try_expire_head_segment<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        metrics: &crate::metrics::CacheMetrics,
        now: clocksource::coarse::Instant,
    ) -> Result<u32, &'static str> {
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (head, tail) = Self::unpack_head_tail(current_packed);

        if head == Self::INVALID_ID {
            return Err("Empty bucket");
        }

        let segment = cache.pool().get(head).ok_or("Invalid head segment ID")?;

        // Check if the segment is actually expired
        if segment.expire_at() > now {
            return Err("Segment not expired");
        }

        let next_id = segment.next();

        // Transition to Draining state (can be from Live, Sealed, or Reserved)
        let initial_state = segment.state();
        match initial_state {
            State::Free => return Err("Segment already freed"),
            State::Draining | State::Locked => return Err("Segment already being processed"),
            _ => {}
        }

        if !segment.cas_metadata(
            initial_state,
            State::Draining,
            None,
            None,
            metrics,
        ) {
            return Err("Failed to transition to Draining state");
        }

        // Wait for readers to finish - bounded spinning for async compatibility
        #[cfg(not(feature = "loom"))]
        {
            const MAX_SPINS: u32 = 100_000;
            let mut spin_count = 0;

            while segment.ref_count() > 0 {
                if spin_count >= MAX_SPINS {
                    // Readers taking too long - abort expiration
                    segment.cas_metadata(
                        State::Draining,
                        initial_state,
                        None,
                        None,
                        metrics,
                    );
                    return Err("Segment has active readers after timeout");
                }

                if spin_count < 10_000 {
                    spin_loop();
                } else {
                    spin_loop();
                    spin_loop();
                }
                spin_count += 1;
            }
        }

        #[cfg(feature = "loom")]
        {
            if segment.ref_count() > 0 {
                return Err("Segment has active readers");
            }
        }

        // Transition to Locked state
        if !segment.cas_metadata(
            State::Draining,
            State::Locked,
            None,
            None,
            metrics,
        ) {
            return Err("Failed to lock segment");
        }

        // Update the bucket's head pointer
        let new_head = next_id.unwrap_or(Self::INVALID_ID);
        let new_tail = if head == tail {
            // This was the only segment, bucket becomes empty
            Self::INVALID_ID
        } else {
            tail
        };
        let new_packed = Self::pack_head_tail(new_head, new_tail);

        if self
            .head_tail
            .compare_exchange(
                current_packed,
                new_packed,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_err()
        {
            // Someone else modified the bucket - restore state
            segment.cas_metadata(State::Locked, initial_state, next_id, None, metrics);
            return Err("Bucket head pointer was modified");
        }

        // Clear the expired segment's chain links
        segment.cas_metadata(
            State::Locked,
            State::Locked,
            Some(INVALID_SEGMENT_ID),
            Some(INVALID_SEGMENT_ID),
            metrics,
        );

        // Update the next segment's prev pointer
        if let Some(next_id) = next_id
            && let Some(next_segment) = cache.pool().get(next_id)
        {
            let next_state = next_segment.state();
            if next_segment.prev() == Some(head) {
                next_segment.cas_metadata(
                    next_state,
                    next_state,
                    None,
                    Some(INVALID_SEGMENT_ID),
                    metrics,
                );
            }
        }

        // Unlink all items from hashtable before freeing the segment.
        // Don't create ghost entries for expired items - they naturally aged out.
        segment.unlink_all_items(cache.hashtable(), metrics, false);

        // Transition Locked -> Free, then reserve to reset stats
        // We have exclusive access (segment is Locked and unlinked from all chains)
        segment.cas_metadata(
            State::Locked,
            State::Free,
            Some(INVALID_SEGMENT_ID),
            Some(INVALID_SEGMENT_ID),
            metrics,
        );

        // try_reserve transitions Free -> Reserved and resets all stats
        segment.try_reserve();

        metrics.item_expire.increment();

        Ok(head)
    }

    /// Count the number of segments in the chain from head, up to max_count.
    ///
    /// # Returns
    /// The number of segments in the chain (at most max_count).
    pub(crate) fn chain_len<P: Pool>(&self, cache: &impl CacheOps<P>, max_count: usize) -> usize {
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (head, _) = Self::unpack_head_tail(current_packed);

        if head == Self::INVALID_ID {
            return 0;
        }

        let mut count = 0;
        let mut current = head;

        while current != Self::INVALID_ID && count < max_count {
            count += 1;
            if let Some(segment) = cache.pool().get(current) {
                match segment.next() {
                    Some(next) if next != INVALID_SEGMENT_ID => current = next,
                    _ => break,
                }
            } else {
                break;
            }
        }

        count
    }

    /// Perform merge eviction on this TTL bucket.
    ///
    /// Merge eviction combines multiple segments from the head of the chain,
    /// prunes low-frequency items, and copies surviving high-frequency items
    /// into fewer destination segments. This frees up segments while retaining
    /// popular data.
    ///
    /// # Process
    /// 1. Check if we have enough segments to merge (at least merge_ratio + target_ratio)
    /// 2. Calculate frequency threshold based on utilization
    /// 3. For each source segment (merge_ratio segments from head):
    ///    a. Prune low-frequency items (collect them for async demotion)
    ///    b. Copy surviving items to destination segment(s)
    ///    c. Unlink source segment and release to free pool
    /// 4. Demote collected items to next tier asynchronously
    /// 5. Return the number of segments freed
    ///
    /// # Parameters
    /// - `cache`: Cache operations interface (provides hashtable for demotion)
    /// - `merge_ratio`: Number of segments to merge
    /// - `target_ratio`: Target number of segments after merging
    /// - `metrics`: Cache metrics
    /// - `demote_layer`: Optional layer to demote pruned items to
    ///
    /// # Returns
    /// - `Some(freed_count)`: Number of segments freed
    /// - `None`: Not enough segments to merge or merge failed
    pub async fn merge_evict<P: Pool, P2: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        merge_ratio: usize,
        _target_ratio: usize,
        metrics: &crate::metrics::CacheMetrics,
        demote_layer: Option<&CacheLayer<P2>>,
    ) -> Option<usize> {
        // We need at least 2 segments to merge (the destination + at least 1 source)
        // Plus we need to keep the tail Live, so min 3 segments in the chain
        let min_segments_needed = 3;

        // Acquire write lock for the entire merge operation
        let _guard = self.write_lock.lock().await;

        // Walk the chain to find segments and their merge counts
        // We want to start merging from the first segment with merge_count < head's merge_count
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (head_id, _tail_id) = Self::unpack_head_tail(current_packed);

        if head_id == Self::INVALID_ID {
            return None;
        }

        // Collect all segments in the chain with their merge counts
        let mut chain_info: Vec<(u32, u16)> = Vec::new(); // (segment_id, merge_count)
        let mut current_id = head_id;

        while current_id != Self::INVALID_ID {
            if let Some(segment) = cache.pool().get(current_id) {
                chain_info.push((current_id, segment.merge_count()));
                match segment.next() {
                    Some(next) if next != INVALID_SEGMENT_ID => current_id = next,
                    _ => break,
                }
            } else {
                break;
            }
        }

        let chain_length = chain_info.len();
        if chain_length < min_segments_needed {
            // Not enough segments for merge - try FIFO eviction of head instead
            // This helps with fragmentation across TTL buckets
            if chain_length >= 2 {
                // Need at least 2 segments (keep tail live)
                if let Some(_evicted_id) = self.evict_head_segment(cache, metrics) {
                    return Some(1);
                }
            }
            return None;
        }

        // Find the best starting position for merge:
        // Look for the first segment with merge_count < head's merge_count
        // This avoids re-merging segments that were just merged
        let head_merge_count = chain_info[0].1;
        let segments_to_merge = merge_ratio.min(chain_length - 1); // Keep at least 1 segment (tail)

        // Find first segment with lower merge_count that still leaves enough segments
        let mut start_idx = 0;
        for (idx, &(_seg_id, merge_count)) in chain_info.iter().enumerate() {
            // Check if starting here leaves enough segments for merge
            let remaining = chain_length - idx;
            if remaining < min_segments_needed {
                break; // Not enough segments remaining
            }

            if merge_count < head_merge_count {
                start_idx = idx;
                break;
            }
        }

        // Collect segments starting from start_idx
        let end_idx = (start_idx + segments_to_merge).min(chain_length - 1); // Keep tail live
        if end_idx <= start_idx {
            return None; // Not enough segments to merge
        }

        let all_segments: Vec<u32> = chain_info[start_idx..=end_idx]
            .iter()
            .map(|(id, _)| *id)
            .collect();

        if all_segments.len() < 2 {
            return None; // Need at least dest + 1 source
        }

        // First segment is the destination, rest are sources
        let dest_seg_id = all_segments[0];
        let source_segments = &all_segments[1..];

        // Calculate frequency threshold for pruning
        // Use a simple threshold - items below this frequency get pruned
        let threshold = 2u8;

        // Phase 1: Prune the destination segment and compact it
        // This creates free space at the end for items from source segments
        let dest_segment = cache.pool().get(dest_seg_id)?;

        // Acquire exclusive access to destination segment: CAS Sealed -> Draining
        if !dest_segment.cas_metadata(State::Sealed, State::Draining, None, None, metrics) {
            // Another thread is processing this segment
            return None;
        }

        // Wait for readers to finish before modifying
        #[cfg(not(feature = "loom"))]
        {
            const MAX_SPINS: u32 = 100_000;
            let mut spin_count = 0;
            while dest_segment.ref_count() > 0 {
                if spin_count >= MAX_SPINS {
                    // Readers taking too long - abort and restore state
                    dest_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                    return None;
                }
                if spin_count < 10_000 {
                    spin_loop();
                } else {
                    spin_loop();
                    spin_loop();
                }
                spin_count += 1;
            }
        }
        #[cfg(feature = "loom")]
        {
            if dest_segment.ref_count() > 0 {
                dest_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                return None;
            }
        }

        // Collect items to demote if we have a demote layer
        let mut all_items_to_demote: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();

        // Prune destination - collect items for demotion if layer provided
        if demote_layer.is_some() {
            let (_, _, _, _, items_to_demote) =
                dest_segment.prune_collecting_for_demote(cache.hashtable(), threshold, metrics);
            all_items_to_demote.extend(items_to_demote);
        } else {
            dest_segment.prune(cache.hashtable(), threshold, metrics);
        }
        dest_segment.compact(cache.hashtable());

        // Release exclusive access on destination: CAS Draining -> Sealed
        dest_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);

        let dest_capacity = dest_segment.data_len();
        #[allow(unused_assignments)]
        let mut dest_free_space = dest_capacity;

        // Phase 2: Process each source segment - prune (collecting demotes), then copy survivors
        let mut segments_freed = 0usize;

        for &src_seg_id in source_segments {
            let src_segment = match cache.pool().get(src_seg_id) {
                Some(seg) => seg,
                None => continue,
            };

            // Acquire exclusive access to source segment: CAS Sealed -> Draining
            if !src_segment.cas_metadata(State::Sealed, State::Draining, None, None, metrics) {
                // Another thread is processing this segment - skip
                continue;
            }

            // Wait for readers to finish before modifying
            #[cfg(not(feature = "loom"))]
            {
                const MAX_SPINS: u32 = 100_000;
                let mut spin_count = 0;
                while src_segment.ref_count() > 0 {
                    if spin_count >= MAX_SPINS {
                        // Readers taking too long - abort and restore state
                        src_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                        continue;
                    }
                    if spin_count < 10_000 {
                        spin_loop();
                    } else {
                        spin_loop();
                        spin_loop();
                    }
                    spin_count += 1;
                }
            }
            #[cfg(feature = "loom")]
            {
                if src_segment.ref_count() > 0 {
                    src_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                    continue;
                }
            }

            // Prune the source segment (now in Draining state with no readers)
            let items_retained = if demote_layer.is_some() {
                let (retained, _, _, _, items_to_demote) =
                    src_segment.prune_collecting_for_demote(cache.hashtable(), threshold, metrics);
                all_items_to_demote.extend(items_to_demote);
                retained
            } else {
                let (retained, _, _, _) = src_segment.prune(cache.hashtable(), threshold, metrics);
                retained
            };

            if items_retained == 0 {
                // Source segment is empty after pruning - restore to Sealed then remove
                src_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                if self.try_remove_segment(cache, src_seg_id, metrics).is_ok() {
                    segments_freed += 1;
                    metrics.segment_evict.increment();
                    metrics.merge_evict_segments.increment();
                }
                continue;
            }

            // Copy surviving items from source to destination
            match src_segment.copy_into(
                dest_segment,
                cache.hashtable(),
                metrics,
                |_| true,
            ) {
                Some(_) => {
                    // Restore to Sealed then remove
                    src_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                    if self.try_remove_segment(cache, src_seg_id, metrics).is_ok() {
                        segments_freed += 1;
                        metrics.segment_evict.increment();
                        metrics.merge_evict_segments.increment();
                    }
                    let new_write_offset = dest_segment.offset() as usize;
                    dest_free_space = dest_capacity - new_write_offset;
                }
                None => {
                    // Restore state and stop - destination is full
                    src_segment.cas_metadata(State::Draining, State::Sealed, None, None, metrics);
                    break;
                }
            }

            if dest_free_space < dest_capacity / 4 {
                break;
            }
        }

        // Phase 3: Demote collected items to the next tier asynchronously
        if let Some(layer) = demote_layer {
            // Get current time for TTL calculation
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);
            // Use a default TTL for demoted items (1 hour)
            let default_demote_ttl = 3600u32;
            let expire_at = current_time.saturating_add(default_demote_ttl);

            for (key, value, optional) in all_items_to_demote {
                // Best-effort demotion - ignore failures
                // Use the cache's hashtable since it's shared across all layers
                let _ = layer.append_to_small_queue(
                    &key,
                    &value,
                    &optional,
                    expire_at,
                    cache.hashtable(),
                    metrics,
                ).await;
            }
        }

        if segments_freed > 0 {
            // Increment merge count on destination to avoid re-merging it immediately
            dest_segment.increment_merge_count();
            metrics.merge_evict.increment();
            Some(segments_freed)
        } else {
            None
        }
    }

    /// Try to remove a segment from the chain (single attempt, no retry).
    /// The segment must be in Sealed state.
    ///
    /// # Chain Update Protocol
    ///
    /// When removing segment B from chain A <-> B <-> C:
    /// 1. Lock target B: Sealed → Draining → Locked
    /// 2. Lock prev A: Sealed → Relinking (A cannot be Live since B exists after it)
    /// 3. Update A's next pointer to C
    /// 4. Unlock A: Relinking → Sealed
    /// 5. Lock next C: Sealed | Live → Relinking (remembering original state)
    /// 6. Update C's prev pointer to A
    /// 7. Unlock C: Relinking → (original state)
    /// 8. Update bucket head/tail if needed
    /// 9. Clear B's links and release
    ///
    /// The Relinking state prevents concurrent modifications to chain pointers while
    /// still allowing reads of segment data.
    pub(crate) fn try_remove_segment<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        segment_id: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), RemoveSegmentError> {
        let segment = cache
            .pool()
            .get(segment_id)
            .ok_or(RemoveSegmentError::InvalidSegmentId)?;

        // Step 1: Lock target segment B: Sealed → Draining → Locked
        if !segment.cas_metadata(
            State::Sealed,
            State::Draining,
            None, // Preserve next
            None, // Preserve prev
            metrics,
        ) {
            return Err(RemoveSegmentError::FailedToTransitionToDraining);
        }

        // Wait for readers to finish
        #[cfg(not(feature = "loom"))]
        {
            const MAX_SPINS: u32 = 100_000;
            let mut spin_count = 0;

            while segment.ref_count() > 0 {
                if spin_count >= MAX_SPINS {
                    // Readers taking too long - abort removal
                    segment.cas_metadata(
                        State::Draining,
                        State::Sealed,
                        None,
                        None,
                        metrics,
                    );
                    return Err(RemoveSegmentError::SegmentHasActiveReaders);
                }

                // Exponential backoff: 1 spin for first 10k, then 2 spins
                if spin_count < 10_000 {
                    spin_loop();
                } else {
                    spin_loop();
                    spin_loop();
                }
                spin_count += 1;
            }
        }

        #[cfg(feature = "loom")]
        {
            if segment.ref_count() > 0 {
                return Err(RemoveSegmentError::SegmentHasActiveReaders);
            }
        }

        // Transition to Locked state
        if !segment.cas_metadata(
            State::Draining,
            State::Locked,
            None, // Preserve next
            None, // Preserve prev
            metrics,
        ) {
            return Err(RemoveSegmentError::FailedToTransitionToLocked);
        }

        let next_id = segment.next();
        let prev_id = segment.prev();

        // Step 2-4: Update previous segment's next pointer using Relinking protocol
        if let Some(prev_id) = prev_id {
            let prev_segment = cache
                .pool()
                .get(prev_id)
                .ok_or(RemoveSegmentError::InvalidPrevSegmentId)?;

            // Verify chain integrity
            let prev_next = prev_segment.next();
            if prev_next != Some(segment_id) {
                panic!(
                    "Chain corruption detected: segment {} prev pointer points to {}, but segment {} next pointer is {:?}",
                    segment_id, prev_id, prev_id, prev_next
                );
            }

            // Lock prev segment: Sealed → Relinking
            // (prev cannot be Live since segment_id exists after it)
            if !prev_segment.cas_metadata(
                State::Sealed,
                State::Relinking,
                None,
                None,
                metrics,
            ) {
                return Err(RemoveSegmentError::FailedToLockPrevSegment);
            }

            // Update prev's next pointer to skip over removed segment
            prev_segment.cas_metadata(
                State::Relinking,
                State::Relinking,
                Some(next_id.unwrap_or(INVALID_SEGMENT_ID)),
                None,
                metrics,
            );

            // Unlock prev segment: Relinking → Sealed
            prev_segment.cas_metadata(
                State::Relinking,
                State::Sealed,
                None,
                None,
                metrics,
            );
        }

        // Step 5-7: Update next segment's prev pointer using Relinking protocol
        if let Some(next_id) = next_id {
            let next_segment = cache
                .pool()
                .get(next_id)
                .ok_or(RemoveSegmentError::InvalidNextSegmentId)?;

            // Verify chain integrity
            let next_prev = next_segment.prev();
            if next_prev != Some(segment_id) {
                panic!(
                    "Chain corruption detected: segment {} next pointer points to {}, but segment {} prev pointer is {:?}",
                    segment_id, next_id, next_id, next_prev
                );
            }

            // Remember original state (could be Live or Sealed)
            let original_state = next_segment.state();

            // Lock next segment: (Sealed | Live) → Relinking
            if !next_segment.cas_metadata(
                original_state,
                State::Relinking,
                None,
                None,
                metrics,
            ) {
                return Err(RemoveSegmentError::FailedToLockNextSegment);
            }

            // Update next's prev pointer to skip over removed segment
            next_segment.cas_metadata(
                State::Relinking,
                State::Relinking,
                None,
                Some(prev_id.unwrap_or(INVALID_SEGMENT_ID)),
                metrics,
            );

            // Unlock next segment: Relinking → (original state)
            next_segment.cas_metadata(
                State::Relinking,
                original_state,
                None,
                None,
                metrics,
            );
        }

        // Step 8: Update bucket head/tail if necessary (single attempt)
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (current_head, current_tail) = Self::unpack_head_tail(current_packed);

        let new_head = if current_head == segment_id {
            next_id.unwrap_or(Self::INVALID_ID)
        } else {
            current_head
        };

        let new_tail = if current_tail == segment_id {
            prev_id.unwrap_or(Self::INVALID_ID)
        } else {
            current_tail
        };

        // Only update if necessary
        if new_head != current_head || new_tail != current_tail {
            let new_packed = Self::pack_head_tail(new_head, new_tail);
            if self
                .head_tail
                .compare_exchange(
                    current_packed,
                    new_packed,
                    Ordering::Release,
                    Ordering::Acquire,
                )
                .is_err()
            {
                return Err(RemoveSegmentError::BucketModified);
            }
        }

        // Step 9: Clear segment's links and mark as Reserved
        if !segment.cas_metadata(
            State::Locked,
            State::Reserved,
            Some(INVALID_SEGMENT_ID),
            Some(INVALID_SEGMENT_ID),
            metrics,
        ) {
            panic!(
                "Failed to clear segment {} links and transition to Reserved - segment in unexpected state",
                segment_id
            );
        }

        // Clear bucket ID before returning to free pool
        segment.clear_bucket_id();

        // Return the segment to the free pool for reuse
        cache.pool().release(segment_id, metrics);

        Ok(())
    }

    /// Try to append an item to the current tail segment (single attempt, no retry)
    ///
    /// Returns Ok((segment_id, offset)) if the item was successfully appended to the current tail.
    /// Returns Err with reason if:
    /// - No tail segment exists
    /// - Tail segment is not in Live state
    /// - Tail segment is full
    ///
    /// # Loom Test Coverage
    /// - `try_append_item` - Low-level single-attempt item append without retry loops
    pub(crate) fn try_append_item<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(u32, u32), AppendItemError> {
        let current_packed = self.head_tail.load(Ordering::Acquire);
        let (_head, tail) = Self::unpack_head_tail(current_packed);

        // Check if we have a tail segment
        if tail == Self::INVALID_ID {
            return Err(AppendItemError::NoTailSegment);
        }

        let segment = cache.pool().get(tail).ok_or(AppendItemError::InvalidTailSegmentId)?;

        // Only append to Live segments
        if segment.state() != State::Live {
            return Err(AppendItemError::TailNotLive);
        }

        // Try to append the item
        match segment.append_item(key, value, optional, metrics) {
            Some(offset) => Ok((tail, offset)),
            None => Err(AppendItemError::TailSegmentFull),
        }
    }

    /// Append an item to this TTL bucket
    /// Returns (segment_id, offset) if successful
    ///
    /// # Loom Test Coverage
    /// - `concurrent_insert_same_segment` (ignored) - Two threads inserting through full path with retry loops
    pub async fn append_item<P: Pool>(
        &self,
        cache: &impl CacheOps<P>,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<(u32, u32)> {
        let mut attempts = 0;

        loop {
            attempts += 1;
            if attempts > 16 {
                // Tried 16 times, give up
                return None;
            }

            // Add backoff after first few attempts to reduce contention
            if attempts > 4 {
                // Use async yield instead of spin loop
                tokio::task::yield_now().await;
            }

            // Step 1: Try to append to current tail if it's Live
            match self.try_append_item(cache, key, value, optional, metrics) {
                Ok((segment_id, offset)) => return Some((segment_id, offset)),
                Err(error) => {
                    // If tail is full, try to seal it before proceeding
                    if error == AppendItemError::TailSegmentFull {
                        let current_packed = self.head_tail.load(Ordering::Acquire);
                        let (_head, tail) = Self::unpack_head_tail(current_packed);
                        if tail != Self::INVALID_ID {
                            let segment = cache.pool().get(tail).expect("Invalid tail segment ID");
                            // Try to seal it - preserving links
                            segment.cas_metadata(
                                State::Live,
                                State::Sealed,
                                None,  // Preserve current next
                                None,  // Preserve current prev
                                metrics,
                            );
                        }
                    }
                    // Fall through to try allocating a new segment
                }
            }

            let current_packed = self.head_tail.load(Ordering::Acquire);
            let (_head, tail) = Self::unpack_head_tail(current_packed);

            // Step 2: Try to get a free main cache segment
            match cache.pool().reserve_main_cache(metrics) {
                Some(new_segment_id) => {
                    // Try to append the new segment to this bucket
                    match self.append_segment(cache, new_segment_id, metrics).await {
                        Ok(_) => {
                            // New segment added to bucket, now append the item
                            let segment = cache.pool().get(new_segment_id).expect("Invalid segment ID");
                            if let Some(offset) = segment.append_item(key, value, optional, metrics) {
                                return Some((new_segment_id, offset));
                            }
                            // This shouldn't happen with a fresh segment, but handle gracefully
                            // The segment is already linked, so we can't release it back
                            // Just continue to try another segment
                        }
                        Err(_) => {
                            // Failed to append segment to bucket
                            // append_segment already released the segment back to the pool
                            // Just retry with a new segment
                            continue;
                        }
                    }
                }
                None => {
                    // Step 3: No free segments, try to expire one segment first (cheaper than eviction)
                    let segment_id = if let Some(id) = cache.try_expire() {
                        id
                    } else {
                        // No expired segments, use cache-wide eviction according to policy
                        match cache.evict_segment_by_policy().await {
                        Some(id) => id,
                        None => {
                            // Eviction failed, try to append to tail as last resort
                            // (another thread might have made space available)
                            if tail != Self::INVALID_ID {
                                let segment = cache.pool().get(tail).expect("Invalid tail segment ID");
                                if segment.state() == State::Live
                                    && let Some(offset) = segment.append_item(key, value, optional, metrics)
                                {
                                    return Some((tail, offset));
                                }
                            }
                            // Continue retry without yield for async compatibility
                            continue;
                        }
                        }
                    };

                    // Now append the expired/evicted segment as the new tail
                    match self.append_segment(cache, segment_id, metrics).await {
                        Ok(_) => {
                            // Successfully appended the segment, now append the item
                            let segment = cache.pool().get(segment_id).expect("Invalid segment ID");
                            if let Some(offset) = segment.append_item(key, value, optional, metrics) {
                                return Some((segment_id, offset));
                            }
                            // This shouldn't happen with a freshly cleared segment
                            return None;
                        }
                        Err(_) => {
                            // Failed to append the segment
                            // append_segment already released the segment back to the pool
                            // Just return None
                            return None;
                        }
                    }
                }
            }
        }
    }
}

// #[cfg(all(test, not(feature = "loom")))]
// mod tests {
//     use super::*;
//     use crate::{Cache, State};
//     use crate::segments::INVALID_SEGMENT_ID;

//     #[test]
//     fn test_item_append_increments_live_items() {
//         let cache = Cache::new();

//         // Reserve a segment
//         let seg_id = cache.segments().reserve(cache.metrics()).unwrap();
//         let segment = cache.segments().get(seg_id).unwrap();

//         // Initially no items
//         assert_eq!(segment.live_items(), 0);

//         // Append an item
//         let offset = segment.append_item(b"key1", b"value1", b"", cache.metrics());
//         assert!(offset.is_some(), "Should successfully append item");
//         assert_eq!(segment.live_items(), 1);

//         // Append another item
//         let offset2 = segment.append_item(b"key2", b"value2", b"", cache.metrics());
//         assert!(offset2.is_some(), "Should successfully append second item");
//         assert_eq!(segment.live_items(), 2);

//         // Offsets should be different
//         assert_ne!(offset.unwrap(), offset2.unwrap());
//     }

//     #[test]
//     fn test_segment_clear_reduces_live_items() {
//         let cache = Cache::new();

//         // Reserve and append items
//         let seg_id = cache.segments().reserve(cache.metrics()).unwrap();
//         let segment = cache.segments().get(seg_id).unwrap();

//         segment.append_item(b"key1", b"value1", b"", cache.metrics()).unwrap();
//         segment.append_item(b"key2", b"value2", b"", cache.metrics()).unwrap();
//         assert_eq!(segment.live_items(), 2);

//         // Transition to Locked state (required for evict_and_clear)
//         segment.cas_metadata(
//             SegmentState::Reserved,
//             SegmentState::Locked,
//             None,
//             None,
//             cache.metrics(),
//         );

//         // Clear the segment
//         let cleared = cache.segments().evict_and_clear(seg_id, cache.hashtable(), cache.metrics());
//         assert!(cleared, "Should successfully clear segment");

//         // Live items should be reset
//         assert_eq!(segment.live_items(), 0);
//         assert_eq!(segment.write_offset(), 0);
//     }

//     #[tokio::test]
//     async fn test_ttl_bucket_append_segment() {
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Initially empty
//         assert_eq!(bucket.head(), None);
//         assert_eq!(bucket.tail(), None);

//         // Reserve and append a segment
//         let seg_id = cache.segments().reserve(cache.metrics()).unwrap();
//         let result = bucket.append_segment(&cache, seg_id, cache.metrics());
//         assert!(result.await.is_ok(), "Should successfully append segment");

//         // Bucket should have segment as both head and tail
//         assert_eq!(bucket.head(), Some(seg_id));
//         assert_eq!(bucket.tail(), Some(seg_id));

//         // Segment should be in Live state
//         let segment = cache.segments().get(seg_id).unwrap();
//         assert_eq!(segment.state(), SegmentState::Live);
//     }

//     #[tokio::test]
//     async fn test_ttl_bucket_append_multiple_segments() {
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Append first segment
//         let seg_id1 = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_id1, cache.metrics()).await.unwrap();

//         // Append second segment
//         let seg_id2 = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_id2, cache.metrics()).await.unwrap();

//         // Head should be first, tail should be second
//         assert_eq!(bucket.head(), Some(seg_id1));
//         assert_eq!(bucket.tail(), Some(seg_id2));

//         // Verify linked list structure
//         let seg1 = cache.segments().get(seg_id1).unwrap();
//         let seg2 = cache.segments().get(seg_id2).unwrap();

//         assert_eq!(seg1.prev(), None);
//         assert_eq!(seg1.next(), Some(seg_id2));
//         assert_eq!(seg2.prev(), Some(seg_id1));
//         assert_eq!(seg2.next(), None);

//         // First segment should be sealed, second should be live
//         assert_eq!(seg1.state(), SegmentState::Sealed);
//         assert_eq!(seg2.state(), SegmentState::Live);
//     }

//     #[tokio::test]
//     async fn test_ttl_bucket_append_item() {
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Append an item (should allocate a segment automatically)
//         let result = bucket.append_item(&cache, b"key1", b"value1", b"", cache.metrics()).await;
//         assert!(result.is_some(), "Should successfully append item");

//         let (seg_id, offset) = result.unwrap();

//         // Verify segment was created
//         assert_eq!(bucket.tail(), Some(seg_id));

//         // Verify item was added
//         let segment = cache.segments().get(seg_id).unwrap();
//         assert_eq!(segment.live_items(), 1);
//         assert!(offset < segment.write_offset());
//     }

//     #[tokio::test]
//     async fn test_segment_full_triggers_new_segment() {
//         // Create a cache with small segments
//         let cache = crate::CacheBuilder::new()
//             .hashtable_power(4) // Small hashtable (16 buckets)
//             .segment_size(256) // Small segments
//             .heap_size(1024)   // Room for multiple segments
//             .build();

//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Append a large item to fill most of the first segment
//         let result1 = bucket.append_item(&cache, b"key1", &[0u8; 128], b"", cache.metrics()).await;
//         assert!(result1.is_some());
//         let (seg_id1, _) = result1.unwrap();

//         // Append another large item - should trigger new segment
//         let result2 = bucket.append_item(&cache, b"key2", &[0u8; 128], b"", cache.metrics()).await;
//         assert!(result2.is_some());
//         let (seg_id2, _) = result2.unwrap();

//         // Should be in different segments
//         assert_ne!(seg_id1, seg_id2);

//         // First segment should be sealed, second should be live
//         assert_eq!(cache.segments().get(seg_id1).unwrap().state(), SegmentState::Sealed);
//         assert_eq!(cache.segments().get(seg_id2).unwrap().state(), SegmentState::Live);
//     }

//     #[tokio::test]
//     async fn test_evict_head_segment() {
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Append two segments
//         let seg_id1 = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_id1, cache.metrics()).await.unwrap();

//         let seg_id2 = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_id2, cache.metrics()).await.unwrap();

//         assert_eq!(bucket.head(), Some(seg_id1));
//         assert_eq!(bucket.tail(), Some(seg_id2));

//         // Evict the head
//         let evicted = bucket.evict_head_segment(&cache, cache.metrics());
//         assert_eq!(evicted, Some(seg_id1));

//         // Head should now be seg_id2
//         assert_eq!(bucket.head(), Some(seg_id2));
//         assert_eq!(bucket.tail(), Some(seg_id2));

//         // Evicted segment should be back in Reserved state
//         assert_eq!(cache.segments().get(seg_id1).unwrap().state(), SegmentState::Reserved);
//     }

//     #[tokio::test]
//     async fn test_remove_middle_segment() {
//         // Test removing segment B from chain A <-> B <-> C
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Create 3-segment chain: A <-> B <-> C
//         let seg_a = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_a, cache.metrics()).await.unwrap();

//         let seg_b = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_b, cache.metrics()).await.unwrap();

//         let seg_c = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_c, cache.metrics()).await.unwrap();

//         // Verify initial chain structure
//         assert_eq!(bucket.head(), Some(seg_a));
//         assert_eq!(bucket.tail(), Some(seg_c));

//         let segment_a = cache.segments().get(seg_a).unwrap();
//         let segment_b = cache.segments().get(seg_b).unwrap();
//         let segment_c = cache.segments().get(seg_c).unwrap();

//         assert_eq!(segment_a.prev(), None);
//         assert_eq!(segment_a.next(), Some(seg_b));
//         assert_eq!(segment_a.state(), SegmentState::Sealed);

//         assert_eq!(segment_b.prev(), Some(seg_a));
//         assert_eq!(segment_b.next(), Some(seg_c));
//         assert_eq!(segment_b.state(), SegmentState::Sealed);

//         assert_eq!(segment_c.prev(), Some(seg_b));
//         assert_eq!(segment_c.next(), None);
//         assert_eq!(segment_c.state(), SegmentState::Live);

//         // Remove middle segment B
//         let result = bucket.remove_segment(&cache, seg_b, cache.metrics()).await;
//         assert!(result.is_ok(), "Should successfully remove middle segment");

//         // Verify chain is now A <-> C
//         assert_eq!(bucket.head(), Some(seg_a));
//         assert_eq!(bucket.tail(), Some(seg_c));

//         // A should now point to C
//         assert_eq!(segment_a.prev(), None);
//         assert_eq!(segment_a.next(), Some(seg_c));
//         assert_eq!(segment_a.state(), SegmentState::Sealed);

//         // C should now point back to A
//         assert_eq!(segment_c.prev(), Some(seg_a));
//         assert_eq!(segment_c.next(), None);
//         assert_eq!(segment_c.state(), SegmentState::Live);

//         // B should have no links and be in Free state (after release)
//         assert_eq!(segment_b.prev(), None);
//         assert_eq!(segment_b.next(), None);
//         assert_eq!(segment_b.state(), SegmentState::Free);
//     }

//     #[tokio::test]
//     async fn test_remove_segment_from_two_segment_chain() {
//         // Test removing first segment from a 2-segment chain
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Create 2-segment chain: A <-> B
//         let seg_a = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_a, cache.metrics()).await.unwrap();

//         let seg_b = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_b, cache.metrics()).await.unwrap();

//         assert_eq!(bucket.head(), Some(seg_a));
//         assert_eq!(bucket.tail(), Some(seg_b));

//         let segment_a = cache.segments().get(seg_a).unwrap();
//         let segment_b = cache.segments().get(seg_b).unwrap();

//         // Remove A (which only has next, no prev)
//         let result = bucket.remove_segment(&cache, seg_a, cache.metrics()).await;
//         assert!(result.is_ok(), "Should successfully remove first segment");

//         // Head should now be B
//         assert_eq!(bucket.head(), Some(seg_b));
//         assert_eq!(bucket.tail(), Some(seg_b));

//         // B should have no prev link
//         assert_eq!(segment_b.prev(), None);
//         assert_eq!(segment_b.next(), None);
//         assert_eq!(segment_b.state(), SegmentState::Live);

//         // A should be freed
//         assert_eq!(segment_a.state(), SegmentState::Free);
//     }

//     #[tokio::test]
//     async fn test_remove_segment_preserves_live_state() {
//         // Test that the next segment remains Live after removal if it was Live
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Create 2-segment chain: A <-> B (B is Live)
//         let seg_a = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_a, cache.metrics()).await.unwrap();

//         let seg_b = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_b, cache.metrics()).await.unwrap();

//         let segment_b = cache.segments().get(seg_b).unwrap();
//         assert_eq!(segment_b.state(), SegmentState::Live, "B should start Live");

//         // Remove A
//         let result = bucket.remove_segment(&cache, seg_a, cache.metrics()).await;
//         assert!(result.is_ok());

//         // B should still be Live
//         assert_eq!(segment_b.state(), SegmentState::Live, "B should remain Live after removal");
//     }

//     #[tokio::test]
//     #[should_panic(expected = "Chain corruption detected")]
//     async fn test_remove_segment_detects_chain_corruption() {
//         // Test that remove_segment panics on corrupted chain
//         let cache = Cache::new();
//         let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//         // Create 3-segment chain
//         let seg_a = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_a, cache.metrics()).await.unwrap();

//         let seg_b = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_b, cache.metrics()).await.unwrap();

//         let seg_c = cache.segments().reserve(cache.metrics()).unwrap();
//         bucket.append_segment(&cache, seg_c, cache.metrics()).await.unwrap();

//         // Corrupt the chain by breaking A's next pointer
//         let segment_a = cache.segments().get(seg_a).unwrap();
//         segment_a.cas_metadata(
//             SegmentState::Sealed,
//             SegmentState::Sealed,
//             Some(INVALID_SEGMENT_ID), // Break the link to B
//             None,
//             cache.metrics(),
//         );

//         // This should panic when it detects B's prev points to A but A's next doesn't point to B
//         bucket.remove_segment(&cache, seg_b, cache.metrics()).await.unwrap();
//     }
// }

// #[cfg(all(test, feature = "loom"))]
// mod loom_tests {
//     use super::*;
//     use loom::sync::Arc;
//     use loom::thread;
//     use crate::{Cache, SegmentState};

//     #[test]
//     fn try_append_item() {
//         // Test low-level item append to existing live segment without retry loops
//         // Set up a live segment at the tail to avoid testing append_segment/evict_head_segment
//         use crate::segments::INVALID_SEGMENT_ID;

//         loom::model(|| {
//             let cache = Arc::new(Cache::new());

//             // Reserve a segment and manually set it up as a Live tail
//             let seg_id = cache.segments().reserve(cache.metrics()).unwrap();
//             let segment = cache.segments().get(seg_id).unwrap();

//             // Manually transition to Live state with no links (it's both head and tail)
//             segment.cas_metadata(
//                 SegmentState::Reserved,
//                 SegmentState::Live,
//                 Some(INVALID_SEGMENT_ID), // next = invalid
//                 Some(INVALID_SEGMENT_ID), // prev = invalid
//                 cache.metrics(),
//             );

//             // Set expiration time
//             let expire_at = clocksource::coarse::Instant::now() + clocksource::coarse::Duration::from_secs(60);
//             segment.set_expire_at(expire_at);

//             // Manually set up the bucket to have this segment as both head and tail
//             let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);
//             bucket.test_set_head_tail(seg_id, seg_id);

//             let c1 = Arc::clone(&cache);
//             let c2 = Arc::clone(&cache);

//             // Two threads try to append items to the same live segment
//             let t1 = thread::spawn(move || {
//                 let bucket = c1.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_append_item(c1.as_ref(), b"key1", b"value1", b"", c1.metrics())
//             });

//             let t2 = thread::spawn(move || {
//                 let bucket = c2.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_append_item(c2.as_ref(), b"key2", b"value2", b"", c2.metrics())
//             });

//             let result1 = t1.join().unwrap();
//             let result2 = t2.join().unwrap();

//             // Both should succeed (the segment should have space for both items)
//             assert!(result1.is_ok(), "First append should succeed");
//             assert!(result2.is_ok(), "Second append should succeed");

//             let (seg1, offset1) = result1.unwrap();
//             let (seg2, offset2) = result2.unwrap();

//             // Both should be to the same segment
//             assert_eq!(seg1, seg_id);
//             assert_eq!(seg2, seg_id);

//             // Offsets should be different
//             assert_ne!(offset1, offset2, "Offsets should be different");

//             // Segment should have 2 items
//             assert_eq!(segment.live_items(), 2, "Segment should have 2 live items");
//         });
//     }

//     #[test]
//     fn try_append_segment() {
//         // Test low-level segment append without retry loops
//         // Two threads racing to append to an empty bucket
//         loom::model(|| {
//             let cache = Arc::new(Cache::new());

//             // Reserve 2 segments
//             let seg_id1 = cache.segments().reserve(cache.metrics()).unwrap();
//             let seg_id2 = cache.segments().reserve(cache.metrics()).unwrap();

//             // Set expiration times manually (since try_append_segment doesn't do this)
//             let expire_at = clocksource::coarse::Instant::now() + clocksource::coarse::Duration::from_secs(60);
//             cache.segments().get(seg_id1).unwrap().set_expire_at(expire_at);
//             cache.segments().get(seg_id2).unwrap().set_expire_at(expire_at);

//             let c1 = Arc::clone(&cache);
//             let c2 = Arc::clone(&cache);

//             // Two threads try to append to the same empty bucket
//             let t1 = thread::spawn(move || {
//                 let bucket = c1.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_append_segment(c1.as_ref(), seg_id1, c1.metrics())
//             });

//             let t2 = thread::spawn(move || {
//                 let bucket = c2.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_append_segment(c2.as_ref(), seg_id2, c2.metrics())
//             });

//             let result1 = t1.join().unwrap();
//             let result2 = t2.join().unwrap();

//             // At least one should succeed
//             let success_count = [result1.is_ok(), result2.is_ok()].iter().filter(|&&x| x).count();
//             assert!(success_count >= 1, "At least one append should succeed");

//             if success_count == 2 {
//                 // Both succeeded - they should be linked
//                 let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);
//                 let head = bucket.head().unwrap();
//                 let tail = bucket.tail().unwrap();
//                 assert_ne!(head, tail, "Head and tail should be different when both appends succeed");

//                 // Verify the linked list structure
//                 let head_segment = cache.segments().get(head).unwrap();
//                 let tail_segment = cache.segments().get(tail).unwrap();

//                 assert_eq!(head_segment.prev(), None, "Head should have no prev");
//                 assert_eq!(tail_segment.next(), None, "Tail should have no next");
//                 assert_eq!(head_segment.next(), Some(tail), "Head should point to tail");
//                 assert_eq!(tail_segment.prev(), Some(head), "Tail should point to head");
//             } else {
//                 // Only one succeeded - it should be both head and tail
//                 let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);
//                 let head = bucket.head().unwrap();
//                 let tail = bucket.tail().unwrap();
//                 assert_eq!(head, tail, "When only one append succeeds, head should equal tail");

//                 // The failed segment's state depends on where the failure occurred:
//                 // - "Failed to transition to Linking state" → stays Reserved
//                 // - "Bucket was concurrently modified" → released to Free
//                 let (success_id, failed_id) = if result1.is_ok() {
//                     (seg_id1, seg_id2)
//                 } else {
//                     (seg_id2, seg_id1)
//                 };

//                 assert_eq!(head, success_id);
//                 let failed_state = cache.segments().get(failed_id).unwrap().state();
//                 assert!(
//                     matches!(failed_state, SegmentState::Reserved | SegmentState::Free),
//                     "Failed segment should be Reserved or Free, got {:?}",
//                     failed_state
//                 );
//             }
//         });
//     }

//     #[test]
//     fn try_evict_head_segment() {
//         // Test low-level head eviction without retry loops
//         // Set up state manually to avoid going through append_segment's retry logic
//         use crate::segments::INVALID_SEGMENT_ID;

//         loom::model(|| {
//             let cache = Arc::new(Cache::new());

//             // Reserve 2 segments
//             let seg_id1 = cache.segments().reserve(cache.metrics()).unwrap();
//             let seg_id2 = cache.segments().reserve(cache.metrics()).unwrap();

//             let seg1 = cache.segments().get(seg_id1).unwrap();
//             let seg2 = cache.segments().get(seg_id2).unwrap();

//             // Manually set up the chain: seg1 (Sealed) -> seg2 (Live)
//             // seg1: head, Sealed state, next=seg2, prev=None
//             seg1.cas_metadata(SegmentState::Reserved, SegmentState::Sealed, Some(seg_id2), Some(INVALID_SEGMENT_ID), cache.metrics());

//             // seg2: tail, Live state, next=None, prev=seg1
//             seg2.cas_metadata(SegmentState::Reserved, SegmentState::Live, Some(INVALID_SEGMENT_ID), Some(seg_id1), cache.metrics());

//             // Manually set up the bucket's head and tail pointers using test helper
//             let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);
//             bucket.test_set_head_tail(seg_id1, seg_id2);

//             let c1 = Arc::clone(&cache);
//             let c2 = Arc::clone(&cache);

//             // Two threads try to evict the head
//             let t1 = thread::spawn(move || {
//                 let bucket = c1.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_evict_head_segment(c1.as_ref(), c1.metrics())
//             });

//             let t2 = thread::spawn(move || {
//                 let bucket = c2.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_evict_head_segment(c2.as_ref(), c2.metrics())
//             });

//             let result1 = t1.join().unwrap();
//             let result2 = t2.join().unwrap();

//             // Exactly one should succeed
//             let success_count = [result1.is_ok(), result2.is_ok()].iter().filter(|&&x| x).count();
//             assert_eq!(success_count, 1, "Exactly one eviction should succeed");

//             // The successful eviction should return seg_id1
//             if result1.is_ok() {
//                 assert_eq!(result1.unwrap(), seg_id1);
//             } else {
//                 assert_eq!(result2.unwrap(), seg_id1);
//             }

//             // After eviction, bucket head should be seg_id2
//             let new_head = bucket.head().unwrap();
//             assert_eq!(new_head, seg_id2);

//             // seg_id1 should be in Reserved state (ready for reuse)
//             assert_eq!(cache.segments().get(seg_id1).unwrap().state(), SegmentState::Reserved);
//         });
//     }

//     #[test]
//     fn concurrent_ttl_bucket_append() {
//         loom::model(|| {

//             let cache = Arc::new(Cache::new());

//             // Reserve two segments
//             let seg_id1 = cache.segments().reserve(cache.metrics()).unwrap();
//             let seg_id2 = cache.segments().reserve(cache.metrics()).unwrap();

//             let cache1 = Arc::clone(&cache);
//             let cache2 = Arc::clone(&cache);

//             // Two threads try to append segments to the same bucket
//             let t1 = thread::spawn(move || {
//                 let bucket = cache1.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.append_segment(cache1.as_ref(), seg_id1, cache1.metrics())
//             });
//             let t2 = thread::spawn(move || {
//                 let bucket = cache2.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.append_segment(cache2.as_ref(), seg_id2, cache2.metrics())
//             });

//             let result1 = t1.join().unwrap();
//             let result2 = t2.join().unwrap();

//             // In some interleavings, one may fail due to concurrent modification
//             // At least one should succeed
//             let success_count = [result1.is_ok(), result2.is_ok()].iter().filter(|&&x| x).count();
//             assert!(success_count >= 1, "At least one append should succeed");

//             if success_count == 2 {
//                 // Both succeeded - verify linked list structure
//                 let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);
//                 let head = bucket.head().unwrap();
//                 let tail = bucket.tail().unwrap();
//                 assert_ne!(head, tail, "Head and tail should be different");

//                 let head_segment = cache.segments().get(head).unwrap();
//                 let tail_segment = cache.segments().get(tail).unwrap();

//                 // Head should have no prev, tail should have no next
//                 assert_eq!(head_segment.prev(), None, "Head should have no prev");
//                 assert_eq!(tail_segment.next(), None, "Tail should have no next");

//                 // They should be linked to each other
//                 assert_eq!(head_segment.next(), Some(tail), "Head should point to tail");
//                 assert_eq!(tail_segment.prev(), Some(head), "Tail should point to head");

//                 // Both should be in Live or Sealed state
//                 assert!(
//                     matches!(head_segment.state(), SegmentState::Live | SegmentState::Sealed),
//                     "Head segment in unexpected state: {:?}",
//                     head_segment.state()
//                 );
//                 assert!(
//                     matches!(tail_segment.state(), SegmentState::Live | SegmentState::Sealed),
//                     "Tail segment in unexpected state: {:?}",
//                     tail_segment.state()
//                 );
//             }

//             // Metrics are tracked correctly (verified by simpler tests with LOOM_MAX_PREEMPTIONS=1)
//         });
//     }

//     #[test]
//     fn try_remove_segment() {
//         // Test low-level segment removal with Relinking protocol
//         // Two threads trying to remove the same segment - only one should succeed
//         use crate::segments::INVALID_SEGMENT_ID;

//         loom::model(|| {
//             let cache = Arc::new(Cache::new());
//             let bucket = cache.ttl_buckets().get_bucket_for_seconds(60);

//             // Create 3-segment chain: A <-> B <-> C
//             let seg_a = cache.segments().reserve(cache.metrics()).unwrap();
//             let seg_b = cache.segments().reserve(cache.metrics()).unwrap();
//             let seg_c = cache.segments().reserve(cache.metrics()).unwrap();

//             // Set up the chain manually to avoid testing append_segment
//             let segment_a = cache.segments().get(seg_a).unwrap();
//             let segment_b = cache.segments().get(seg_b).unwrap();
//             let segment_c = cache.segments().get(seg_c).unwrap();

//             let expire_at = clocksource::coarse::Instant::now() + clocksource::coarse::Duration::from_secs(60);
//             segment_a.set_expire_at(expire_at);
//             segment_b.set_expire_at(expire_at);
//             segment_c.set_expire_at(expire_at);

//             // A: Sealed, no prev, next -> B
//             segment_a.cas_metadata(
//                 SegmentState::Reserved,
//                 SegmentState::Sealed,
//                 Some(seg_b),
//                 Some(INVALID_SEGMENT_ID),
//                 cache.metrics(),
//             );

//             // B: Sealed, prev -> A, next -> C
//             segment_b.cas_metadata(
//                 SegmentState::Reserved,
//                 SegmentState::Sealed,
//                 Some(seg_c),
//                 Some(seg_a),
//                 cache.metrics(),
//             );

//             // C: Live, prev -> B, no next
//             segment_c.cas_metadata(
//                 SegmentState::Reserved,
//                 SegmentState::Live,
//                 Some(INVALID_SEGMENT_ID),
//                 Some(seg_b),
//                 cache.metrics(),
//             );

//             // Set bucket head/tail
//             bucket.test_set_head_tail(seg_a, seg_c);

//             let c1 = Arc::clone(&cache);
//             let c2 = Arc::clone(&cache);

//             // Two threads try to remove B concurrently
//             let t1 = thread::spawn(move || {
//                 let bucket = c1.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_remove_segment(c1.as_ref(), seg_b, c1.metrics())
//             });

//             let t2 = thread::spawn(move || {
//                 let bucket = c2.ttl_buckets().get_bucket_for_seconds(60);
//                 bucket.try_remove_segment(c2.as_ref(), seg_b, c2.metrics())
//             });

//             let result1 = t1.join().unwrap();
//             let result2 = t2.join().unwrap();

//             // Exactly one should succeed
//             let success_count = [result1.is_ok(), result2.is_ok()].iter().filter(|&&x| x).count();
//             assert_eq!(success_count, 1, "Exactly one removal should succeed");

//             // After removal, A should link to C
//             assert_eq!(segment_a.next(), Some(seg_c), "A should link to C");
//             assert_eq!(segment_c.prev(), Some(seg_a), "C should link back to A");

//             // B should be freed
//             assert_eq!(segment_b.state(), SegmentState::Free, "B should be Free");
//             assert_eq!(segment_b.next(), None, "B should have no next link");
//             assert_eq!(segment_b.prev(), None, "B should have no prev link");

//             // Bucket head/tail should still be correct
//             assert_eq!(bucket.head(), Some(seg_a));
//             assert_eq!(bucket.tail(), Some(seg_c));
//         });
//     }
// }
