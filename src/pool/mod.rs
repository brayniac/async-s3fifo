mod memory;
mod mmap;
mod direct_io;

use crate::segment::Segment;
pub use mmap::*;
pub use memory::*;
pub use direct_io::*;

pub trait Pool {
    type Segment: Segment;

    /// Get a segment by ID from the segment pool
    fn get(&self, id: u32) -> Option<&Self::Segment>;

    /// Get the total number of segments in this pool
    fn segment_count(&self) -> usize;

    /// Reserve a small queue segment from the pool.
    ///
    /// Returns the segment ID if one is available, or None if all small queue segments are in use.
    /// The segment's statistics are reset upon reservation.
    ///
    /// # Note
    /// Small queue segments have per-item TTL and are used for the admission queue.
    fn reserve_small_queue(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32>;

    /// Reserve a main cache segment from the pool.
    ///
    /// Returns the segment ID if one is available, or None if all main cache segments are in use.
    /// The segment's statistics are reset upon reservation.
    ///
    /// # Note
    /// Main cache segments have segment-level TTL and are used for TTL buckets.
    fn reserve_main_cache(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32>;

    /// Release a segment back to the appropriate free queue for reuse.
    ///
    /// The segment is automatically returned to the correct free queue (small queue or main cache)
    /// based on its type.
    ///
    /// # Panics
    /// Panics if the segment is not in Reserved or Linking state, or if the segment ID is invalid.
    ///
    /// # Note
    /// The caller should ensure the segment is no longer referenced before releasing.
    /// Attempting to release a segment in Free state (double-release) will panic.
    ///
    /// # Loom Test Coverage
    /// - `single_segment_reserve_metrics` - Single-threaded release with metric validation
    /// - `concurrent_segment_reserve_and_release` - Two threads releasing different segments
    fn release(&self, id: u32, metrics: &crate::metrics::CacheMetrics);

    // /// Clear a segment and prepare it for reuse without adding to free queue.
    // ///
    // /// Used by the eviction path where the segment is already in Locked state
    // /// and has been unlinked from chains.
    // ///
    // /// Unlike `clear()`, does NOT handle concurrent calls - the caller must ensure
    // /// exclusive access by transitioning the segment to Locked state first.
    // ///
    // /// # Returns
    // /// - `true` if segment was successfully cleared and is now in Reserved state
    // /// - `false` if segment is not in Locked state
    // ///
    // /// # Panics
    // /// - If the segment has chain links (must be unlinked from TTL bucket first)
    // /// - If segment data is corrupted
    // /// - If state transition from Locked to Reserved fails
    // fn evict_and_clear(&self, id: u32, hashtable: &Hashtable, metrics: &crate::metrics::CacheMetrics) -> bool;
}