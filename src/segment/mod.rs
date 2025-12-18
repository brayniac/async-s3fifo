use crate::GetItemError;

mod slice;

pub use slice::SliceSegment;

pub trait Segment {
    /// Get the segment ID
    fn id(&self) -> u32;

    /// Get the pool ID this segment belongs to
    fn pool_id(&self) -> u8;

    /// Check if this segment is part of the small queue (admission queue)
    fn is_small_queue(&self) -> bool;

    /// Get a raw slice of the segment data at the given offset and length.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - `offset + len <= data_len()`
    /// - The returned slice is only used while the segment is in a readable state
    ///
    /// # Returns
    /// `Some(&[u8])` if the range is valid, `None` otherwise.
    fn data_slice(&self, offset: u32, len: usize) -> Option<&[u8]>;

    /// Get the current count of live bytes in the segment
    fn live_bytes(&self) -> u32;

    /// Get the current count of live items in the segment
    fn live_items(&self) -> u32;

    /// Get the data length (capacity) of this segment in bytes
    fn data_len(&self) -> usize;

    /// Get the current write offset in the segment
    fn offset(&self) -> u32;

    /// Get the current reference count for the segment
    fn ref_count(&self) -> u32;

    /// Decrement the current reference count for the segment by one
    fn decr_ref_count(&self);

    /// Get the expiration time of the segment
    fn expire_at(&self) -> clocksource::coarse::Instant;

    /// Set the expiration time for the segment
    fn set_expire_at(&self, expire_at: clocksource::coarse::Instant);

    /// Get the TTL bucket ID this segment belongs to
     fn bucket_id(&self) -> Option<u16>;

    /// Set the TTL bucket ID for this segment
     fn set_bucket_id(&self, bucket_id: u16);

    /// Clear the TTL bucket ID (mark as not in bucket)
     fn clear_bucket_id(&self);

    /// Get the current state of the segment
    fn state(&self) -> State;

    /// Attempt to transition segment from Free to Reserved state.
    ///
    /// This atomically transitions the segment state and resets all statistics
    /// (write_offset, live_items, live_bytes, expire_at) for reuse.
    ///
    /// # Returns
    /// - `true` if successfully reserved (was in Free state)
    /// - `false` if segment was not in Free state (no state change)
    ///
    /// # Panics
    /// Panics if the segment state changed during the CAS operation, which
    /// indicates a serious bug (segments in free queue should be untouched).
    fn try_reserve(&self) -> bool;

    /// Attempt to transition segment from Reserved/Linking to Free state.
    ///
    /// This atomically transitions the segment state back to Free, clearing
    /// the next/prev pointers.
    ///
    /// # Returns
    /// - `true` if successfully released
    /// - `false` if segment was already Free (idempotent)
    ///
    /// # Panics
    /// Panics if segment is in an invalid state for release (Live, Sealed, etc.)
    fn try_release(&self) -> bool;

    /// Get the next segment ID in the linked list
    fn next(&self) -> Option<u32>;

    /// Get the previous segment ID in the linked list
    fn prev(&self) -> Option<u32>;

    /// Get the merge count (number of times this segment has been a merge destination)
    fn merge_count(&self) -> u16;

    /// Increment the merge count after this segment is used as a merge destination.
    /// Saturates at u16::MAX to avoid overflow.
    fn increment_merge_count(&self);

    /// Get the generation counter for this segment.
    ///
    /// The generation is incremented each time the segment is reused (reserved from free state).
    /// Combined with (pool_id, segment_id, offset), this forms a unique CAS token that
    /// prevents the ABA problem where a key could land at the same location after eviction.
    fn generation(&self) -> u16;

    /// Increment the generation counter.
    ///
    /// Called when a segment is reserved for reuse. Wraps around at u16::MAX.
    fn increment_generation(&self);

    /// Atomically update the segment metadata with CAS
    /// Returns true if successful, false if the current value doesn't match expected
    ///
    /// # Arguments
    /// * `new_next` - New next pointer, or None to preserve current value
    /// * `new_prev` - New prev pointer, or None to preserve current value
    ///
    /// # Loom Test Coverage
    /// - `concurrent_packed_metadata_cas` - Low-level packed metadata CAS without retry logic
    /// - Called extensively by `concurrent_ttl_bucket_append` for state transitions
    fn cas_metadata(
        &self,
        expected_state: State,
        new_state: State,
        new_next: Option<u32>,
        new_prev: Option<u32>,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool;

    /// Appends an item to the segment atomically.
    ///
    /// # Safety and Synchronization
    ///
    /// This method uses lock-free CAS operations to reserve space and includes
    /// a release fence after writing data to ensure visibility. The synchronization
    /// protocol guarantees that:
    /// - Space reservation is atomic via compare_exchange on write_offset
    /// - All data writes complete before the release fence
    /// - Items are not accessible until linked into the hashtable (happens after this returns)
    /// - Concurrent readers cannot see partially written data
    ///
    /// Returns the offset where the item was written.
    /// Low-level segment append that bypasses most state validation.
    ///
    /// # Safety and State Validation
    ///
    /// This method intentionally does NOT validate segment state to allow:
    /// - Testing scenarios where segments are used directly
    /// - Emergency operations that need to bypass normal state machines
    ///
    /// For normal production usage, use `TtlBucket::append_item()` which
    /// includes proper state validation and only appends to Live segments.
    ///
    /// **Note**: This creates an intentional inconsistency with `mark_deleted()`
    /// which DOES validate state. This is by design to separate:
    /// - Low-level operations (append_item): permissive for testing/flexibility
    /// - Safety operations (mark_deleted): strict to prevent races with clear
    ///
    /// # Loom Test Coverage
    /// - `concurrent_write_offset_cas` - Low-level CAS on write_offset without retry logic
    /// - `single_item_append_metrics` - Single-threaded append with metric validation
    /// - `concurrent_item_append_to_segment` - Two threads appending to same segment
    /// - `segment_full_tracking` - Segment capacity limits and ITEM_APPEND_FULL metric
    /// - `cas_retry_tracking` (ignored) - Three threads for CAS retry validation
    fn append_item(&self, key: &[u8], value: &[u8], optional: &[u8], metrics: &crate::metrics::CacheMetrics) -> Option<u32>;

    /// Get a zero-copy guard that provides access to an item's data in the segment.
    ///
    /// This method returns an ItemGuard that holds references directly into the segment's
    /// memory, avoiding any allocations or copies. The segment's reference count is held
    /// while the guard exists, preventing eviction or clearing.
    ///
    /// # Parameters
    ///
    /// * `id` - The segment ID
    /// * `offset` - The offset within the segment
    /// * `key` - The expected key (for verification)
    ///
    /// # Returns
    ///
    /// - `Ok(ItemGuard)` - Guard providing zero-copy access to key, value, and optional data
    /// - `Err(GetItemError::ItemDeleted)` - Item is marked as deleted
    /// - `Err(GetItemError::KeyMismatch)` - Key doesn't match (hash collision)
    /// - `Err(GetItemError::SegmentNotAccessible)` - Segment is being cleared
    /// - `Err(GetItemError::InvalidOffset)` - Offset is invalid or segment ID out of bounds
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let guard = segments.get_item_guard(seg_id, offset, b"key")?;
    /// // Zero-copy access
    /// let value = guard.value();
    /// socket.write_all(value)?; // Serialize directly
    /// // Guard dropped here, ref_count decremented
    /// ```
    fn get_item_guard<'a>(
        &'a self,
        offset: u32,
        key: &[u8],
    ) -> Result<crate::item::ItemGuard<'a, Self>, GetItemError>;

    fn mark_deleted(&self, offset: u32, key: &[u8], metrics: &crate::metrics::CacheMetrics) -> Result<bool, ()>;

    /// Verify that the key at the given offset matches.
    ///
    /// This is used by the hashtable to verify that a tag match corresponds to the
    /// actual key being searched for (since tags are only 12 bits and can collide).
    ///
    /// # Parameters
    /// * `offset` - The offset within the segment
    /// * `key` - The key to compare against
    /// * `allow_deleted` - If true, matches deleted items; if false, returns false for deleted items
    ///
    /// # Returns
    /// `true` if the key matches (and item is not deleted, unless allow_deleted is true),
    /// `false` otherwise (including invalid offsets)
    fn verify_key_at_offset(&self, offset: u32, key: &[u8], allow_deleted: bool) -> bool;

    /// Compact this segment by moving all live items to the beginning.
    ///
    /// This eliminates gaps from deleted items, making space available at the end
    /// of the segment for new items or items merged from other segments.
    ///
    /// # Parameters
    /// - `hashtable`: Hashtable to update with new item locations
    /// - `metrics`: Cache metrics for tracking operations
    ///
    /// # Returns
    /// The new write_offset after compaction (where free space begins).
    ///
    /// # Safety
    /// This function performs in-place memory moves. The caller must ensure
    /// exclusive access to the segment during compaction.
    fn compact(
        &self,
        hashtable: &crate::hashtable::Hashtable,
    ) -> u32;

    /// Copy live items from this segment into a destination segment, filtered by a predicate.
    ///
    /// Used for tier migration (promotion/demotion) and merge eviction. Items matching
    /// the predicate are appended to the destination. The hashtable is updated to point
    /// to the new locations.
    ///
    /// # Parameters
    /// - `dest`: Destination segment to copy items into
    /// - `hashtable`: Hashtable to update with new item locations
    /// - `metrics`: Cache metrics for tracking operations
    /// - `predicate`: Function that takes item frequency and returns true if item should be copied
    ///
    /// # Returns
    /// Number of items successfully copied, or None if destination is full.
    ///
    /// # Examples
    /// ```ignore
    /// // Promote hot items (freq >= 10) to memory tier
    /// src.copy_into(dest, ht, metrics, |freq| freq >= 10);
    ///
    /// // Demote cold items (freq < 5) to disk tier
    /// src.copy_into(dest, ht, metrics, |freq| freq < 5);
    ///
    /// // Copy all items (equivalent to old behavior)
    /// src.copy_into(dest, ht, metrics, |_| true);
    /// ```
    ///
    /// # Note
    /// - Source items are marked as deleted after copying.
    /// - Items not in the hashtable are treated as frequency 0.
    /// - Item frequency is reset to 1 in the new location.
    /// - This method expects both segments to be in appropriate states.
    fn copy_into<S: Segment, F: Fn(u8) -> bool>(
        &self,
        dest: &S,
        hashtable: &crate::hashtable::Hashtable,
        metrics: &crate::metrics::CacheMetrics,
        predicate: F,
    ) -> Option<u32>;

    /// Prune low-frequency items from this segment during merge eviction.
    ///
    /// Walks through all items in the segment and marks items with frequency below
    /// the threshold as deleted. Items at or above the threshold are retained.
    ///
    /// # Parameters
    /// - `hashtable`: Reference to the hashtable for looking up item frequencies
    /// - `threshold`: Minimum frequency required to survive (items below are pruned)
    /// - `metrics`: Cache metrics for tracking pruned/retained counts
    ///
    /// # Returns
    /// Tuple of (items_retained, items_pruned, bytes_retained, bytes_pruned)
    ///
    /// # Note
    /// This method expects the segment to be in a state where items can be marked
    /// as deleted (Live, Sealed, or Relinking). Items already marked as deleted
    /// are skipped and counted as pruned.
    fn prune(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
    ) -> (u32, u32, u32, u32);

    /// Prune low-frequency items with a demotion callback for multi-tier caching.
    ///
    /// Similar to `prune`, but instead of just dropping pruned items, calls a callback
    /// with the item data so it can be demoted to the next tier's admission queue.
    ///
    /// # Parameters
    /// - `hashtable`: Reference to the hashtable for looking up item frequencies
    /// - `threshold`: Minimum frequency required to survive (items below are pruned)
    /// - `metrics`: Cache metrics for tracking pruned/retained counts
    /// - `on_demote`: Callback called for each pruned item with (key, value, optional).
    ///   The callback should insert the item into the next tier's small queue.
    ///   Items are unlinked from the hashtable before the callback is called.
    ///
    /// # Returns
    /// Tuple of (items_retained, items_demoted, bytes_retained, bytes_demoted)
    fn prune_with_demote<F>(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
        on_demote: F,
    ) -> (u32, u32, u32, u32)
    where
        F: FnMut(&[u8], &[u8], &[u8]); // (key, value, optional)

    /// Similar to `prune_with_demote`, but instead of calling a callback, collects
    /// items to be demoted into a Vec for later async processing.
    ///
    /// This is useful when demotion requires async operations (like appending to
    /// another tier's small queue with async mutex).
    ///
    /// # Parameters
    /// - `hashtable`: Reference to the hashtable for looking up item frequencies
    /// - `threshold`: Minimum frequency required to survive (items below are pruned)
    /// - `metrics`: Cache metrics for tracking pruned/retained counts
    ///
    /// # Returns
    /// Tuple of (items_retained, items_demoted, bytes_retained, bytes_demoted, items_to_demote)
    /// where items_to_demote is a Vec of (key, value, optional) for each pruned item.
    fn prune_collecting_for_demote(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
    ) -> (u32, u32, u32, u32, Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>);

    /// Unlink all items in this segment from the hashtable.
    ///
    /// This is used during FIFO eviction to ensure no stale hashtable entries
    /// remain after the segment is freed and potentially reused. Without this,
    /// reads for evicted items would hit garbage data in the reused segment.
    ///
    /// # Parameters
    /// - `hashtable`: Reference to the hashtable for unlinking items
    /// - `metrics`: Cache metrics for tracking unlink operations
    /// - `create_ghosts`: If true, convert entries to ghost entries to preserve
    ///   frequency history. If false (for TTL expiration), just remove entries.
    ///
    /// # Returns
    /// Number of items unlinked from the hashtable.
    ///
    /// # Note
    /// Already-deleted items are skipped.
    fn unlink_all_items(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        metrics: &crate::metrics::CacheMetrics,
        create_ghosts: bool,
    ) -> u32;

    /// Append an item to a small queue segment with per-item TTL.
    ///
    /// This is similar to `append_item` but uses `SmallQueueItemHeader` which
    /// includes an expiration timestamp for per-item TTL tracking.
    ///
    /// # Parameters
    /// - `key`: Item key
    /// - `value`: Item value
    /// - `optional`: Optional metadata
    /// - `expire_at`: Expiration time as coarse seconds (absolute timestamp)
    /// - `metrics`: Cache metrics
    ///
    /// # Returns
    /// The offset where the item was written, or None if segment is full.
    fn append_small_queue_item(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<u32>;
}

/// State of a segment in its lifecycle
///
/// # State Semantics
///
/// - **Free**: In free queue, not in use
/// - **Reserved**: Allocated for use, being prepared for chain insertion
/// - **Linking**: Being added to TTL bucket chain (next/prev being set)
/// - **Live**: Active tail segment accepting writes and reads
/// - **Sealed**: No more writes accepted, but data readable and chain stable
/// - **Relinking**: Chain pointers being updated during neighbor removal.
///   Data remains readable, only next/prev pointers are being modified.
///   This state allows safe updates to the doubly-linked list structure
///   without blocking read access to segment data.
/// - **Draining**: Segment is being processed (merge eviction or removal).
///   Provides exclusive access - only one thread can hold a segment in Draining.
///   New reads are rejected; must wait for ref_count to drop before modifying data.
///   After processing: CAS back to Sealed (if survives) or continue to Locked (if removed).
/// - **Locked**: Being cleared, all access rejected
///
/// # Relinking State and Chain Update Protocol
///
/// When removing a middle segment (B) from a chain A <-> B <-> C:
/// 1. Lock target B: Sealed -> Draining (wait for readers) -> Locked
/// 2. Lock prev segment A: Sealed -> Relinking (A cannot be Live since B exists after it)
/// 3. Update A's next pointer to point to C
/// 4. Unlock A: Relinking -> Sealed
/// 5. Lock next segment C: Sealed | Live -> Relinking
/// 6. Update C's prev pointer to point to A
/// 7. Unlock C: Relinking -> (original state)
/// 8. Update bucket head/tail if needed
/// 9. Clear B and release
///
/// The Relinking state provides mutual exclusion for chain pointer updates
/// while still allowing reads of segment data.
///
/// # Future: Migration to Bitflags
///
/// Consider migrating from enum to bitflags for more flexible combinations:
/// ```ignore
/// bitflags! {
///     struct SegmentFlags: u8 {
///         const SEALED = 1 << 0;       // No more writes accepted
///         const METADATA_LOCK = 1 << 1; // Chain pointers locked (Relinking)
///         const DATA_LOCK = 1 << 2;     // Data being cleared (Draining/Locked)
///         const LIVE = 1 << 3;          // Accepting writes (tail)
///     }
/// }
/// ```
/// This would allow more granular state combinations and easier reasoning
/// about which operations are permitted in each state.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Free = 0,
    Reserved = 1,
    Linking = 2,  // Intermediate state while linking into chain
    Live = 3,
    Sealed = 4,
    Relinking = 5, // Chain pointers being updated (data still readable)
    Draining = 6,  // Being processed/evicted (exclusive access, reads rejected)
    Locked = 7,    // Being cleared (all access rejected)
}

/// Packed representation of segment metadata in a single AtomicU64
/// Layout: [8 bits unused] [8 bits state] [24 bits prev] [24 bits next]
pub struct Metadata {
    pub next: u32,  // Only uses 24 bits
    pub prev: u32,  // Only uses 24 bits
    pub state: State,
}

pub const INVALID_SEGMENT_ID: u32 = 0xFFFFFF; // 24-bit max value

impl Metadata {
    pub fn pack(self) -> u64 {
        // Mask to ensure we only use 24 bits for IDs
        let next_24 = (self.next & 0xFFFFFF) as u64;
        let prev_24 = (self.prev & 0xFFFFFF) as u64;
        let state_8 = self.state as u64;

        // Pack: [8 unused][8 state][24 prev][24 next]
        (state_8 << 48) | (prev_24 << 24) | next_24
    }

    pub fn unpack(packed: u64) -> Self {
        let state_val = ((packed >> 48) & 0xFF) as u8;
        let state = match state_val {
            0 => State::Free,
            1 => State::Reserved,
            2 => State::Linking,
            3 => State::Live,
            4 => State::Sealed,
            5 => State::Relinking,
            6 => State::Draining,
            7 => State::Locked,
            _ => {
                panic!("Corrupted segment metadata: invalid state {}", state_val);
            }
        };

        Self {
            next: (packed & 0xFFFFFF) as u32,
            prev: ((packed >> 24) & 0xFFFFFF) as u32,
            state,
        }
    }
}
