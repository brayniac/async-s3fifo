//! Small Queue (Admission Queue) for S3-FIFO cache policy
//!
//! The small queue acts as an admission filter to prevent one-hit-wonders from
//! polluting the main cache. Items are inserted here first and must be accessed
//! again (frequency > 0) to be promoted to the main cache.
//!
//! # Design
//!
//! - FIFO eviction order (no TTL bucket organization)
//! - Items have per-item TTL stored in their header (SmallQueueItemHeader)
//! - On eviction: items with freq > 0 are promoted to main cache, others are dropped
//! - Cascading: items evicted from RAM small queue may go to SSD small queue
//!
//! # Segment Chain
//!
//! Segments form a singly-linked FIFO chain:
//! ```text
//! head (oldest) -> seg1 -> seg2 -> ... -> tail (newest, accepting writes)
//! ```

use crate::pool::Pool;
use crate::segment::{Segment, State, INVALID_SEGMENT_ID};
use crate::sync::AtomicU64;
use std::sync::atomic::Ordering;

/// A FIFO admission queue using a chain of segments
pub struct SmallQueue {
    /// Packed head and tail segment IDs
    /// Layout: [32 bits head][32 bits tail]
    /// Uses INVALID_SEGMENT_ID (0xFFFFFF) to indicate empty
    head_tail: AtomicU64,

    /// Async mutex for serializing chain manipulation operations.
    /// This eliminates CAS retry loops and allows other tasks to yield while waiting.
    write_lock: tokio::sync::Mutex<()>,
}

impl SmallQueue {
    /// Sentinel value for empty queue (both head and tail invalid)
    const EMPTY: u64 = Self::pack(INVALID_SEGMENT_ID, INVALID_SEGMENT_ID);

    pub fn new() -> Self {
        Self {
            head_tail: AtomicU64::new(Self::EMPTY),
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Pack head and tail into a single u64
    const fn pack(head: u32, tail: u32) -> u64 {
        ((head as u64) << 32) | (tail as u64)
    }

    /// Unpack head and tail from a u64
    fn unpack(packed: u64) -> (u32, u32) {
        let head = (packed >> 32) as u32;
        let tail = packed as u32;
        (head, tail)
    }

    /// Get the current head segment ID (oldest segment)
    #[cfg(test)]
    pub fn head(&self) -> Option<u32> {
        let (head, _) = Self::unpack(self.head_tail.load(Ordering::Acquire));
        if head == INVALID_SEGMENT_ID {
            None
        } else {
            Some(head)
        }
    }

    /// Get the current tail segment ID (newest segment, accepting writes)
    pub fn tail(&self) -> Option<u32> {
        let (_, tail) = Self::unpack(self.head_tail.load(Ordering::Acquire));
        if tail == INVALID_SEGMENT_ID {
            None
        } else {
            Some(tail)
        }
    }

    /// Check if the queue is empty
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.head_tail.load(Ordering::Acquire) == Self::EMPTY
    }

    /// Append a new segment to the tail of the queue
    ///
    /// This acquires the write lock to serialize chain operations, eliminating
    /// CAS contention and allowing other tasks to yield while waiting.
    ///
    /// # Returns
    /// - `Ok(())` if successful
    /// - `Err(())` if segment is in wrong state or other error
    pub async fn append_segment<P: Pool>(
        &self,
        segment_id: u32,
        pool: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), ()> {
        let _guard = self.write_lock.lock().await;

        // With mutex held, we have exclusive access - no CAS retries needed
        let segment = pool.get(segment_id).ok_or(())?;

        if segment.state() != State::Reserved {
            return Err(());
        }

        let current = self.head_tail.load(Ordering::Acquire);
        let (head, tail) = Self::unpack(current);

        if tail == INVALID_SEGMENT_ID {
            // Empty queue - this segment becomes both head and tail
            if !segment.cas_metadata(
                State::Reserved,
                State::Linking,
                Some(INVALID_SEGMENT_ID),
                Some(INVALID_SEGMENT_ID),
                metrics,
            ) {
                return Err(());
            }

            let new = Self::pack(segment_id, segment_id);
            self.head_tail.store(new, Ordering::Release);

            segment.cas_metadata(State::Linking, State::Live, None, None, metrics);
            Ok(())
        } else {
            // Non-empty queue - append after current tail
            let old_tail = pool.get(tail).ok_or(())?;

            // Seal the old tail first
            if old_tail.state() == State::Live {
                old_tail.cas_metadata(State::Live, State::Sealed, None, None, metrics);
            }

            // Set up the new segment
            if !segment.cas_metadata(
                State::Reserved,
                State::Linking,
                Some(INVALID_SEGMENT_ID),
                Some(tail),
                metrics,
            ) {
                return Err(());
            }

            // Update old tail's next pointer
            if !old_tail.cas_metadata(
                State::Sealed,
                State::Sealed,
                Some(segment_id),
                None,
                metrics,
            ) {
                segment.cas_metadata(State::Linking, State::Reserved, None, None, metrics);
                return Err(());
            }

            // Update queue tail
            let new = Self::pack(head, segment_id);
            self.head_tail.store(new, Ordering::Release);

            segment.cas_metadata(State::Linking, State::Live, None, None, metrics);
            Ok(())
        }
    }

    /// Remove and return the head segment for eviction
    ///
    /// This acquires the write lock to serialize eviction operations.
    /// Other tasks can yield while waiting for the lock.
    ///
    /// # Returns
    /// - `Some(segment_id)` of the evicted head segment
    /// - `None` if queue is empty
    pub async fn evict_head<P: Pool>(
        &self,
        pool: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<u32> {
        let _guard = self.write_lock.lock().await;

        let current = self.head_tail.load(Ordering::Acquire);
        let (head, tail) = Self::unpack(current);

        if head == INVALID_SEGMENT_ID {
            return None; // Empty queue
        }

        let head_segment = pool.get(head)?;
        let next = head_segment.next();

        // Transition head to Draining
        let current_state = head_segment.state();
        if current_state != State::Sealed && current_state != State::Live {
            return None;
        }

        if !head_segment.cas_metadata(current_state, State::Draining, None, None, metrics) {
            return None;
        }

        // Update queue head
        let new_head = next.unwrap_or(INVALID_SEGMENT_ID);
        let new_tail = if new_head == INVALID_SEGMENT_ID {
            INVALID_SEGMENT_ID
        } else {
            tail
        };

        let new = Self::pack(new_head, new_tail);
        self.head_tail.store(new, Ordering::Release);

        // Transition to Locked for clearing
        head_segment.cas_metadata(State::Draining, State::Locked, None, None, metrics);
        Some(head)
    }
}

impl Default for SmallQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack() {
        let head = 0x123456;
        let tail = 0x789ABC;
        let packed = SmallQueue::pack(head, tail);
        let (h, t) = SmallQueue::unpack(packed);
        assert_eq!(h, head);
        assert_eq!(t, tail);
    }

    #[test]
    fn test_empty_queue() {
        let queue = SmallQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.head(), None);
        assert_eq!(queue.tail(), None);
    }
}
