use crate::segment::Metadata;
use crate::segment::State;
use crate::GetItemError;
use crate::item::ItemHeader;
use crate::segment::Segment;
use crate::util::*;
use std::ptr::NonNull;
use clocksource::coarse::AtomicInstant;
use crate::segment::INVALID_SEGMENT_ID;
use crate::sync::*;

#[repr(C, align(64))]
pub struct SliceSegment<'a> {
    // Packed metadata: next pointer, prev pointer, and state in single AtomicU64
    // This enables atomic updates of all three fields together
    pub(crate) metadata: AtomicU64,

    // Hot path: frequently accessed during append operations
    // Keep these in the first cache line for better performance
    pub(crate) write_offset: AtomicU32,

    pub(crate) live_items: AtomicU32,
    pub(crate) live_bytes: AtomicU32,
    pub(crate) ref_count: AtomicU32, // Reference count for live readers

    pub(crate) id: u32,
    pub(crate) data_len: u32,
    data: NonNull<u8>,

    // Cold path: expiration tracking
    // Rarely accessed except during TTL management
    pub(crate) expire_at: AtomicInstant,
    pub(crate) bucket_id: AtomicU16, // TTL bucket ID (0xFFFF = not in bucket)

    // Packed: [2 bits pool_id][1 bit is_small_queue][5 bits reserved]
    pub(crate) pool_id: u8,

    // Number of times this segment has been a merge destination
    // Stored separately (not packed) to avoid 8-bit saturation issues
    pub(crate) merge_count: AtomicU16,

    // Generation counter for CAS tokens - incremented each time segment is reused.
    // Combined with (pool_id, segment_id, offset) to form unique CAS tokens.
    // 16 bits = 65536 generations before wrap, sufficient to prevent ABA issues.
    pub(crate) generation: AtomicU16,

    _lifetime: std::marker::PhantomData<&'a u8>,
}

// SAFETY: Segment synchronizes access via atomics
unsafe impl<'a> Send for SliceSegment<'a> {}
unsafe impl<'a> Sync for SliceSegment<'a> {}

impl<'a> SliceSegment<'a> {
    /// Sentinel value indicating segment is not in a TTL bucket
    const INVALID_BUCKET_ID: u16 = 0xFFFF;

    /// Mask for pool_id bits (bits 0-1)
    const POOL_ID_MASK: u8 = 0x03;
    /// Bit flag for small queue segments (bit 2)
    const SMALL_QUEUE_BIT: u8 = 0x04;

    /// Create a new segment from a data slice
    /// SAFETY: data must outlive the segment
    pub(crate) unsafe fn new(pool_id: u8, is_small_queue: bool, id: u32, data: *mut u8, len: usize) -> Self {
        debug_assert!(pool_id <= 3, "pool_id {} exceeds 2-bit limit", pool_id);
        let pool_id_packed = (pool_id & Self::POOL_ID_MASK)
            | if is_small_queue { Self::SMALL_QUEUE_BIT } else { 0 };
        // Initially, segment is Free with no links
        let initial_meta = Metadata {
            next: INVALID_SEGMENT_ID,
            prev: INVALID_SEGMENT_ID,
            state: State::Free,
        };

        Self {
            // Hot path atomics
            metadata: AtomicU64::new(initial_meta.pack()),
            write_offset: AtomicU32::new(0),
            live_items: AtomicU32::new(0),
            live_bytes: AtomicU32::new(0),
            ref_count: AtomicU32::new(0),

            id,
            data_len: len as u32,
            data: unsafe { NonNull::new_unchecked(data) },

            // Cold path
            expire_at: AtomicInstant::now(),
            bucket_id: AtomicU16::new(Self::INVALID_BUCKET_ID),
            pool_id: pool_id_packed,
            merge_count: AtomicU16::new(0),
            generation: AtomicU16::new(0),

            _lifetime: std::marker::PhantomData,
        }
    }

    /// Returns true if this segment is part of the small queue (admission queue)
    pub fn is_small_queue(&self) -> bool {
        self.pool_id & Self::SMALL_QUEUE_BIT != 0
    }
}

impl Segment for SliceSegment<'_> {
    fn id(&self) -> u32 {
        self.id
    }
    fn pool_id(&self) -> u8 {
        self.pool_id & Self::POOL_ID_MASK
    }
    fn is_small_queue(&self) -> bool {
        self.pool_id & Self::SMALL_QUEUE_BIT != 0
    }
    fn data_slice(&self, offset: u32, len: usize) -> Option<&[u8]> {
        let end = offset as usize + len;
        if end > self.data_len as usize {
            return None;
        }
        // SAFETY: We've validated that offset + len is within bounds
        Some(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr().add(offset as usize), len)
        })
    }
    fn live_bytes(&self) -> u32 {
        self.live_bytes.load(Ordering::Relaxed)
    }
    fn live_items(&self) -> u32 {
        self.live_items.load(Ordering::Relaxed)
    }
    fn data_len(&self) -> usize {
        self.data_len as usize
    }
    fn offset(&self) -> u32 {
        self.write_offset.load(Ordering::Relaxed)
    }
    fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }
    fn decr_ref_count(&self) {
        self.ref_count.fetch_sub(1, Ordering::AcqRel);
    }
    fn expire_at(&self) -> clocksource::coarse::Instant {
        self.expire_at.load(Ordering::Acquire)
    }
    fn set_expire_at(&self, expire_at: clocksource::coarse::Instant) {
        self.expire_at.store(expire_at, Ordering::Relaxed);
    }
    fn bucket_id(&self) -> Option<u16> {
        let id = self.bucket_id.load(Ordering::Acquire);
        if id == Self::INVALID_BUCKET_ID {
            None
        } else {
            Some(id)
        }
    }
    fn set_bucket_id(&self, bucket_id: u16) {
        self.bucket_id.store(bucket_id, Ordering::Release);
    }
    fn clear_bucket_id(&self) {
        self.bucket_id.store(Self::INVALID_BUCKET_ID, Ordering::Release);
    }
    fn state(&self) -> State {
        let packed = self.metadata.load(Ordering::Acquire);
        Metadata::unpack(packed).state
    }

    fn try_reserve(&self) -> bool {
        let current_packed = self.metadata.load(Ordering::Acquire);
        let current_meta = Metadata::unpack(current_packed);

        // Only reserve if currently Free
        if current_meta.state != State::Free {
            return false;
        }

        // Prepare new state: Reserved with no links
        let new_meta = Metadata {
            next: INVALID_SEGMENT_ID,
            prev: INVALID_SEGMENT_ID,
            state: State::Reserved,
        };

        match self.metadata.compare_exchange(
            current_packed,
            new_meta.pack(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Successfully reserved - reset segment statistics
                self.write_offset.store(0, Ordering::Relaxed);
                self.live_items.store(0, Ordering::Relaxed);
                self.live_bytes.store(0, Ordering::Relaxed);
                self.expire_at.store(clocksource::coarse::Instant::now(), Ordering::Relaxed);
                self.merge_count.store(0, Ordering::Relaxed);
                // Increment generation for CAS token uniqueness (wrapping is fine)
                self.generation.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                // State changed during CAS - serious bug if segment was in free queue
                panic!(
                    "Segment {} state changed during reservation (loaded state: {:?}, expected Free)",
                    self.id, current_meta.state
                );
            }
        }
    }

    fn try_release(&self) -> bool {
        loop {
            let current_packed = self.metadata.load(Ordering::Acquire);
            let current_meta = Metadata::unpack(current_packed);

            match current_meta.state {
                State::Reserved | State::Linking | State::Locked => {
                    // Valid states for release:
                    // - Reserved: segment was allocated but never used
                    // - Linking: segment was being linked but operation failed
                    // - Locked: segment was evicted and is ready for reuse
                }
                State::Free => {
                    // Already Free - idempotent success
                    return false;
                }
                _ => {
                    // Invalid state for release
                    panic!(
                        "Attempt to release segment {} in invalid state {:?}",
                        self.id, current_meta.state
                    );
                }
            }

            let new_meta = Metadata {
                next: INVALID_SEGMENT_ID,
                prev: INVALID_SEGMENT_ID,
                state: State::Free,
            };

            match self.metadata.compare_exchange(
                current_packed,
                new_meta.pack(),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Reset merge_count when released
                    self.merge_count.store(0, Ordering::Relaxed);
                    return true;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    fn next(&self) -> Option<u32> {
        let packed = self.metadata.load(Ordering::Acquire);
        let metadata = Metadata::unpack(packed);
        if metadata.next == INVALID_SEGMENT_ID {
            None
        } else {
            Some(metadata.next)
        }
    }
    fn prev(&self) -> Option<u32> {
        let packed = self.metadata.load(Ordering::Acquire);
        let metadata = Metadata::unpack(packed);
        if metadata.prev == INVALID_SEGMENT_ID {
            None
        } else {
            Some(metadata.prev)
        }
    }

    fn merge_count(&self) -> u16 {
        self.merge_count.load(Ordering::Acquire)
    }

    fn increment_merge_count(&self) {
        // Use saturating arithmetic via CAS loop to avoid overflow
        loop {
            let current = self.merge_count.load(Ordering::Acquire);
            if current == u16::MAX {
                // Already saturated, nothing to do
                return;
            }
            let new_count = current.saturating_add(1);
            if self.merge_count.compare_exchange(
                current,
                new_count,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                return;
            }
            // CAS failed, retry
        }
    }

    fn generation(&self) -> u16 {
        self.generation.load(Ordering::Acquire)
    }

    fn increment_generation(&self) {
        // Wrapping add is fine - we just need uniqueness over short time periods
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    fn cas_metadata(
        &self,
        expected_state: State,
        new_state: State,
        new_next: Option<u32>,
        new_prev: Option<u32>,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool {
        let result = retry_cas_metadata(
            &self.metadata,
            expected_state,
            |current_meta| {
                // Create new metadata with updated fields
                // If None is passed, preserve the current value
                Some(Metadata {
                    next: new_next.unwrap_or(current_meta.next),
                    prev: new_prev.unwrap_or(current_meta.prev),
                    state: new_state,
                })
            },
            CasRetryConfig::default(),
            metrics,
        );

        let success = matches!(result, CasResult::Success(()));

        // Update state gauges on successful transition
        if success && expected_state != new_state {
            // Decrement old state gauge
            match expected_state {
                State::Live => { metrics.segments_live.decrement(); }
                State::Sealed => { metrics.segments_sealed.decrement(); }
                _ => {}
            }

            // Increment new state gauge
            match new_state {
                State::Live => { metrics.segments_live.increment(); }
                State::Sealed => { metrics.segments_sealed.increment(); }
                _ => {}
            }
        }

        success
    }
    fn append_item(&self, key: &[u8], value: &[u8], optional: &[u8], metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        // Validate segment data structures are not corrupted
        assert!(self.data_len > 0,
            "CORRUPTION: segment {} has data_len=0", self.id);

        if key.is_empty() || key.len() > ItemHeader::MAX_KEY_LEN {
            panic!(
                "key size is out of range: must be 1-{} bytes",
                ItemHeader::MAX_KEY_LEN
            );
        }

        if optional.len() > ItemHeader::MAX_OPTIONAL_LEN {
            panic!(
                "optional size is out of range: must be 0-{} bytes",
                ItemHeader::MAX_OPTIONAL_LEN
            );
        }

        if value.len() > ItemHeader::MAX_VALUE_LEN {
            panic!(
                "value size is out of range: must be 0-{} bytes",
                ItemHeader::MAX_VALUE_LEN
            );
        }

        let header = ItemHeader::new(
            key.len() as u8,
            optional.len() as u8,
            value.len() as u32,
            false, // is_deleted
            false, // is_numeric
        );

        if header.padded_size() as u32 > self.data_len {
            panic!("item size is out of range. increase segment size");
        }

        let item_size = header.padded_size() as u32;

        // Use the standard CAS retry pattern for reserving space
        let reserved_offset = match retry_cas_u32(
            &self.write_offset,
            |current_offset| {
                let new_offset = current_offset.saturating_add(item_size);

                // Check if there's enough space
                if new_offset > self.data_len {
                    return None; // Segment is full
                }

                Some((new_offset, current_offset))
            },
            CasRetryConfig {
                max_attempts: 16,
                early_spin_threshold: 4,
            },
            metrics,
        ) {
            CasResult::Success(offset) => offset,
            CasResult::Failed(_) | CasResult::Aborted => {
                // Segment is full, return None
                metrics.item_append_full.increment();
                return None;
            }
        };

        // Space successfully reserved, now write the data
        {
            // CRITICAL: Check reserved_offset BEFORE any pointer arithmetic
            // A corrupted offset could cause segfault in pointer addition itself
            if reserved_offset >= self.data_len {
                panic!(
                    "CORRUPTION: reserved_offset ({}) >= data_len ({}) in segment {}. \
                     write_offset was: {}. This should be impossible after CAS!",
                    reserved_offset, self.data_len, self.id,
                    self.write_offset.load(Ordering::Relaxed)
                );
            }

            // Check if data pointer looks valid (not null, not clearly corrupted)
            let base_ptr = self.data.as_ptr() as usize;
            if base_ptr == 0 {
                panic!("CORRUPTION: segment {} has null data pointer", self.id);
            }
            // Check if pointer is in a reasonable range (not 0xFFFF... or very low addresses)
            if base_ptr < 0x1000 || base_ptr == usize::MAX {
                panic!("CORRUPTION: segment {} has invalid data pointer: {:p}", self.id, self.data.as_ptr());
            }

            // Runtime assertions to catch corruption early (always enabled for safety)
            // These have minimal performance impact but will catch serious bugs
            assert!(reserved_offset < self.data_len,
                "CORRUPTION: reserved_offset ({}) >= data_len ({}) in segment {}",
                reserved_offset, self.data_len, self.id);
            assert!(reserved_offset.saturating_add(item_size) <= self.data_len,
                "CORRUPTION: reserved_offset ({}) + item_size ({}) > data_len ({}) in segment {}",
                reserved_offset, item_size, self.data_len, self.id);

            let mut data_ptr = unsafe { self.data.as_ptr().add(reserved_offset as usize) };

            // Validate pointer is within segment bounds before any writes
            let segment_end = unsafe { self.data.as_ptr().add(self.data_len as usize) };
            let write_end = unsafe { data_ptr.add(item_size as usize) };
            assert!(write_end <= segment_end,
                "CORRUPTION: write would extend past segment end. data_ptr offset={}, item_size={}, data_len={}, segment={}",
                reserved_offset, item_size, self.data_len, self.id);

            {
                let data =
                    unsafe { std::slice::from_raw_parts_mut(data_ptr, ItemHeader::SIZE) };
                header.to_bytes(data);
            }

            unsafe {
                data_ptr = data_ptr.add(ItemHeader::SIZE);

                // Copy optional metadata (usually small)
                if !optional.is_empty() {
                    assert!(data_ptr.add(optional.len()) <= segment_end,
                        "CORRUPTION: optional write would exceed segment bounds in segment {}", self.id);
                    std::ptr::copy_nonoverlapping(optional.as_ptr(), data_ptr, optional.len());
                    data_ptr = data_ptr.add(optional.len());
                }

                // Copy key (usually small)
                assert!(data_ptr.add(key.len()) <= segment_end,
                    "CORRUPTION: key write would exceed segment bounds in segment {}", self.id);
                std::ptr::copy_nonoverlapping(key.as_ptr(), data_ptr, key.len());
                data_ptr = data_ptr.add(key.len());

                // Copy value
                if !value.is_empty() {
                    assert!(data_ptr.add(value.len()) <= segment_end,
                        "CORRUPTION: value write (len={}) would exceed segment bounds in segment {}", value.len(), self.id);
                    // Use regular copy for all values (non-temporal stores disabled due to alignment issues)
                    std::ptr::copy_nonoverlapping(value.as_ptr(), data_ptr, value.len());
                }
            }

            // Ensure all writes are visible before returning offset
            // The item won't be accessible until linked into hashtable
            fence(Ordering::Release);

            // Update segment statistics
            self.live_items.fetch_add(1, Ordering::Relaxed);
            self.live_bytes.fetch_add(item_size, Ordering::Relaxed);

            metrics.item_append.increment();
            // Update cache-wide item tracking
            metrics.items_live.increment();
            metrics.bytes_live.add(item_size as i64);
            Some(reserved_offset)
        }
    }
    fn get_item_guard<'a>(
        &'a self,
        offset: u32,
        key: &[u8],
    ) -> Result<crate::item::ItemGuard<'a, Self>, GetItemError> {
        use crate::item::SmallQueueItemHeader;
        use clocksource::coarse::UnixInstant;
        use std::sync::atomic::Ordering;
        use std::sync::atomic::fence;

        // Atomically increment reference count only if segment is accessible
        {
            let state = self.state();
            if state == State::Draining || state == State::Locked {
                return Err(GetItemError::SegmentNotAccessible);
            }

            // Increment reference count
            self.ref_count.fetch_add(1, Ordering::Acquire);

            // Double-check state after increment to handle race
            let state_after = self.state();
            if state_after == State::Draining || state_after == State::Locked {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::SegmentNotAccessible);
            }

            // Successfully acquired reference
            fence(Ordering::Acquire);
        }

        let data_ptr = unsafe { self.data.as_ptr().add(offset as usize) };

        // Use appropriate header type based on segment type
        if self.is_small_queue() {
            // Small queue segment with per-item TTL
            // Validate offset allows for at least a small queue header
            if offset.saturating_add(SmallQueueItemHeader::SIZE as u32) > self.data_len {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::InvalidOffset);
            }

            // Read and validate header
            let header = match SmallQueueItemHeader::try_from_bytes(unsafe {
                std::slice::from_raw_parts(data_ptr, SmallQueueItemHeader::SIZE)
            }) {
                Some(h) => h,
                None => {
                    self.ref_count.fetch_sub(1, Ordering::Release);
                    return Err(GetItemError::InvalidOffset);
                }
            };

            // Check if item is deleted
            if header.is_deleted() {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::ItemDeleted);
            }

            // Check expiration
            let now = UnixInstant::now().duration_since(UnixInstant::EPOCH).as_secs();
            if header.is_expired(now) {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::ItemDeleted);
            }

            // Validate that full item fits within segment
            let item_size = header.padded_size() as u32;
            if offset.saturating_add(item_size) > self.data_len {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::InvalidOffset);
            }

            // Get raw slice for the entire item
            let raw = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };

            // Calculate ranges
            let optional_start = SmallQueueItemHeader::SIZE;
            let optional_end = optional_start + header.optional_len() as usize;
            let key_start = optional_end;
            let key_end = key_start + header.key_len() as usize;
            let value_start = key_end;
            let value_end = value_start + header.value_len() as usize;

            // Verify key matches
            if &raw[key_start..key_end] != key {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::KeyMismatch);
            }

            // Create guard with slices into segment data
            Ok(crate::item::ItemGuard::new(
                self,
                &raw[key_start..key_end],
                &raw[value_start..value_end],
                &raw[optional_start..optional_end],
            ))
        } else {
            // Main cache segment with standard header
            // Check segment-level expiration first
            let now = clocksource::coarse::Instant::now();
            if self.expire_at() < now {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::ItemDeleted);
            }

            // Validate offset
            if offset.saturating_add(ItemHeader::MIN_ITEM_SIZE as u32) > self.data_len {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::InvalidOffset);
            }

            // Read and validate header
            // Use try_from_bytes to handle races where segment is recycled between
            // verify_key_at_offset and get_item_guard (TOCTOU race)
            let header = match ItemHeader::try_from_bytes(unsafe {
                std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE)
            }) {
                Some(h) => h,
                None => {
                    self.ref_count.fetch_sub(1, Ordering::Release);
                    return Err(GetItemError::InvalidOffset);
                }
            };

            // Check if item is deleted
            if header.is_deleted() {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::ItemDeleted);
            }

            // Validate that full item fits within segment
            let item_size = header.padded_size() as u32;
            if offset.saturating_add(item_size) > self.data_len {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::InvalidOffset);
            }

            // Get raw slice for the entire item
            let raw = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };

            // Calculate ranges
            let optional_start = ItemHeader::SIZE;
            let optional_end = optional_start + header.optional_len() as usize;
            let key_start = optional_end;
            let key_end = key_start + header.key_len() as usize;
            let value_start = key_end;
            let value_end = value_start + header.value_len() as usize;

            // Verify key matches
            if &raw[key_start..key_end] != key {
                self.ref_count.fetch_sub(1, Ordering::Release);
                return Err(GetItemError::KeyMismatch);
            }

            // Create guard with slices into segment data
            // The guard will decrement ref_count on drop
            Ok(crate::item::ItemGuard::new(
                self,
                &raw[key_start..key_end],
                &raw[value_start..value_end],
                &raw[optional_start..optional_end],
            ))
        }
    }

    fn mark_deleted(&self, offset: u32, key: &[u8], metrics: &crate::metrics::CacheMetrics) -> Result<bool, ()> {
        // Check segment state first
        let current_state = self.state();
        match current_state {
            State::Free
            | State::Reserved
            | State::Linking
            | State::Live
            | State::Sealed
            | State::Relinking => {
                // OK to mark items deleted in these states
                // Free: segment not yet reserved but can have items (testing/direct use)
                // Reserved: segment has items but not yet in TTL bucket
                // Linking: segment being added to TTL bucket
                // Live: normal active segment
                // Sealed: full segment, no new appends but deletions still valid
                // Relinking: chain pointers locked but data still accessible
            }
            State::Draining | State::Locked => {
                // Segment is being cleared - don't interfere, treat as "already deleted"
                return Ok(false);
            }
        }

        // Validate that we have at least enough space for the header
        if offset.saturating_add(ItemHeader::MIN_ITEM_SIZE as u32) > self.data_len {
            // Invalid offset - treat as "already handled"
            return Ok(false);
        }

        let data_ptr = unsafe { self.data.as_ptr().add(offset as usize) };

        // Read header to validate item and get size
        // Use try_from_bytes to handle races where segment is recycled between
        // hashtable lookup and mark_deleted (TOCTOU race)
        let header = match ItemHeader::try_from_bytes(unsafe {
            std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE)
        }) {
            Some(h) => h,
            None => {
                // Header validation failed - segment likely recycled, treat as "already handled"
                return Ok(false);
            }
        };

        // Check if already deleted before doing more work
        if header.is_deleted() {
            return Ok(false);
        }

        // Validate that the full item fits within the segment
        let item_size = header.padded_size() as u32;
        if offset.saturating_add(item_size) > self.data_len {
            // Invalid item size - treat as "already handled"
            return Ok(false);
        }

        // Read the item to verify key match
        let raw = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };

        // Calculate key range relative to the start of this item (offset 0 within raw slice)
        let key_start = ItemHeader::SIZE + header.optional_len() as usize;
        let key_end = key_start + header.key_len() as usize;

        if &raw[key_start..key_end] != key {
            // Key mismatch - collision case that caller needs to handle
            return Err(());
        }

        // SAFETY: Atomically set the deleted bit in the flags byte.
        // The flags byte is at offset 4 in the packed header struct.
        let flags_ptr = unsafe { data_ptr.add(4) };

        #[cfg(not(feature = "loom"))]
        let old_flags = {
            // In production, we can use AtomicU8 directly (naturally aligned)
            let flags_atomic = unsafe { &*(flags_ptr as *const AtomicU8) };
            flags_atomic.fetch_or(0x40, Ordering::Release)
        };

        #[cfg(feature = "loom")]
        let old_flags = {
            // In loom, AtomicU8 requires 8-byte alignment which we don't have at offset 4.
            // Use a CAS loop with volatile operations. Loom tracks data races even through
            // volatile operations, so this is safe for testing.
            let mut old_val = unsafe { std::ptr::read_volatile(flags_ptr) };
            loop {
                // Check if already deleted
                if (old_val & 0x40) != 0 {
                    break old_val;
                }

                let new_val = old_val | 0x40;

                // Atomic fence before the write
                fence(Ordering::Release);

                // Try to write the new value
                // In loom, this volatile write will be tracked for races
                unsafe {
                    let current = std::ptr::read_volatile(flags_ptr);
                    if current == old_val {
                        std::ptr::write_volatile(flags_ptr, new_val);
                        break old_val;
                    }
                    old_val = current;
                }
            }
        };

        // Check if the item was already deleted
        if (old_flags & 0x40) != 0 {
            // Another thread already deleted it
            return Ok(false);
        }

        // Update segment statistics to reflect the deletion
        // fetch_sub returns the OLD value before decrement
        self.live_items.fetch_sub(1, Ordering::Relaxed);
        self.live_bytes.fetch_sub(item_size, Ordering::Relaxed);

        // Update cache-wide item tracking
        metrics.items_live.decrement();
        metrics.bytes_live.sub(item_size as i64);

        Ok(true)
    }

    fn verify_key_at_offset(&self, offset: u32, key: &[u8], allow_deleted: bool) -> bool {
        let data_ptr = unsafe { self.data.as_ptr().add(offset as usize) };

        // Use appropriate header type based on segment type
        if self.is_small_queue() {
            use crate::item::SmallQueueItemHeader;

            // Validate offset allows for at least a small queue header
            if offset.saturating_add(SmallQueueItemHeader::SIZE as u32) > self.data_len {
                return false;
            }

            // Try to parse the small queue header
            let header = match SmallQueueItemHeader::try_from_bytes(unsafe {
                std::slice::from_raw_parts(data_ptr, SmallQueueItemHeader::SIZE)
            }) {
                Some(h) => h,
                None => return false,
            };

            // Check if item is deleted (unless we allow deleted items)
            if !allow_deleted && header.is_deleted() {
                return false;
            }

            // Check if item is expired (small queue uses per-item TTL)
            if !allow_deleted {
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as u32;
                if header.expire_at() < now_secs {
                    return false;
                }
            }

            // Validate that full item fits within segment
            let item_size = header.padded_size() as u32;
            if offset.saturating_add(item_size) > self.data_len {
                return false;
            }

            // Extract and compare the key
            let raw = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let key_start = SmallQueueItemHeader::SIZE + header.optional_len() as usize;
            let key_end = key_start + header.key_len() as usize;

            if key_end > raw.len() {
                return false;
            }

            &raw[key_start..key_end] == key
        } else {
            // Main cache segment - use standard ItemHeader
            // Validate offset allows for at least a header
            if offset.saturating_add(ItemHeader::SIZE as u32) > self.data_len {
                return false;
            }

            // Try to parse the header - returns None if validation fails
            let header = match ItemHeader::try_from_bytes(unsafe {
                std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE)
            }) {
                Some(h) => h,
                None => return false,
            };

            // Check if item is deleted (unless we allow deleted items)
            if !allow_deleted && header.is_deleted() {
                return false;
            }

            // Check if segment is expired (main cache uses segment-level TTL)
            if !allow_deleted {
                let now = clocksource::coarse::Instant::now();
                if self.expire_at() < now {
                    return false;
                }
            }

            // Validate that full item fits within segment
            let item_size = header.padded_size() as u32;
            if offset.saturating_add(item_size) > self.data_len {
                return false;
            }

            // Extract and compare the key
            let raw = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let key_start = ItemHeader::SIZE + header.optional_len() as usize;
            let key_end = key_start + header.key_len() as usize;

            if key_end > raw.len() {
                return false;
            }

            &raw[key_start..key_end] == key
        }
    }

    fn compact(
        &self,
        hashtable: &crate::hashtable::Hashtable,
    ) -> u32 {
        // Check segment state - compact requires Draining state for exclusive access
        let state = self.state();
        if state != State::Draining {
            return 0;
        }

        let write_offset = self.write_offset.load(Ordering::Acquire);
        fence(Ordering::Acquire);

        // First pass: collect info about live items (offset and size)
        let mut live_items: Vec<(u32, u32)> = Vec::new(); // (offset, size)
        let mut current_offset = 0u32;

        while current_offset < write_offset {
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = ItemHeader::from_bytes_with_context(
                unsafe { std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE) },
                self.id,
                current_offset,
                "compact (first pass)",
            );

            let item_size = header.padded_size() as u32;

            if current_offset + item_size > write_offset {
                break;
            }

            if !header.is_deleted() {
                live_items.push((current_offset, item_size));
            }

            current_offset += item_size;
        }

        // Second pass: move items to the beginning, updating hashtable
        let mut dest_offset = 0u32;

        for (src_offset, item_size) in live_items {
            if src_offset != dest_offset {
                // Need to move this item
                let src_ptr = unsafe { self.data.as_ptr().add(src_offset as usize) };
                let dest_ptr = unsafe { self.data.as_ptr().add(dest_offset as usize) };

                // Extract key before moving (for hashtable update)
                let header = ItemHeader::from_bytes_with_context(
                    unsafe { std::slice::from_raw_parts(src_ptr, ItemHeader::SIZE) },
                    self.id,
                    src_offset,
                    "compact (second pass)",
                );
                let key_start = ItemHeader::SIZE + header.optional_len() as usize;
                let key_end = key_start + header.key_len() as usize;
                let key: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(src_ptr, item_size as usize)[key_start..key_end].to_vec()
                };

                // Move the item data
                unsafe {
                    std::ptr::copy(src_ptr, dest_ptr, item_size as usize);
                }

                // Update hashtable to point to new location (atomic offset update)
                // Note: We use update_item_location which atomically updates the offset
                // while preserving the tag and frequency. No unlinking needed since the
                // item stays in the same segment, just moves to a different offset.
                hashtable.update_item_location(&key, self.pool_id, self.id, src_offset, dest_offset);
            }

            dest_offset += item_size;
        }

        // Update write_offset to reflect compacted size
        self.write_offset.store(dest_offset, Ordering::Release);

        dest_offset
    }

    fn copy_into<S: Segment, F: Fn(u8) -> bool>(
        &self,
        dest: &S,
        hashtable: &crate::hashtable::Hashtable,
        metrics: &crate::metrics::CacheMetrics,
        predicate: F,
    ) -> Option<u32> {
        // Check segment state - copy_into requires Draining state for exclusive access
        let state = self.state();
        if state != State::Draining {
            return None;
        }

        let mut items_copied = 0u32;
        let mut current_offset = 0u32;
        let write_offset = self.write_offset.load(Ordering::Acquire);

        let dest_seg_id = dest.id();

        // Synchronize with append_item's Release fence
        fence(Ordering::Acquire);

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = ItemHeader::from_bytes_with_context(
                unsafe { std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE) },
                self.id,
                current_offset,
                "copy_into",
            );

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip deleted items
            if header.is_deleted() {
                current_offset += item_size;
                continue;
            }

            // Extract item components
            let raw_item = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let optional_start = ItemHeader::SIZE;
            let optional_end = optional_start + header.optional_len() as usize;
            let key_start = optional_end;
            let key_end = key_start + header.key_len() as usize;
            let value_start = key_end;
            let value_end = value_start + header.value_len() as usize;

            let optional = &raw_item[optional_start..optional_end];
            let key = &raw_item[key_start..key_end];
            let value = &raw_item[value_start..value_end];

            // Check if item passes the frequency predicate
            let freq = hashtable.get_item_frequency(key, self.id, current_offset).unwrap_or(0);
            if !predicate(freq) {
                current_offset += item_size;
                continue;
            }

            // Append to destination segment - this increments global metrics
            match dest.append_item(key, value, optional, metrics) {
                Some(new_offset) => {
                    // Check if offset exceeds hashtable limit (stored as offset/8 in 20 bits)
                    // Max offset is 0xFFFFF * 8 = 8,388,600 bytes
                    if new_offset > 0xFFFFF << 3 {
                        // Destination segment is effectively full for hashtable purposes
                        return None;
                    }
                    // Update hashtable to point to new location
                    // Use relink_item which is more efficient since we know the old location
                    // Reset frequency to 1 so item must prove itself in new tier
                    hashtable.relink_item(
                        key,
                        self.pool_id,
                        self.id,
                        current_offset,
                        dest.pool_id(),
                        dest_seg_id,
                        new_offset,
                        false, // reset frequency
                    );

                    // Mark the source item as deleted. This properly decrements the global metrics
                    // (items_live, bytes_live) since the source item is being consumed.
                    // The net effect is: append_item (+1) + mark_deleted (-1) = 0 change,
                    // which is correct since we're moving, not duplicating.
                    let _ = self.mark_deleted(current_offset, key, metrics);

                    items_copied += 1;
                }
                None => {
                    // Destination segment is full
                    return None;
                }
            }

            current_offset += item_size;
        }

        Some(items_copied)
    }

    fn prune(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
    ) -> (u32, u32, u32, u32) {
        // Check segment state - prune requires Draining state for exclusive access
        let state = self.state();
        if state != State::Draining {
            return (0, 0, 0, 0);
        }

        let mut items_retained = 0u32;
        let mut items_pruned = 0u32;
        let mut bytes_retained = 0u32;
        let mut bytes_pruned = 0u32;

        let mut current_offset = 0u32;
        let write_offset = self.write_offset.load(Ordering::Acquire);

        // Synchronize with append_item's Release fence
        fence(Ordering::Acquire);

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = ItemHeader::from_bytes_with_context(
                unsafe { std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE) },
                self.id,
                current_offset,
                "prune",
            );

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip already deleted items
            if header.is_deleted() {
                items_pruned += 1;
                bytes_pruned += item_size;
                current_offset += item_size;
                continue;
            }

            // Extract the key to look up frequency in hashtable
            let raw_item = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let key_start = ItemHeader::SIZE + header.optional_len() as usize;
            let key_end = key_start + header.key_len() as usize;
            let key = &raw_item[key_start..key_end];

            // Look up the item's frequency from the hashtable
            // If not found in hashtable, treat as frequency 0 (will be pruned)
            let freq = hashtable.get_item_frequency(key, self.id, current_offset).unwrap_or(0);

            if freq >= threshold {
                // Item survives - frequency is at or above threshold
                items_retained += 1;
                bytes_retained += item_size;
            } else {
                // Item is pruned - mark it as deleted and convert to ghost entry
                // Ghost entries preserve frequency history for detecting thrashing
                let _ = self.mark_deleted(current_offset, key, metrics);
                hashtable.unlink_item_to_ghost(key, self.id, current_offset, metrics);
                items_pruned += 1;
                bytes_pruned += item_size;
                metrics.merge_evict_items_pruned.increment();
            }

            current_offset += item_size;
        }

        // Update merge eviction metrics
        for _ in 0..items_retained {
            metrics.merge_evict_items_retained.increment();
        }

        (items_retained, items_pruned, bytes_retained, bytes_pruned)
    }

    fn prune_with_demote<F>(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
        mut on_demote: F,
    ) -> (u32, u32, u32, u32)
    where
        F: FnMut(&[u8], &[u8], &[u8]), // (key, value, optional)
    {
        // Check segment state - prune requires Draining state for exclusive access
        let state = self.state();
        if state != State::Draining {
            return (0, 0, 0, 0);
        }

        let mut items_retained = 0u32;
        let mut items_demoted = 0u32;
        let mut bytes_retained = 0u32;
        let mut bytes_demoted = 0u32;

        let mut current_offset = 0u32;
        let write_offset = self.write_offset.load(Ordering::Acquire);

        // Synchronize with append_item's Release fence
        fence(Ordering::Acquire);

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = ItemHeader::from_bytes_with_context(
                unsafe { std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE) },
                self.id,
                current_offset,
                "prune_with_demote",
            );

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip already deleted items (count as demoted since they're gone)
            if header.is_deleted() {
                items_demoted += 1;
                bytes_demoted += item_size;
                current_offset += item_size;
                continue;
            }

            // Extract item data
            let raw_item = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let optional_start = ItemHeader::SIZE;
            let optional_end = optional_start + header.optional_len() as usize;
            let key_start = optional_end;
            let key_end = key_start + header.key_len() as usize;
            let value_start = key_end;
            let value_end = value_start + header.value_len() as usize;

            let optional = &raw_item[optional_start..optional_end];
            let key = &raw_item[key_start..key_end];
            let value = &raw_item[value_start..value_end];

            // Look up the item's frequency from the hashtable
            // If not found in hashtable, treat as frequency 0 (will be demoted)
            let freq = hashtable.get_item_frequency(key, self.id, current_offset).unwrap_or(0);

            if freq >= threshold {
                // Item survives - frequency is at or above threshold
                items_retained += 1;
                bytes_retained += item_size;
            } else {
                // Item is demoted - unlink from hashtable first, then call callback
                hashtable.unlink_item(key, self.id, current_offset, metrics);

                // Call the demotion callback with item data
                on_demote(key, value, optional);

                // Mark as deleted in this segment
                let _ = self.mark_deleted(current_offset, key, metrics);

                items_demoted += 1;
                bytes_demoted += item_size;
                metrics.merge_evict_items_pruned.increment();
            }

            current_offset += item_size;
        }

        // Update merge eviction metrics
        for _ in 0..items_retained {
            metrics.merge_evict_items_retained.increment();
        }

        (items_retained, items_demoted, bytes_retained, bytes_demoted)
    }

    fn prune_collecting_for_demote(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        threshold: u8,
        metrics: &crate::metrics::CacheMetrics,
    ) -> (u32, u32, u32, u32, Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) {
        // Check segment state - prune requires Draining state for exclusive access
        let state = self.state();
        if state != State::Draining {
            return (0, 0, 0, 0, Vec::new());
        }

        let mut items_retained = 0u32;
        let mut items_demoted = 0u32;
        let mut bytes_retained = 0u32;
        let mut bytes_demoted = 0u32;
        let mut items_to_demote = Vec::new();

        let mut current_offset = 0u32;
        let write_offset = self.write_offset.load(Ordering::Acquire);

        // Synchronize with append_item's Release fence
        fence(Ordering::Acquire);

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = ItemHeader::from_bytes_with_context(
                unsafe { std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE) },
                self.id,
                current_offset,
                "prune_collecting_for_demote",
            );

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip already deleted items (count as demoted since they're gone)
            if header.is_deleted() {
                items_demoted += 1;
                bytes_demoted += item_size;
                current_offset += item_size;
                continue;
            }

            // Extract item data
            let raw_item = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let optional_start = ItemHeader::SIZE;
            let optional_end = optional_start + header.optional_len() as usize;
            let key_start = optional_end;
            let key_end = key_start + header.key_len() as usize;
            let value_start = key_end;
            let value_end = value_start + header.value_len() as usize;

            let optional = &raw_item[optional_start..optional_end];
            let key = &raw_item[key_start..key_end];
            let value = &raw_item[value_start..value_end];

            // Look up the item's frequency from the hashtable
            // If not found in hashtable, treat as frequency 0 (will be demoted)
            let freq = hashtable.get_item_frequency(key, self.id, current_offset).unwrap_or(0);

            if freq >= threshold {
                // Item survives - frequency is at or above threshold
                items_retained += 1;
                bytes_retained += item_size;
            } else {
                // Item is demoted - unlink from hashtable first
                hashtable.unlink_item(key, self.id, current_offset, metrics);

                // Collect item data for later async demotion
                items_to_demote.push((key.to_vec(), value.to_vec(), optional.to_vec()));

                // Mark as deleted in this segment
                let _ = self.mark_deleted(current_offset, key, metrics);

                items_demoted += 1;
                bytes_demoted += item_size;
                metrics.merge_evict_items_pruned.increment();
            }

            current_offset += item_size;
        }

        // Update merge eviction metrics
        for _ in 0..items_retained {
            metrics.merge_evict_items_retained.increment();
        }

        (items_retained, items_demoted, bytes_retained, bytes_demoted, items_to_demote)
    }

    fn unlink_all_items(
        &self,
        hashtable: &crate::hashtable::Hashtable,
        metrics: &crate::metrics::CacheMetrics,
        create_ghosts: bool,
    ) -> u32 {
        let mut items_unlinked = 0u32;

        let mut current_offset = 0u32;
        let write_offset = self.write_offset.load(Ordering::Acquire);

        // Synchronize with append_item's Release fence
        fence(Ordering::Acquire);

        while current_offset < write_offset {
            // Validate header fits within segment
            if current_offset + ItemHeader::SIZE as u32 > self.data_len {
                break;
            }

            let data_ptr = unsafe { self.data.as_ptr().add(current_offset as usize) };
            let header = match ItemHeader::try_from_bytes(unsafe {
                std::slice::from_raw_parts(data_ptr, ItemHeader::SIZE)
            }) {
                Some(h) => h,
                None => break, // Invalid header, stop scanning
            };

            let item_size = header.padded_size() as u32;

            // Validate full item fits within segment
            if current_offset + item_size > write_offset {
                break;
            }

            // Skip already deleted items - they're already unlinked
            if header.is_deleted() {
                current_offset += item_size;
                continue;
            }

            // Extract the key to unlink from hashtable
            let raw_item = unsafe { std::slice::from_raw_parts(data_ptr, header.padded_size()) };
            let key_start = ItemHeader::SIZE + header.optional_len() as usize;
            let key_end = key_start + header.key_len() as usize;
            let key = &raw_item[key_start..key_end];

            // Unlink from hashtable
            if create_ghosts {
                // Convert to ghost entry to preserve frequency history (for FIFO eviction)
                hashtable.unlink_item_to_ghost(key, self.id, current_offset, metrics);
            } else {
                // Just remove the entry (for TTL expiration - items naturally aged out)
                hashtable.unlink_item(key, self.id, current_offset, metrics);
            }
            items_unlinked += 1;

            current_offset += item_size;
        }

        items_unlinked
    }

    fn append_small_queue_item(
        &self,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<u32> {
        use crate::item::SmallQueueItemHeader;

        // Validate segment data structures are not corrupted
        assert!(self.data_len > 0,
            "CORRUPTION: segment {} has data_len=0", self.id);

        if key.is_empty() || key.len() > SmallQueueItemHeader::MAX_KEY_LEN {
            panic!(
                "key size is out of range: must be 1-{} bytes",
                SmallQueueItemHeader::MAX_KEY_LEN
            );
        }

        if optional.len() > SmallQueueItemHeader::MAX_OPTIONAL_LEN {
            panic!(
                "optional size is out of range: must be 0-{} bytes",
                SmallQueueItemHeader::MAX_OPTIONAL_LEN
            );
        }

        if value.len() > SmallQueueItemHeader::MAX_VALUE_LEN {
            panic!(
                "value size is out of range: must be 0-{} bytes",
                SmallQueueItemHeader::MAX_VALUE_LEN
            );
        }

        let header = SmallQueueItemHeader::new(
            key.len() as u8,
            optional.len() as u8,
            value.len() as u32,
            false, // is_deleted
            false, // is_numeric
            expire_at,
        );

        if header.padded_size() as u32 > self.data_len {
            panic!("item size is out of range. increase segment size");
        }

        let item_size = header.padded_size() as u32;

        // Use the standard CAS retry pattern for reserving space
        let reserved_offset = match retry_cas_u32(
            &self.write_offset,
            |current_offset| {
                let new_offset = current_offset.saturating_add(item_size);

                // Check if there's enough space
                if new_offset > self.data_len {
                    return None; // Segment is full
                }

                Some((new_offset, current_offset))
            },
            CasRetryConfig {
                max_attempts: 16,
                early_spin_threshold: 4,
            },
            metrics,
        ) {
            CasResult::Success(offset) => offset,
            CasResult::Failed(_) | CasResult::Aborted => {
                // Segment is full, return None
                metrics.item_append_full.increment();
                return None;
            }
        };

        // Space successfully reserved, now write the data
        {
            // Validation checks
            assert!(reserved_offset < self.data_len,
                "CORRUPTION: reserved_offset ({}) >= data_len ({}) in segment {}",
                reserved_offset, self.data_len, self.id);
            assert!(reserved_offset.saturating_add(item_size) <= self.data_len,
                "CORRUPTION: reserved_offset ({}) + item_size ({}) > data_len ({}) in segment {}",
                reserved_offset, item_size, self.data_len, self.id);

            let mut data_ptr = unsafe { self.data.as_ptr().add(reserved_offset as usize) };
            let segment_end = unsafe { self.data.as_ptr().add(self.data_len as usize) };

            // Write header
            {
                let data = unsafe { std::slice::from_raw_parts_mut(data_ptr, SmallQueueItemHeader::SIZE) };
                header.to_bytes(data);
            }

            unsafe {
                data_ptr = data_ptr.add(SmallQueueItemHeader::SIZE);

                // Copy optional metadata
                if !optional.is_empty() {
                    assert!(data_ptr.add(optional.len()) <= segment_end,
                        "CORRUPTION: optional write would exceed segment bounds");
                    std::ptr::copy_nonoverlapping(optional.as_ptr(), data_ptr, optional.len());
                    data_ptr = data_ptr.add(optional.len());
                }

                // Copy key
                assert!(data_ptr.add(key.len()) <= segment_end,
                    "CORRUPTION: key write would exceed segment bounds");
                std::ptr::copy_nonoverlapping(key.as_ptr(), data_ptr, key.len());
                data_ptr = data_ptr.add(key.len());

                // Copy value
                if !value.is_empty() {
                    assert!(data_ptr.add(value.len()) <= segment_end,
                        "CORRUPTION: value write would exceed segment bounds");
                    std::ptr::copy_nonoverlapping(value.as_ptr(), data_ptr, value.len());
                }
            }
        }

        // Memory fence to ensure all writes are visible before item is linked
        fence(Ordering::Release);

        // Update live items/bytes counters
        self.live_items.fetch_add(1, Ordering::Relaxed);
        self.live_bytes.fetch_add(item_size, Ordering::Relaxed);

        metrics.item_append.increment();

        Some(reserved_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::CacheMetrics;
    use std::alloc::{alloc, dealloc, Layout};

    /// Create a test segment with the given size
    fn create_test_segment(pool_id: u8, is_small_queue: bool, id: u32, size: usize) -> (SliceSegment<'static>, *mut u8, Layout) {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null());

        // Zero the memory
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }

        let segment = unsafe { SliceSegment::new(pool_id, is_small_queue, id, ptr, size) };
        (segment, ptr, layout)
    }

    /// Free a test segment
    unsafe fn free_test_segment(ptr: *mut u8, layout: Layout) {
        unsafe { dealloc(ptr, layout); }
    }

    #[test]
    fn test_segment_creation() {
        let (segment, ptr, layout) = create_test_segment(0, false, 42, 1024);

        assert_eq!(segment.id(), 42);
        assert_eq!(segment.pool_id(), 0);
        assert!(!segment.is_small_queue());
        assert_eq!(segment.data_len(), 1024);
        assert_eq!(segment.state(), State::Free);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_small_queue_flag() {
        let (segment, ptr, layout) = create_test_segment(1, true, 0, 1024);

        assert!(segment.is_small_queue());
        assert_eq!(segment.pool_id(), 1);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_reserve_release() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // Initially Free
        assert_eq!(segment.state(), State::Free);

        // Reserve
        assert!(segment.try_reserve());
        assert_eq!(segment.state(), State::Reserved);

        // Release
        assert!(segment.try_release());
        assert_eq!(segment.state(), State::Free);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_double_reserve() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // First reserve succeeds
        assert!(segment.try_reserve());

        // Second reserve fails (already Reserved)
        assert!(!segment.try_reserve());

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_merge_count() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // Reserve first
        segment.try_reserve();

        // Initial merge count is 0
        assert_eq!(segment.merge_count(), 0);

        // Increment
        segment.increment_merge_count();
        assert_eq!(segment.merge_count(), 1);

        segment.increment_merge_count();
        assert_eq!(segment.merge_count(), 2);

        // Release resets merge count
        segment.try_release();
        segment.try_reserve();
        assert_eq!(segment.merge_count(), 0);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_data_slice() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // Write some test data directly using the raw pointer
        unsafe {
            std::ptr::write_bytes(ptr.add(100), 0xAB, 10);
        }

        // Read it back via data_slice
        let slice = segment.data_slice(100, 10);
        assert!(slice.is_some());
        let slice = slice.unwrap();
        assert_eq!(slice.len(), 10);
        assert!(slice.iter().all(|&b| b == 0xAB));

        // Out of bounds should return None
        let oob = segment.data_slice(1020, 10);
        assert!(oob.is_none());

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_append_item() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 4096);
        let metrics = CacheMetrics::new();

        // Reserve the segment first
        segment.try_reserve();

        // Append an item
        let key = b"test_key";
        let value = b"test_value";
        let optional = b"";

        let offset = segment.append_item(key, value, optional, &metrics);
        assert!(offset.is_some());
        let offset = offset.unwrap();
        assert_eq!(offset, 0); // First item at offset 0

        // Verify counters
        assert_eq!(segment.live_items(), 1);
        assert!(segment.live_bytes() > 0);

        // Append another item
        let offset2 = segment.append_item(b"key2", b"value2", b"", &metrics);
        assert!(offset2.is_some());
        assert!(offset2.unwrap() > offset); // Should be after first item

        assert_eq!(segment.live_items(), 2);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_append_item_full() {
        // Create a small segment (256 bytes)
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 256);
        let metrics = CacheMetrics::new();

        segment.try_reserve();

        // First, append a valid item that takes up most of the space
        let key = b"k";
        let value = vec![0u8; 200]; // Takes most of the segment
        let result = segment.append_item(key, &value, b"", &metrics);
        assert!(result.is_some());

        // Second item should fail - not enough space
        let result2 = segment.append_item(b"k2", &value, b"", &metrics);
        assert!(result2.is_none());

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_bucket_id() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // Initially no bucket
        assert_eq!(segment.bucket_id(), None);

        // Set bucket
        segment.set_bucket_id(42);
        assert_eq!(segment.bucket_id(), Some(42));

        // Clear bucket
        segment.clear_bucket_id();
        assert_eq!(segment.bucket_id(), None);

        unsafe { free_test_segment(ptr, layout); }
    }

    #[test]
    fn test_segment_ref_count() {
        let (segment, ptr, layout) = create_test_segment(0, false, 0, 1024);

        // Reserve first
        segment.try_reserve();

        // Initial ref count is 0
        assert_eq!(segment.ref_count(), 0);

        // Manually increment ref count via the internal atomic
        segment.ref_count.fetch_add(1, Ordering::AcqRel);
        assert_eq!(segment.ref_count(), 1);

        // Decrement
        segment.decr_ref_count();
        assert_eq!(segment.ref_count(), 0);

        unsafe { free_test_segment(ptr, layout); }
    }
}
