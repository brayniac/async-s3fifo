use std::alloc::{alloc_zeroed, dealloc, Layout};
use crate::pool::Pool;
use crate::segment::{Segment, SliceSegment};

pub struct MemoryPool {
    /// Pointer to the allocated heap memory
    heap_ptr: *mut u8,

    /// Layout used for allocation (needed for deallocation)
    layout: Layout,

    /// Segment metadata (in RAM for fast access)
    segments: Vec<SliceSegment<'static>>,

    /// Lock-free free list for small queue segments
    small_queue_free: crossbeam_deque::Injector<u32>,

    /// Lock-free free list for main cache segments
    main_cache_free: crossbeam_deque::Injector<u32>,
}

// SAFETY: MemoryPool is safe to send between threads because:
// 1. heap_ptr is allocated once at construction and never moved or freed until Drop
// 2. All segment access is through SliceSegment which uses atomic operations
// 3. The pool itself doesn't mutate heap_ptr after construction
// 4. The free lists (Injector) are already Send + Sync
unsafe impl Send for MemoryPool {}

// SAFETY: MemoryPool is safe to share between threads because:
// 1. All methods take &self and use lock-free operations
// 2. Segment state transitions use atomic CAS operations
// 3. The heap memory is accessed through SliceSegment's atomic slice access
unsafe impl Sync for MemoryPool {}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // SAFETY: heap_ptr was allocated with this layout in build()
        // and has not been deallocated yet
        unsafe {
            dealloc(self.heap_ptr, self.layout);
        }
    }
}

impl Pool for MemoryPool {
    type Segment = SliceSegment<'static>;

    fn get(&self, id: u32) -> Option<&Self::Segment> {
        self.segments.get(id as usize)
    }

    fn segment_count(&self) -> usize {
        self.segments.len()
    }

    fn reserve_small_queue(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        match self.small_queue_free.steal() {
            crossbeam_deque::Steal::Success(segment_id) => {
                let segment = &self.segments[segment_id as usize];
                debug_assert!(segment.is_small_queue(), "segment from small_queue_free must be small queue type");

                // Transition Free -> Reserved and reset segment state
                // If this fails (segment not in Free state), don't return the segment
                if !segment.try_reserve() {
                    // Segment was not in Free state - likely a duplicate in the queue
                    // or concurrent modification. Push it back and return None.
                    self.small_queue_free.push(segment_id);
                    return None;
                }

                metrics.segment_reserve.increment();
                metrics.segments_free.decrement();
                Some(segment_id)
            }
            crossbeam_deque::Steal::Empty => None,
            crossbeam_deque::Steal::Retry => None,
        }
    }

    fn reserve_main_cache(&self, metrics: &crate::metrics::CacheMetrics) -> Option<u32> {
        match self.main_cache_free.steal() {
            crossbeam_deque::Steal::Success(segment_id) => {
                let segment = &self.segments[segment_id as usize];
                debug_assert!(!segment.is_small_queue(), "segment from main_cache_free must be main cache type");

                // Transition Free -> Reserved and reset segment state
                // If this fails (segment not in Free state), don't return the segment
                if !segment.try_reserve() {
                    // Segment was not in Free state - likely a duplicate in the queue
                    // or concurrent modification. Push it back and return None.
                    self.main_cache_free.push(segment_id);
                    return None;
                }

                metrics.segment_reserve.increment();
                metrics.segments_free.decrement();
                Some(segment_id)
            }
            crossbeam_deque::Steal::Empty => None,
            crossbeam_deque::Steal::Retry => None,
        }
    }

    fn release(&self, id: u32, metrics: &crate::metrics::CacheMetrics) {
        let id_usize = id as usize;

        // Bounds check
        if id_usize >= self.segments.len() {
            panic!("Invalid segment ID: {id}");
        }

        let segment = &self.segments[id_usize];

        // Transition Reserved/Linking/Locked -> Free
        // Only add to free queue if we successfully transitioned to Free
        if segment.try_release() {
            metrics.segment_release.increment();
            metrics.segments_free.increment();

            // Add segment ID back to the appropriate free queue
            if segment.is_small_queue() {
                self.small_queue_free.push(id);
            } else {
                self.main_cache_free.push(id);
            }
        }
        // If try_release returns false, segment was already Free
        // Don't push to queue again - would create duplicate
    }
}

pub struct MemoryPoolBuilder {
    pool_id: u8,
    segment_size: usize,
    heap_size: usize,
    small_queue_percent: u8,
}

impl MemoryPoolBuilder {
    pub fn new(pool_id: u8) -> Self {
        Self {
            pool_id,
            segment_size: 1024 * 1024,
            heap_size: 64 * 1024 * 1024,
            small_queue_percent: 10,
        }
    }

    /// Set the segment size in bytes (default: 1MB)
    pub fn segment_size(mut self, size: usize) -> Self {
        self.segment_size = size;
        self
    }

    /// Set the total heap size in bytes (default: 64MB)
    ///
    /// The number of segments is calculated as heap_size / segment_size.
    pub fn heap_size(mut self, size: usize) -> Self {
        self.heap_size = size;
        self
    }

    /// Set the percentage of segments allocated to the small queue (admission queue)
    pub fn small_queue_percent(mut self, percent: u8) -> Self {
        debug_assert!(percent <= 100, "small_queue_percent must be <= 100");
        self.small_queue_percent = percent;
        self
    }

    pub fn build(self) -> Result<MemoryPool, std::io::Error> {
        let num_segments = self.heap_size / self.segment_size;
        let small_queue_count = (num_segments * self.small_queue_percent as usize) / 100;

        let actual_size = num_segments * self.segment_size;

        // Allocate the entire heap as a single page-aligned block
        // Use 2MB alignment for potential huge page support on systems that support it
        // Falls back to regular pages if huge pages are not available
        const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB
        const REGULAR_PAGE_SIZE: usize = 4096;

        let alignment = if actual_size >= HUGE_PAGE_SIZE && actual_size.is_multiple_of(HUGE_PAGE_SIZE)
        {
            HUGE_PAGE_SIZE
        } else {
            REGULAR_PAGE_SIZE
        };

        let layout =
            Layout::from_size_align(actual_size, alignment).expect("Failed to create layout");

        // Use alloc_zeroed to get zero-initialized memory
        // This is important for security and consistency
        let heap_ptr = unsafe { alloc_zeroed(layout) };
        if heap_ptr.is_null() {
            panic!(
                "Failed to allocate {} bytes for segments heap",
                self.heap_size
            );
        }

        // Pre-fault all pages by touching them
        // This forces the OS to allocate physical pages now rather than on first access
        // which avoids page faults during critical write operations
        unsafe {
            // Touch only one location per page to minimize memory traffic
            // One write per page is sufficient to fault the entire page
            const PAGE_SIZE: usize = 4096;

            for i in (0..actual_size).step_by(PAGE_SIZE) {
                std::ptr::write_volatile(heap_ptr.add(i) as *mut u64, 0);
            }
        }

        // Initialize segment metadata
        let mut segments = Vec::with_capacity(num_segments);
        let small_queue_free = crossbeam_deque::Injector::new();
        let main_cache_free = crossbeam_deque::Injector::new();

        for id in 0..num_segments {
            let mmap_offset = id * self.segment_size;
            let segment_ptr = unsafe { heap_ptr.add(mmap_offset) };

            // Segments [0..small_queue_count) are small queue, rest are main cache
            let is_small_queue = id < small_queue_count;

            // Create lock-free segment
            let segment = unsafe { SliceSegment::new(self.pool_id, is_small_queue, id as u32, segment_ptr, self.segment_size) };

            segments.push(segment);

            // Add to appropriate free queue
            if is_small_queue {
                small_queue_free.push(id as u32);
            } else {
                main_cache_free.push(id as u32);
            }
        }

        Ok(MemoryPool {
            heap_ptr,
            layout,
            segments,
            small_queue_free,
            main_cache_free,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::CacheMetrics;
    use crate::segment::Segment;

    fn create_test_pool() -> MemoryPool {
        MemoryPoolBuilder::new(0)
            .segment_size(64 * 1024) // 64KB segments
            .heap_size(640 * 1024)   // 640KB = 10 segments
            .small_queue_percent(20) // 2 small queue, 8 main cache
            .build()
            .expect("Failed to create test pool")
    }

    #[test]
    fn test_pool_builder_defaults() {
        let pool = MemoryPoolBuilder::new(0)
            .build()
            .expect("Failed to create pool with defaults");

        // Default is 64MB heap / 1MB segments = 64 segments
        assert_eq!(pool.segments.len(), 64);
    }

    #[test]
    fn test_pool_segment_count() {
        let pool = create_test_pool();
        assert_eq!(pool.segments.len(), 10);
    }

    #[test]
    fn test_reserve_small_queue() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        // Should be able to reserve 2 small queue segments (20% of 10)
        let seg1 = pool.reserve_small_queue(&metrics);
        assert!(seg1.is_some());

        let seg2 = pool.reserve_small_queue(&metrics);
        assert!(seg2.is_some());

        // Third should fail - only 2 small queue segments
        let seg3 = pool.reserve_small_queue(&metrics);
        assert!(seg3.is_none());

        // Verify the segments are marked as small queue
        if let Some(id) = seg1 {
            let segment = pool.get(id).unwrap();
            assert!(segment.is_small_queue());
        }
    }

    #[test]
    fn test_reserve_main_cache() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        // Should be able to reserve 8 main cache segments (80% of 10)
        let mut reserved = Vec::new();
        for _ in 0..8 {
            let seg = pool.reserve_main_cache(&metrics);
            assert!(seg.is_some(), "Should be able to reserve main cache segment");
            reserved.push(seg.unwrap());
        }

        // Ninth should fail - only 8 main cache segments
        let seg9 = pool.reserve_main_cache(&metrics);
        assert!(seg9.is_none());

        // Verify segments are NOT small queue
        for id in &reserved {
            let segment = pool.get(*id).unwrap();
            assert!(!segment.is_small_queue());
        }
    }

    #[test]
    fn test_reserve_and_release() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        // Reserve all small queue segments
        let seg1 = pool.reserve_small_queue(&metrics).unwrap();
        let seg2 = pool.reserve_small_queue(&metrics).unwrap();
        assert!(pool.reserve_small_queue(&metrics).is_none());

        // Release one
        pool.release(seg1, &metrics);

        // Should be able to reserve again
        let seg3 = pool.reserve_small_queue(&metrics);
        assert!(seg3.is_some());

        // Clean up
        pool.release(seg2, &metrics);
        pool.release(seg3.unwrap(), &metrics);
    }

    #[test]
    fn test_get_segment() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        let seg_id = pool.reserve_main_cache(&metrics).unwrap();
        let segment = pool.get(seg_id);
        assert!(segment.is_some());

        // Invalid segment ID should return None
        let invalid = pool.get(999);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_segment_data_accessible() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        let seg_id = pool.reserve_main_cache(&metrics).unwrap();
        let segment = pool.get(seg_id).unwrap();

        // Should be able to get data length (capacity)
        let data_len = segment.data_len();
        assert_eq!(data_len, 64 * 1024); // 64KB

        // Clean up
        pool.release(seg_id, &metrics);
    }

    #[test]
    fn test_segment_types_dont_mix() {
        let pool = create_test_pool();
        let metrics = CacheMetrics::new();

        // Reserve a small queue segment
        let small_id = pool.reserve_small_queue(&metrics).unwrap();
        let small_seg = pool.get(small_id).unwrap();
        assert!(small_seg.is_small_queue());

        // Reserve a main cache segment
        let main_id = pool.reserve_main_cache(&metrics).unwrap();
        let main_seg = pool.get(main_id).unwrap();
        assert!(!main_seg.is_small_queue());

        // Release small queue segment
        pool.release(small_id, &metrics);

        // Reserving main cache should NOT get the released small queue segment
        // (This would require draining all main cache segments first to test properly)

        pool.release(main_id, &metrics);
    }
}