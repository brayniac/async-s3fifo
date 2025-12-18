use std::fs::OpenOptions;
use std::path::PathBuf;
use memmap2::MmapMut;
use crate::pool::Pool;
use crate::segment::{Segment, SliceSegment};

pub struct MmapPool {
    /// Memory mapped region for backing the pool.
    /// This field appears unused but is essential - segments hold raw pointers
    /// into this mmap region, and it must live as long as the pool.
    /// Dropping this unmaps the memory.
    #[allow(dead_code)]
    mmap: MmapMut,

    /// Segment metadata (in RAM for fast access)
    segments: Vec<SliceSegment<'static>>,

    /// Lock-free free list for small queue segments
    small_queue_free: crossbeam_deque::Injector<u32>,

    /// Lock-free free list for main cache segments
    main_cache_free: crossbeam_deque::Injector<u32>,
}

impl Pool for MmapPool {
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

pub struct MmapPoolBuilder {
    pool_id: u8,
    segment_size: usize,
    heap_size: usize,
    small_queue_percent: u8,
    path: PathBuf,
}

impl MmapPoolBuilder {
    pub fn new(pool_id: u8, path: impl Into<PathBuf>) -> Self {
        Self {
            pool_id,
            segment_size: 1024 * 1024,
            heap_size: 64 * 1024 * 1024,
            small_queue_percent: 10,
            path: path.into(),
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

    pub fn build(self) -> Result<MmapPool, std::io::Error> {
        let num_segments = self.heap_size / self.segment_size;
        let small_queue_count = (num_segments * self.small_queue_percent as usize) / 100;

        let actual_size = num_segments * self.segment_size;

        // Create or open the backing file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;

        // Set file size
        file.set_len(actual_size as u64)?;

        // Memory map the file
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize segment metadata
        let mut segments = Vec::with_capacity(num_segments);
        let small_queue_free = crossbeam_deque::Injector::new();
        let main_cache_free = crossbeam_deque::Injector::new();

        let heap_ptr = mmap.as_mut_ptr();

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

        Ok(MmapPool {
            mmap,
            segments,
            small_queue_free,
            main_cache_free,
        })
    }
}