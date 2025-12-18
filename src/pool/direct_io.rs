//! Direct I/O pool implementation
//!
//! This module provides a file-backed storage pool that bypasses the OS page cache
//! using O_DIRECT on Linux and F_NOCACHE on macOS. This gives the cache full control
//! over memory usage - what's in your RAM tier is what's in RAM, period.
//!
//! Unlike MmapPool, DirectIoPool does not provide direct memory access to segment data.
//! Reads must go through explicit I/O operations, and items should be promoted to RAM
//! before returning to callers.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::metrics::CacheMetrics;

/// Alignment requirement for direct I/O (4KB is safe for most systems)
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// A file handle configured for direct I/O (cache-bypassing) operations.
///
/// On Linux, this uses O_DIRECT to bypass the page cache entirely.
/// On macOS, this uses F_NOCACHE which hints to the kernel to not cache pages.
/// On other platforms, this falls back to regular buffered I/O.
pub struct DirectIoFile {
    file: File,
    path: PathBuf,
}

impl DirectIoFile {
    /// Open or create a file with direct I/O flags.
    ///
    /// The file is created/truncated and sized to `size` bytes.
    pub fn open(path: impl AsRef<Path>, size: u64) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create the file first with standard options
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Set file size
        file.set_len(size)?;

        // Apply platform-specific direct I/O flags
        Self::apply_direct_io_flags(&file)?;

        Ok(Self { file, path })
    }

    /// Apply platform-specific flags to bypass the page cache.
    #[cfg(target_os = "linux")]
    fn apply_direct_io_flags(file: &File) -> io::Result<()> {
        use std::os::unix::io::AsRawFd;

        // On Linux, we need to reopen with O_DIRECT
        // Since we can't add O_DIRECT to an existing fd, we use fcntl to get/set flags
        // Note: O_DIRECT requires aligned buffers and offsets
        let fd = file.as_raw_fd();

        // Get current flags
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags == -1 {
            return Err(io::Error::last_os_error());
        }

        // Add O_DIRECT
        let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_DIRECT) };
        if result == -1 {
            // O_DIRECT might not be supported on this filesystem (e.g., tmpfs)
            // Log warning but continue - we'll still work, just without direct I/O
            eprintln!(
                "Warning: O_DIRECT not supported on this filesystem, falling back to buffered I/O"
            );
        }

        Ok(())
    }

    /// Apply platform-specific flags to bypass the page cache.
    #[cfg(target_os = "macos")]
    fn apply_direct_io_flags(file: &File) -> io::Result<()> {
        use std::os::unix::io::AsRawFd;

        let fd = file.as_raw_fd();

        // F_NOCACHE: Turns off data caching for this file
        // Unlike O_DIRECT, this doesn't require alignment
        let result = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
        if result == -1 {
            return Err(io::Error::last_os_error());
        }

        // F_RDAHEAD: Disable read-ahead
        let result = unsafe { libc::fcntl(fd, libc::F_RDAHEAD, 0) };
        if result == -1 {
            // Not critical, continue anyway
            eprintln!("Warning: Failed to disable read-ahead");
        }

        Ok(())
    }

    /// Apply platform-specific flags - no-op on unsupported platforms.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn apply_direct_io_flags(_file: &File) -> io::Result<()> {
        // No direct I/O support on this platform
        // Fall back to regular buffered I/O
        Ok(())
    }

    /// Read data from the file at the given offset.
    ///
    /// For O_DIRECT on Linux, both offset and buffer length should be aligned
    /// to DIRECT_IO_ALIGNMENT for best performance. Unaligned reads may fail
    /// or fall back to buffered I/O depending on the filesystem.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_at(buf, offset)
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix platforms
            let mut file = &self.file;
            file.seek(SeekFrom::Start(offset))?;
            file.read(buf)
        }
    }

    /// Write data to the file at the given offset.
    ///
    /// For O_DIRECT on Linux, both offset and buffer length should be aligned
    /// to DIRECT_IO_ALIGNMENT for best performance.
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<usize> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.write_at(buf, offset)
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix platforms
            let mut file = &self.file;
            file.seek(SeekFrom::Start(offset))?;
            file.write(buf)
        }
    }

    /// Sync file data to disk.
    pub fn sync_data(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// Get the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Segment metadata for direct I/O pool.
///
/// Unlike SliceSegment which holds a pointer to memory-mapped data,
/// DirectIoSegment only tracks metadata. The actual data lives on disk
/// and must be read via DirectIoFile.
#[allow(dead_code)] // Fields reserved for future use (TTL bucket management)
pub struct DirectIoSegment {
    /// Segment ID within the pool
    id: u32,

    /// Pool ID (identifies which pool this segment belongs to)
    pool_id: u8,

    /// Whether this segment is for the small queue (vs main cache)
    is_small_queue: bool,

    /// Current write offset within the segment
    write_offset: crate::sync::AtomicU32,

    /// Number of live items in the segment
    live_items: crate::sync::AtomicU32,

    /// Number of live bytes in the segment
    live_bytes: crate::sync::AtomicU32,

    /// Reference count for readers
    ref_count: crate::sync::AtomicU32,

    /// Packed metadata: state (8b) | prev (24b) | next (24b)
    metadata: crate::sync::AtomicU64,

    /// Expiration time (Unix timestamp)
    expire_at: crate::sync::AtomicU32,

    /// TTL bucket ID
    bucket_id: crate::sync::AtomicU16,

    /// Generation counter for CAS tokens
    generation: crate::sync::AtomicU16,

    /// Merge count for tracking merge destinations
    merge_count: crate::sync::AtomicU16,

    /// Segment size in bytes
    segment_size: u32,
}

impl DirectIoSegment {
    /// Create a new direct I/O segment metadata.
    pub fn new(pool_id: u8, is_small_queue: bool, id: u32, segment_size: usize) -> Self {
        use crate::sync::Ordering;

        let segment = Self {
            id,
            pool_id,
            is_small_queue,
            write_offset: crate::sync::AtomicU32::new(0),
            live_items: crate::sync::AtomicU32::new(0),
            live_bytes: crate::sync::AtomicU32::new(0),
            ref_count: crate::sync::AtomicU32::new(0),
            metadata: crate::sync::AtomicU64::new(0), // Free state
            expire_at: crate::sync::AtomicU32::new(0),
            bucket_id: crate::sync::AtomicU16::new(0),
            generation: crate::sync::AtomicU16::new(0),
            merge_count: crate::sync::AtomicU16::new(0),
            segment_size: segment_size as u32,
        };

        // Initialize to Free state
        segment
            .metadata
            .store(Self::pack_metadata(0, 0xFFFFFF, 0xFFFFFF), Ordering::Release);

        segment
    }

    /// Pack state and chain pointers into metadata word.
    #[inline]
    fn pack_metadata(state: u8, prev: u32, next: u32) -> u64 {
        ((state as u64) << 56) | ((prev as u64 & 0xFFFFFF) << 24) | (next as u64 & 0xFFFFFF)
    }

    /// Unpack state from metadata word.
    #[inline]
    fn unpack_state(metadata: u64) -> u8 {
        (metadata >> 56) as u8
    }

    /// Get the file offset for this segment's data.
    #[inline]
    pub fn file_offset(&self) -> u64 {
        self.id as u64 * self.segment_size as u64
    }

    /// Get the segment size.
    #[inline]
    pub fn segment_size(&self) -> usize {
        self.segment_size as usize
    }

    /// Get the expiration time (unix timestamp in seconds).
    #[inline]
    pub fn expire_at(&self) -> u32 {
        self.expire_at.load(crate::sync::Ordering::Acquire)
    }

    /// Set the expiration time (unix timestamp in seconds).
    #[inline]
    pub fn set_expire_at(&self, expire_at: u32) {
        self.expire_at.store(expire_at, crate::sync::Ordering::Release);
    }
}

/// Direct I/O backed segment pool.
///
/// This pool stores segment data in a file using direct I/O (O_DIRECT on Linux,
/// F_NOCACHE on macOS) to bypass the OS page cache. Segment metadata is kept
/// in RAM for fast access.
///
/// Unlike MmapPool, this pool does not provide direct memory access to segment
/// data. Callers must use read/write methods to access data.
pub struct DirectIoPool {
    /// Direct I/O file handle
    file: Arc<DirectIoFile>,

    /// Segment metadata (in RAM)
    segments: Vec<DirectIoSegment>,

    /// Lock-free free list for small queue segments
    small_queue_free: crossbeam_deque::Injector<u32>,

    /// Lock-free free list for main cache segments
    main_cache_free: crossbeam_deque::Injector<u32>,

    /// Segment size in bytes
    segment_size: usize,
}

impl DirectIoPool {
    /// Read segment data from disk.
    ///
    /// Returns the number of bytes read.
    pub fn read_segment(&self, segment_id: u32, offset: u32, buf: &mut [u8]) -> io::Result<usize> {
        let segment = self
            .segments
            .get(segment_id as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Invalid segment ID"))?;

        let file_offset = segment.file_offset() + offset as u64;
        self.file.read_at(file_offset, buf)
    }

    /// Write data to a segment on disk.
    ///
    /// Returns the number of bytes written.
    pub fn write_segment(&self, segment_id: u32, offset: u32, buf: &[u8]) -> io::Result<usize> {
        let segment = self
            .segments
            .get(segment_id as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Invalid segment ID"))?;

        let file_offset = segment.file_offset() + offset as u64;
        self.file.write_at(file_offset, buf)
    }

    /// Get the segment size.
    pub fn segment_size(&self) -> usize {
        self.segment_size
    }

    /// Get the file handle for advanced operations.
    pub fn file(&self) -> &DirectIoFile {
        &self.file
    }

    /// Get segment metadata by ID.
    pub fn get_segment(&self, id: u32) -> Option<&DirectIoSegment> {
        self.segments.get(id as usize)
    }

    /// Get the total number of segments in this pool.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Append an item to a small queue segment and write to file.
    ///
    /// Returns the offset where the item was written, or None if the segment is full.
    pub fn append_small_queue_item(
        &self,
        segment_id: u32,
        key: &[u8],
        value: &[u8],
        optional: &[u8],
        expire_at: u32,
        metrics: &CacheMetrics,
    ) -> Option<u32> {
        use crate::sync::Ordering;

        let segment = self.segments.get(segment_id as usize)?;

        // Calculate item size
        let header = crate::item::SmallQueueItemHeader::new(
            key.len() as u8,
            optional.len() as u8,
            value.len() as u32,
            false, // is_deleted
            false, // is_numeric
            expire_at,
        );
        let item_size = header.padded_size();

        // Try to reserve space in segment (CAS loop)
        let mut attempts = 0;
        let offset = loop {
            attempts += 1;
            if attempts > 16 {
                return None;
            }

            let current_offset = segment.write_offset.load(Ordering::Acquire);
            let new_offset = current_offset + item_size as u32;

            // Check if item fits
            if new_offset > segment.segment_size {
                return None; // Segment full
            }

            // Try to reserve space
            if segment.write_offset.compare_exchange(
                current_offset,
                new_offset,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                break current_offset;
            }
            // CAS failed, retry
            std::hint::spin_loop();
        };

        // Build item buffer
        let mut item_buf = vec![0u8; item_size];
        header.to_bytes(&mut item_buf[..crate::item::SmallQueueItemHeader::SIZE]);

        // Write optional, key, value after header
        let mut pos = crate::item::SmallQueueItemHeader::SIZE;
        item_buf[pos..pos + optional.len()].copy_from_slice(optional);
        pos += optional.len();
        item_buf[pos..pos + key.len()].copy_from_slice(key);
        pos += key.len();
        item_buf[pos..pos + value.len()].copy_from_slice(value);

        // Write to file
        if self.write_segment(segment_id, offset, &item_buf).is_err() {
            // Write failed - we've reserved space but can't use it
            // This is a problem but rare
            return None;
        }

        // Update segment stats
        segment.live_items.fetch_add(1, Ordering::Relaxed);
        segment.live_bytes.fetch_add(item_size as u32, Ordering::Relaxed);

        metrics.item_append.increment();

        Some(offset)
    }
}

// DirectIoPool implements a minimal Pool trait for hashtable operations.
// It cannot provide get_item_guard() - data must be read from disk.
// The Cache layer handles direct I/O pools specially with mandatory promotion.

impl crate::pool::Pool for DirectIoPool {
    type Segment = DirectIoSegmentWrapper;

    /// Returns None - DirectIo segments don't support direct memory access.
    /// Key verification will be skipped when this returns None.
    fn get(&self, _id: u32) -> Option<&Self::Segment> {
        // DirectIo doesn't support direct segment access
        // Returning None makes verify_key return false, which is acceptable
        // for new item insertion (the key was just written by us)
        None
    }

    fn segment_count(&self) -> usize {
        self.segments.len()
    }

    fn reserve_small_queue(&self, metrics: &CacheMetrics) -> Option<u32> {
        // Delegate to the existing method
        DirectIoPool::reserve_small_queue(self, metrics)
    }

    fn reserve_main_cache(&self, metrics: &CacheMetrics) -> Option<u32> {
        // Delegate to the existing method
        DirectIoPool::reserve_main_cache(self, metrics)
    }

    fn release(&self, id: u32, metrics: &CacheMetrics) {
        // Delegate to the existing method
        DirectIoPool::release(self, id, metrics)
    }
}

/// Wrapper type to satisfy Pool trait requirements.
/// This is never actually instantiated since get() always returns None.
pub struct DirectIoSegmentWrapper {
    _phantom: std::marker::PhantomData<()>,
}

impl crate::segment::Segment for DirectIoSegmentWrapper {
    fn id(&self) -> u32 { unreachable!("DirectIoSegmentWrapper is never instantiated") }
    fn pool_id(&self) -> u8 { unreachable!() }
    fn is_small_queue(&self) -> bool { unreachable!() }
    fn generation(&self) -> u16 { unreachable!() }
    fn state(&self) -> crate::segment::State { unreachable!() }
    fn data_len(&self) -> usize { unreachable!() }
    fn offset(&self) -> u32 { unreachable!() }
    fn live_items(&self) -> u32 { unreachable!() }
    fn live_bytes(&self) -> u32 { unreachable!() }
    fn ref_count(&self) -> u32 { unreachable!() }
    fn decr_ref_count(&self) { unreachable!() }
    fn expire_at(&self) -> clocksource::coarse::Instant { unreachable!() }
    fn set_expire_at(&self, _at: clocksource::coarse::Instant) { unreachable!() }
    fn bucket_id(&self) -> Option<u16> { unreachable!() }
    fn set_bucket_id(&self, _id: u16) { unreachable!() }
    fn clear_bucket_id(&self) { unreachable!() }
    fn merge_count(&self) -> u16 { unreachable!() }
    fn increment_merge_count(&self) { unreachable!() }
    fn increment_generation(&self) { unreachable!() }
    fn data_slice(&self, _offset: u32, _len: usize) -> Option<&[u8]> { unreachable!() }
    fn append_item(&self, _key: &[u8], _value: &[u8], _optional: &[u8], _metrics: &CacheMetrics) -> Option<u32> { unreachable!() }
    fn append_small_queue_item(&self, _key: &[u8], _value: &[u8], _optional: &[u8], _expire_at: u32, _metrics: &CacheMetrics) -> Option<u32> { unreachable!() }
    fn mark_deleted(&self, _offset: u32, _key: &[u8], _metrics: &CacheMetrics) -> Result<bool, ()> { unreachable!() }
    fn verify_key_at_offset(&self, _offset: u32, _key: &[u8], _allow_deleted: bool) -> bool { unreachable!() }
    fn get_item_guard(&self, _offset: u32, _key: &[u8]) -> Result<crate::ItemGuard<'_, Self>, crate::GetItemError> where Self: Sized { unreachable!() }
    fn prev(&self) -> Option<u32> { unreachable!() }
    fn next(&self) -> Option<u32> { unreachable!() }
    fn try_reserve(&self) -> bool { unreachable!() }
    fn try_release(&self) -> bool { unreachable!() }
    fn cas_metadata(
        &self,
        _expected_state: crate::segment::State,
        _new_state: crate::segment::State,
        _new_next: Option<u32>,
        _new_prev: Option<u32>,
        _metrics: &CacheMetrics,
    ) -> bool { unreachable!() }
    fn compact(&self, _hashtable: &crate::hashtable::Hashtable) -> u32 { unreachable!() }
    fn copy_into<S: crate::segment::Segment, F: Fn(u8) -> bool>(
        &self,
        _dest: &S,
        _hashtable: &crate::hashtable::Hashtable,
        _metrics: &CacheMetrics,
        _predicate: F,
    ) -> Option<u32> { unreachable!() }
    fn prune(
        &self,
        _hashtable: &crate::hashtable::Hashtable,
        _threshold: u8,
        _metrics: &CacheMetrics,
    ) -> (u32, u32, u32, u32) { unreachable!() }
    fn prune_with_demote<F>(
        &self,
        _hashtable: &crate::hashtable::Hashtable,
        _threshold: u8,
        _metrics: &CacheMetrics,
        _on_demote: F,
    ) -> (u32, u32, u32, u32)
    where
        F: FnMut(&[u8], &[u8], &[u8]),
    { unreachable!() }
    fn prune_collecting_for_demote(
        &self,
        _hashtable: &crate::hashtable::Hashtable,
        _threshold: u8,
        _metrics: &CacheMetrics,
    ) -> (u32, u32, u32, u32, Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>) { unreachable!() }
    fn unlink_all_items(
        &self,
        _hashtable: &crate::hashtable::Hashtable,
        _metrics: &CacheMetrics,
        _create_ghosts: bool,
    ) -> u32 { unreachable!() }
}

impl DirectIoPool {
    /// Reserve a small queue segment from the pool.
    pub fn reserve_small_queue(&self, metrics: &CacheMetrics) -> Option<u32> {
        use crate::sync::Ordering;

        match self.small_queue_free.steal() {
            crossbeam_deque::Steal::Success(segment_id) => {
                let segment = &self.segments[segment_id as usize];
                debug_assert!(
                    segment.is_small_queue,
                    "segment from small_queue_free must be small queue type"
                );

                // Transition Free -> Reserved
                let old_metadata = segment.metadata.load(Ordering::Acquire);
                let old_state = DirectIoSegment::unpack_state(old_metadata);

                if old_state != 0 {
                    // Not in Free state
                    self.small_queue_free.push(segment_id);
                    return None;
                }

                // CAS to Reserved state (1)
                let new_metadata = DirectIoSegment::pack_metadata(1, 0xFFFFFF, 0xFFFFFF);
                if segment
                    .metadata
                    .compare_exchange(
                        old_metadata,
                        new_metadata,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    self.small_queue_free.push(segment_id);
                    return None;
                }

                // Reset segment state
                segment.write_offset.store(0, Ordering::Release);
                segment.live_items.store(0, Ordering::Release);
                segment.live_bytes.store(0, Ordering::Release);
                segment.ref_count.store(0, Ordering::Release);

                metrics.segment_reserve.increment();
                metrics.segments_free.decrement();
                Some(segment_id)
            }
            crossbeam_deque::Steal::Empty => None,
            crossbeam_deque::Steal::Retry => None,
        }
    }

    /// Reserve a main cache segment from the pool.
    pub fn reserve_main_cache(&self, metrics: &CacheMetrics) -> Option<u32> {
        use crate::sync::Ordering;

        match self.main_cache_free.steal() {
            crossbeam_deque::Steal::Success(segment_id) => {
                let segment = &self.segments[segment_id as usize];
                debug_assert!(
                    !segment.is_small_queue,
                    "segment from main_cache_free must be main cache type"
                );

                // Transition Free -> Reserved
                let old_metadata = segment.metadata.load(Ordering::Acquire);
                let old_state = DirectIoSegment::unpack_state(old_metadata);

                if old_state != 0 {
                    // Not in Free state
                    self.main_cache_free.push(segment_id);
                    return None;
                }

                // CAS to Reserved state (1)
                let new_metadata = DirectIoSegment::pack_metadata(1, 0xFFFFFF, 0xFFFFFF);
                if segment
                    .metadata
                    .compare_exchange(
                        old_metadata,
                        new_metadata,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    self.main_cache_free.push(segment_id);
                    return None;
                }

                // Reset segment state
                segment.write_offset.store(0, Ordering::Release);
                segment.live_items.store(0, Ordering::Release);
                segment.live_bytes.store(0, Ordering::Release);
                segment.ref_count.store(0, Ordering::Release);

                metrics.segment_reserve.increment();
                metrics.segments_free.decrement();
                Some(segment_id)
            }
            crossbeam_deque::Steal::Empty => None,
            crossbeam_deque::Steal::Retry => None,
        }
    }

    /// Release a segment back to the appropriate free queue.
    pub fn release(&self, id: u32, metrics: &CacheMetrics) {
        use crate::sync::Ordering;

        let segment = &self.segments[id as usize];

        // Transition to Free state (0)
        let old_metadata = segment.metadata.load(Ordering::Acquire);
        let new_metadata = DirectIoSegment::pack_metadata(0, 0xFFFFFF, 0xFFFFFF);

        if segment
            .metadata
            .compare_exchange(old_metadata, new_metadata, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Increment generation for CAS token uniqueness
            segment.generation.fetch_add(1, Ordering::Relaxed);
            segment.merge_count.store(0, Ordering::Relaxed);

            metrics.segment_release.increment();
            metrics.segments_free.increment();

            if segment.is_small_queue {
                self.small_queue_free.push(id);
            } else {
                self.main_cache_free.push(id);
            }
        }
    }
}

/// Builder for DirectIoPool.
pub struct DirectIoPoolBuilder {
    pool_id: u8,
    segment_size: usize,
    heap_size: usize,
    small_queue_percent: u8,
    path: PathBuf,
}

impl DirectIoPoolBuilder {
    /// Create a new builder for a direct I/O pool.
    pub fn new(pool_id: u8, path: impl Into<PathBuf>) -> Self {
        Self {
            pool_id,
            segment_size: 4 * 1024 * 1024, // 4MB default for SSD
            heap_size: 64 * 1024 * 1024,   // 64MB default
            small_queue_percent: 10,
            path: path.into(),
        }
    }

    /// Set the segment size in bytes (default: 4MB).
    pub fn segment_size(mut self, size: usize) -> Self {
        self.segment_size = size;
        self
    }

    /// Set the total heap size in bytes (default: 64MB).
    pub fn heap_size(mut self, size: usize) -> Self {
        self.heap_size = size;
        self
    }

    /// Set the percentage of segments for the small queue (default: 10%).
    pub fn small_queue_percent(mut self, percent: u8) -> Self {
        debug_assert!(percent <= 100, "small_queue_percent must be <= 100");
        self.small_queue_percent = percent;
        self
    }

    /// Build the direct I/O pool.
    pub fn build(self) -> io::Result<DirectIoPool> {
        let num_segments = self.heap_size / self.segment_size;
        let small_queue_count = (num_segments * self.small_queue_percent as usize) / 100;
        let actual_size = num_segments * self.segment_size;

        // Create the backing file with direct I/O
        let file = DirectIoFile::open(&self.path, actual_size as u64)?;

        // Initialize segment metadata
        let mut segments = Vec::with_capacity(num_segments);
        let small_queue_free = crossbeam_deque::Injector::new();
        let main_cache_free = crossbeam_deque::Injector::new();

        for id in 0..num_segments {
            let is_small_queue = id < small_queue_count;

            let segment =
                DirectIoSegment::new(self.pool_id, is_small_queue, id as u32, self.segment_size);

            segments.push(segment);

            if is_small_queue {
                small_queue_free.push(id as u32);
            } else {
                main_cache_free.push(id as u32);
            }
        }

        Ok(DirectIoPool {
            file: Arc::new(file),
            segments,
            small_queue_free,
            main_cache_free,
            segment_size: self.segment_size,
        })
    }
}

/// Check if direct I/O is supported on this platform.
#[inline]
pub fn is_direct_io_supported() -> bool {
    cfg!(any(target_os = "linux", target_os = "macos"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_direct_io_file_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.dat");

        let file = DirectIoFile::open(&path, 1024 * 1024).unwrap();
        assert!(path.exists());
        assert_eq!(file.path(), path);
    }

    #[test]
    fn test_direct_io_file_read_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_rw.dat");

        let file = DirectIoFile::open(&path, 1024 * 1024).unwrap();

        // Write some data
        let write_data = b"Hello, Direct I/O!";
        let written = file.write_at(0, write_data).unwrap();
        assert_eq!(written, write_data.len());

        // Read it back
        let mut read_buf = vec![0u8; write_data.len()];
        let read = file.read_at(0, &mut read_buf).unwrap();
        assert_eq!(read, write_data.len());
        assert_eq!(&read_buf, write_data);
    }

    #[test]
    fn test_direct_io_file_read_write_at_offset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_offset.dat");

        let file = DirectIoFile::open(&path, 1024 * 1024).unwrap();

        // Write at offset
        let offset = 4096u64;
        let write_data = b"Data at offset";
        file.write_at(offset, write_data).unwrap();

        // Read from offset
        let mut read_buf = vec![0u8; write_data.len()];
        file.read_at(offset, &mut read_buf).unwrap();
        assert_eq!(&read_buf, write_data);

        // Data at offset 0 should be zeros
        let mut zero_buf = vec![0u8; 16];
        file.read_at(0, &mut zero_buf).unwrap();
        assert!(zero_buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_direct_io_pool_builder() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pool.dat");

        let pool = DirectIoPoolBuilder::new(1, &path)
            .segment_size(64 * 1024) // 64KB segments
            .heap_size(640 * 1024)   // 640KB = 10 segments
            .small_queue_percent(20) // 2 segments for small queue
            .build()
            .unwrap();

        assert_eq!(pool.segments.len(), 10);
        assert_eq!(pool.segment_size(), 64 * 1024);
    }

    #[test]
    fn test_direct_io_pool_reserve_release() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pool_reserve.dat");

        let pool = DirectIoPoolBuilder::new(1, &path)
            .segment_size(64 * 1024)
            .heap_size(640 * 1024) // 10 segments
            .small_queue_percent(20)
            .build()
            .unwrap();

        let metrics = CacheMetrics::new();

        // Reserve a small queue segment
        let sq_id = pool.reserve_small_queue(&metrics);
        assert!(sq_id.is_some());

        // Reserve a main cache segment
        let mc_id = pool.reserve_main_cache(&metrics);
        assert!(mc_id.is_some());

        // Release them
        pool.release(sq_id.unwrap(), &metrics);
        pool.release(mc_id.unwrap(), &metrics);

        // Should be able to reserve again
        assert!(pool.reserve_small_queue(&metrics).is_some());
        assert!(pool.reserve_main_cache(&metrics).is_some());
    }

    #[test]
    fn test_direct_io_pool_read_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pool_io.dat");

        let pool = DirectIoPoolBuilder::new(1, &path)
            .segment_size(64 * 1024)
            .heap_size(128 * 1024) // 2 segments
            .small_queue_percent(50)
            .build()
            .unwrap();

        let metrics = CacheMetrics::new();

        // Reserve a segment
        let segment_id = pool.reserve_main_cache(&metrics).unwrap();

        // Write to segment
        let data = b"Test data for segment";
        pool.write_segment(segment_id, 0, data).unwrap();

        // Read back
        let mut buf = vec![0u8; data.len()];
        pool.read_segment(segment_id, 0, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_is_direct_io_supported() {
        // Should return true on Linux and macOS
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(is_direct_io_supported());

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        assert!(!is_direct_io_supported());
    }
}
