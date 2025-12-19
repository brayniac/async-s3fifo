mod cache;
pub mod hashtable;
mod hugepage;
mod item;
mod layer;
mod metrics;
pub mod pool;
pub mod segment;
mod smallqueue;
mod sync;
mod ttlbuckets;
mod util;

// Public API exports
pub use cache::{Cache, CacheBuilder, SsdBackendType};
pub use hugepage::HugepageSize;
pub use item::ItemGuard;
pub use metrics::CacheMetrics;
pub use pool::{
    MemoryPool, MemoryPoolBuilder, MmapPool, MmapPoolBuilder, Pool,
    DirectIoPool, DirectIoPoolBuilder, DirectIoFile, is_direct_io_supported,
};
pub use segment::{Segment, SliceSegment};

impl CasToken {
    /// Create a CasToken from its internal representation.
    ///
    /// This is primarily for internal use. Users should obtain tokens via `Cache::gets`.
    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Get the raw internal value (for internal use only).
    pub(crate) fn into_raw(self) -> u64 {
        self.0
    }
}

/// Error types for get_item operations
#[derive(Debug, PartialEq)]
pub enum GetItemError {
    /// Item not found in hashtable
    NotFound,
    /// Item has been marked as deleted
    ItemDeleted,
    /// Key doesn't match (hash collision)
    KeyMismatch,
    /// Segment is being cleared/removed
    SegmentNotAccessible,
    /// Invalid offset or corrupted data
    InvalidOffset,
    /// Buffer provided is too small for the item
    BufferTooSmall,
}

/// Error types for cache write operations (SET, ADD, REPLACE, CAS)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    /// Key already exists (returned by ADD when key is present)
    KeyExists,
    /// Key not found (returned by REPLACE/CAS when key is missing)
    KeyNotFound,
    /// CAS token mismatch - value was modified by another client (returned by CAS)
    CasMismatch,
    /// Hashtable bucket(s) are full and cannot accept new items
    HashTableFull,
    /// Storage is full and eviction could not free space
    StorageFull,
}

/// Opaque CAS (compare-and-swap) token for atomic updates.
///
/// Obtained from [`Cache::gets`] and used with [`Cache::cas`] to perform
/// atomic read-modify-write operations. The token changes whenever the
/// item is modified, moved, or evicted.
///
/// # Example
///
/// ```ignore
/// if let Some((guard, cas_token)) = cache.gets(b"counter") {
///     let value: u64 = parse(guard.value());
///     let new_value = (value + 1).to_string();
///     match cache.cas(b"counter", new_value.as_bytes(), cas_token, ttl).await {
///         Ok(()) => println!("Updated!"),
///         Err(CacheError::CasMismatch) => println!("Retry needed"),
///         Err(e) => println!("Error: {:?}", e),
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CasToken(u64);

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
