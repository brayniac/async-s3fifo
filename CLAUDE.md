# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build
cargo build

# Build with all features
cargo build --all-features

# Run tests
cargo test

# Run tests with loom (concurrency model checking)
cargo test --features loom

# Run a single test
cargo test test_name

# Check without building
cargo check
```

## Architecture Overview

This is a high-performance, lock-free cache library (`s3`) written in Rust. It implements a segment-based cache with TTL support, designed for concurrent access without traditional locking.

### Core Components

**Hashtable** (`src/hashtable.rs`)
- Lock-free hashtable using 7-slot buckets with CAS operations
- Items are packed into 64-bit entries: `[12 bits tag][8 bits freq][24 bits segment_id][20 bits offset/8]`
- Uses ASFC (Adaptive Software Frequency Counter) for frequency-based eviction
- Offsets are stored divided by 8 (items are 8-byte aligned)

**Segments** (`src/segments/`)
- Memory-mapped segment pool (`mmap.rs`) backed by a file
- Segments have a lifecycle: Free → Reserved → Linking → Live → Sealed → Draining → Locked
- Packed metadata in single AtomicU64 for atomic state + chain pointer updates
- `INVALID_SEGMENT_ID` (0xFFFFFF) represents null segment pointers

**Items** (`src/item.rs`)
- Header format: `[4 bytes len][1 byte flags][optional: 2 magic bytes + 1 checksum]`
- Validation feature adds magic bytes (0xCA, 0xCE) and checksum for corruption detection
- Items are padded to 8-byte boundaries
- Deleted flag is set atomically via `fetch_or` on the flags byte

**Sync Primitives** (`src/sync.rs`)
- Wrappers that switch between `std::sync::atomic` and `loom::sync::atomic` based on feature flag
- Use these wrappers for all atomic operations to enable loom testing

**CAS Utilities** (`src/util.rs`)
- `retry_cas_u64`/`retry_cas_u32`: Generic CAS retry with exponential backoff
- `retry_cas_metadata`: Specialized for packed segment metadata operations
- Configurable max attempts and spin threshold

### Key Design Patterns

1. **Lock-free operations**: All hot paths use CAS loops with exponential backoff
2. **Packed atomics**: Multiple fields packed into single AtomicU64 for atomic multi-field updates
3. **Segment-based storage**: Items are appended to segments; segments are managed in TTL bucket chains
4. **Reference counting**: Segments use ref_count to prevent eviction while readers are active
5. **Zero-copy reads**: `ItemGuard` provides direct slice access into segment memory

### Feature Flags

- `validation` (default): Adds magic bytes and checksums to item headers (~3 bytes per item)
- `loom`: Enables loom-based concurrency testing (switches atomic implementations)

### Important Constants

- Max segment ID: 24 bits (0xFFFFFF)
- Max offset: 20 bits * 8 = ~8MB per segment
- Bucket slots: 7 items per hashtable bucket
- Tag bits: 12 bits for collision detection

---

## Implementation Status (S3-FIFO Multi-Tier Cache)

### Completed

**Multi-Tier Architecture** (`src/cache.rs`, `src/layer.rs`)
- `Cache` struct: Top-level public API orchestrating RAM and optional SSD layers
- `CacheLayer<P>`: Composes Pool + SmallQueue + TtlBuckets for a single tier
- Demotion callbacks wired up between layers:
  - RAM small queue cold items → SSD small queue
  - RAM main cache merge eviction → SSD small queue
  - SSD evictions → dropped (no further tier)

**Small Queue (Admission Filter)** (`src/smallqueue.rs`)
- FIFO queue using segment chain (head → tail)
- `try_append_segment`: Add new segment to tail
- `try_evict_head`: Remove oldest segment for eviction processing

**SmallQueueItemHeader** (`src/item.rs`)
- Per-item TTL stored as `expire_at: u32` (unix timestamp)
- 12 bytes with validation (vs 8 bytes for main cache ItemHeader)
- Used in small queue segments; main cache uses segment-level TTL

**Frequency-Based Promotion** (`src/layer.rs`)
- `PROMOTION_THRESHOLD = 1`: Items start with freq=1
- freq > 1 = "hot" (accessed since insertion) → promote to main cache
- freq <= 1 = "cold" (one-hit-wonder) → demote to next tier or drop

**Merge Eviction with Demotion** (`src/ttlbuckets.rs`)
- `merge_evict`: Consolidates N segments into fewer, pruning cold items
- `merge_evict_any`: Selects bucket whose head is closest to expiry (earliest expire_at)
- `evict_any`: Simple FIFO eviction, also uses earliest-expiry bucket selection
- `prune_with_demote`: Callback for items below frequency threshold
- **Merge count tracking**: Segments track how many times they've been merge destinations
  - Avoids repeatedly merging the same segments (thrashing)
  - `merge_count` stored as separate `AtomicU16` field in SliceSegment (not packed in metadata)
  - Merge starts from first segment with merge_count < head's merge_count
  - Destination's merge_count incremented after successful merge
  - Reset to 0 when segment is released to free pool
- **FIFO eviction fallback**: If bucket has too few segments for merge, falls back to FIFO eviction of head segment (handles fragmentation across buckets)

**Public Cache API** (`src/cache.rs`)
```rust
// Builder
Cache::builder()
    .ram_size(64 * 1024 * 1024)
    .ssd("/tmp/cache.dat", 1024 * 1024 * 1024)
    .ssd_backend(SsdBackendType::Auto)  // Auto, Mmap, or DirectIo
    .two_choice_hashing(true)   // Higher fill rate, ~20% slower
    .auto_promote(true)         // Promote SSD hits to RAM
    .build()?

// Basic operations
cache.set(key, value, ttl) -> Result<(), ()>
cache.get(key) -> Option<ItemGuard>
cache.delete(key) -> bool
cache.contains(key) -> bool

// Memcached-style atomic operations
cache.add(key, value, ttl) -> Result<(), CacheError>      // Insert if not exists
cache.replace(key, value, ttl) -> Result<(), CacheError>  // Update if exists
cache.gets(key) -> Option<(ItemGuard, u64)>               // Get with CAS token
cache.cas(key, value, cas_token, ttl) -> Result<(), CacheError>  // Compare-and-swap

// Eviction (usually automatic, can be called manually)
cache.evict_ram_small_queue().await
cache.merge_evict_ram().await -> Option<usize>
```

**CacheError** (`src/lib.rs`)
```rust
pub enum CacheError {
    KeyExists,     // ADD failed - key already present
    KeyNotFound,   // REPLACE/CAS failed - key missing
    CasMismatch,   // CAS failed - value was modified
    HashTableFull, // Buckets full
    StorageFull,   // No segments available
}
```

**Two-Choice Hashing** (`src/hashtable.rs`)
- Configurable via `CacheBuilder::two_choice_hashing(bool)`
- Single-choice (default): ~85% fill rate, faster lookups
- Two-choice: ~95% fill rate, ~20% slower (checks two buckets)
- Load balancing: Inserts go to less-loaded of two candidate buckets

**CAS Tokens** (`src/cache.rs`, `src/segment/slice.rs`)
- Location-based tokens: `(generation, pool_id, segment_id, offset)`
- Segment generation counter prevents ABA problem (incremented on segment reuse)
- Token changes when item moves (update, promotion, compaction)
- 64-bit token fits memcached compatibility

**Auto-Promotion** (`src/cache.rs`)
- Configurable via `CacheBuilder::auto_promote(bool)`
- When enabled, SSD hits are copied to RAM small queue
- Item must prove itself hot again (frequency reset to 1)
- Original SSD copy cleaned up on eviction

**SSD Backend Selection** (`src/cache.rs`, `src/pool/direct_io.rs`)
- Configurable via `CacheBuilder::ssd_backend(SsdBackendType)` or `.force_mmap()`
- Three backend options:
  - `SsdBackendType::Auto` (default): DirectIo on Linux/macOS, Mmap elsewhere
  - `SsdBackendType::Mmap`: Memory-mapped file, uses OS page cache
  - `SsdBackendType::DirectIo`: O_DIRECT (Linux) or F_NOCACHE (macOS)
- DirectIo benefits:
  - Predictable memory usage (no page cache interference)
  - RAM tier size = actual RAM used for cache
  - Your frequency counters control what's hot, not kernel LRU
- DirectIo implementation status:
  - DirectIoFile: Opens files with O_DIRECT/F_NOCACHE flags (done)
  - DirectIoPool: Manages segment metadata and file I/O (done)
  - Read path with mandatory promotion: Not yet implemented (uses mmap fallback)
  - Write path: Not yet implemented (uses mmap fallback)

**Separate Free Lists for Segment Types** (`src/pool/`)
- Pool trait now has `reserve_small_queue()` and `reserve_main_cache()` methods
- Both `MemoryPool` and `MmapPool` maintain two separate free queues:
  - `small_queue_free`: For small queue segments (admission filter)
  - `main_cache_free`: For main cache segments (TTL buckets)
- Segments are added to the correct queue during pool initialization based on `is_small_queue` flag
- `release()` automatically returns segments to the appropriate queue based on segment type
- Eliminates inefficient reserve-check-release pattern that could fail even when correct segment types were available

**Pool Builder Configuration** (`src/pool/`)
- `MemoryPoolBuilder` and `MmapPoolBuilder` support:
  - `segment_size(size)`: Set segment size in bytes (default: 1MB)
  - `heap_size(size)`: Set total heap size (default: 64MB)
  - `small_queue_percent(percent)`: Percentage of segments for admission queue (default: 10%)
- Number of segments is calculated as `heap_size / segment_size`

**TTL Enforcement** (`src/segment/slice.rs`, `src/ttlbuckets.rs`)
- **Read path**: Both small queue (per-item TTL) and main cache (segment-level TTL) check expiration in `get_item_guard()`. Expired items return `GetItemError::ItemDeleted`.
- **On-demand cleanup**: `TtlBuckets::try_expire()` scans buckets for expired head segments when no free segments available
- Expired segments transition: Draining -> Locked -> Free -> Reserved (with stats reset)
- No background scanning needed - expiration checked on read, cleanup on memory pressure

**Ghost Entries** (`src/hashtable.rs`)
- Track recently evicted keys to detect thrashing patterns and aid admission decisions
- Ghost entry format: Same as regular item but with `GHOST_SEGMENT_ID = 0x3FFFFF` as sentinel
  - Preserves tag (12 bits) and frequency (8 bits), location fields are sentinel values
  - `is_ghost()`, `pack_ghost()`, `to_ghost()` helper functions
- **When ghosts are created** (via `unlink_item_to_ghost`):
  - Small queue cold item eviction: Helps detect if "one-hit-wonders" actually return
  - Main cache prune during merge eviction: Tracks items evicted due to low frequency
- **When ghosts are NOT created**:
  - Expired items (TTL expiration): Item naturally aged out
  - Explicit user delete: User wants item gone
  - Failed promotions: Can't distinguish append failure from relink race
  - Demotions to next tier: Item still exists, just moved
- **On insert** (`link_item`):
  - First pass: Also tracks ghost entries with matching tag
  - If ghost found, inherits its frequency (helps with re-admission decisions)
  - Third pass: If bucket full with real items, evicts any ghost to make space
- **Ghost-based admission** (`set()`, `add()`):
  - On insert, check if key has ghost entry with freq > 1
  - If ghost was accessed (freq > 1): insert directly to main cache, bypassing small queue
  - Increments `ghost_promote` metric on successful ghost-based promotion
- **Metrics**: `ghost_create`, `ghost_hit`, `ghost_evict`, `ghost_promote`

**Frequency-Based Routing for Replace** (`src/cache.rs`)
- `replace()` routes items based on existing item's frequency:
  - freq > 1 (item was accessed): replacement goes to main cache (TTL buckets)
  - freq <= 1 (item not accessed): replacement goes through small queue
- Ensures replaced items follow the same admission policy as new items
- Uses `get_frequency()` to check item's hotness without incrementing frequency

### Still TODO

1. **DirectIo backend read/write paths**: DirectIoFile and DirectIoPool are implemented, but the Cache layer doesn't yet use them for actual I/O. Currently falls back to mmap. Needed:
   - Read path: Read item from disk into temp buffer, promote to RAM, return RAM ItemGuard
   - Write path: Write items to disk segments instead of mmap'd memory
   - Eviction: Handle DirectIo segments in small queue and TTL bucket eviction

2. **Background eviction task**: No automatic triggering of eviction based on pool utilization. Currently eviction only happens when insert fails or is manually called. Consider adding a background task that monitors free segment count.

3. **Integration tests**: No end-to-end tests exercising the full Cache API with actual data flow through both RAM and SSD tiers.

4. **`set()` error handling**: `set()` still returns `Result<(), ()>`. Could use `CacheError` for consistency with `add`/`replace`/`cas`.

5. **INCR/DECR operations**: Deferred due to concerns about hot counters on SSD causing thrashing with CAS loops.

### Key Design Decisions

- **Segment reuse**: Merge eviction returns segments to pool free list; any bucket can use reclaimed segments
- **Earliest-expiry eviction**: Both `evict_any` and `merge_evict_any` select the bucket whose head segment is closest to expiry. This prioritizes evicting items that are about to expire anyway, preserving items with more remaining useful life. Improves hit rate for workloads with mixed TTLs.
- **Per-item vs segment TTL**: Small queue uses per-item TTL (items have varying lifetimes); main cache uses segment-level TTL (all items in segment share TTL bucket's expiration)
- **Async eviction**: Eviction methods are async to allow TTL bucket append retries with backoff
