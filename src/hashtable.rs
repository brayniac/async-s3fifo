use crate::segment::Segment;
use crate::pool::Pool;
use crate::util::*;
use crate::hugepage::{HugepageAllocation, HugepageSize, allocate};
use ahash::RandomState;
use crate::sync::*;

pub struct Hashtable {
    pub(crate) hash_builder: Box<RandomState>,
    /// Mmap-backed storage for hash buckets.
    pub(crate) allocation: HugepageAllocation,
    /// Number of buckets (2^power).
    pub(crate) num_buckets: usize,
    pub(crate) mask: u64,
    /// If true, use two-choice hashing for higher fill rates (~95%).
    /// If false, use single-choice hashing for higher throughput (~30% faster).
    pub(crate) two_choice: bool,
}

impl Hashtable {
    /// Create a new hashtable with single-choice hashing (default, faster).
    pub fn new(power: u8) -> Result<Self, std::io::Error> {
        Self::with_hugepage_size(power, false, HugepageSize::None)
    }

    /// Create a new hashtable with configurable two-choice hashing.
    ///
    /// - `two_choice = false`: Single-choice hashing, ~30% faster, ~85% fill rate
    /// - `two_choice = true`: Two-choice hashing, higher fill rate (~95%)
    pub fn with_two_choice(power: u8, two_choice: bool) -> Result<Self, std::io::Error> {
        Self::with_hugepage_size(power, two_choice, HugepageSize::None)
    }

    /// Create a new hashtable with specified hugepage size preference.
    pub fn with_hugepage_size(power: u8, two_choice: bool, hugepage_size: HugepageSize) -> Result<Self, std::io::Error> {
        if power < 4 {
            panic!("power too low");
        }

        // Use fixed seeds in tests for deterministic behavior, random seeds in production
        #[cfg(test)]
        let hash_builder = RandomState::with_seeds(
            0xbb8c484891ec6c86,
            0x0522a25ae9c769f9,
            0xeed2797b9571bc75,
            0x4feb29c1fbbd59d0,
        );
        #[cfg(not(test))]
        let hash_builder = RandomState::new();

        let num_buckets = 1_usize << power;
        let mask = (num_buckets as u64) - 1;

        // Calculate size needed for all buckets
        // Each Hashbucket is 64 bytes (8 * AtomicU64)
        let alloc_size = num_buckets * std::mem::size_of::<Hashbucket>();

        // Allocate mmap'd memory with hugepage support
        let allocation = allocate(alloc_size, hugepage_size)?;

        // Initialize all buckets to zero (already done by mmap, but be explicit)
        // The prefault in hugepage::allocate already zeroed the memory via write_volatile
        // AtomicU64::new(0) is equivalent to all-zeros representation

        Ok(Self {
            hash_builder: Box::new(hash_builder),
            allocation,
            num_buckets,
            mask,
            two_choice,
        })
    }

    /// Get a reference to a bucket by index.
    #[inline]
    pub(crate) fn bucket(&self, index: usize) -> &Hashbucket {
        debug_assert!(index < self.num_buckets);
        unsafe {
            let ptr = self.allocation.as_ptr() as *const Hashbucket;
            &*ptr.add(index)
        }
    }

    /// Compute primary and secondary bucket indices using two-choice hashing.
    ///
    /// Uses cheap hash derivation: secondary = hash XOR (hash >> 32)
    /// This provides good distribution without computing a second hash.
    #[inline]
    fn bucket_indices(&self, hash: u64) -> (usize, usize) {
        let primary = (hash & self.mask) as usize;
        let secondary = ((hash ^ (hash >> 32)) & self.mask) as usize;
        (primary, secondary)
    }

    /// Count the number of occupied slots (non-empty, non-ghost) in a bucket.
    #[inline]
    fn count_occupied(&self, bucket_index: usize) -> usize {
        let bucket = self.bucket(bucket_index);
        let mut count = 0;
        for slot in &bucket.items {
            let packed = slot.load(Ordering::Relaxed);
            if packed != 0 && !Hashbucket::is_ghost(packed) {
                count += 1;
            }
        }
        count
    }

    /// Look up an item in the hashtable by key.
    ///
    /// Returns the (segment_id, offset) tuple if the item is found, None otherwise.
    ///
    /// # Process (Two-Choice Hashing)
    /// 1. Hash the key to get bucket indices and tag
    /// 2. Search the primary bucket first (likely hit due to insertion policy)
    /// 3. If not found, search the secondary bucket
    /// 4. For each matching tag, verify the actual key in the segment
    /// 5. Return the first verified match
    ///
    /// # Concurrent Safety
    /// - Safe to call concurrently with other get/insert/unlink operations
    /// - May return None if item is deleted during the lookup
    /// - May return stale location if item is moved/deleted after lookup
    ///
    /// # Parameters
    /// - `key`: The key to search for
    /// - `segments`: Reference to segments for key verification
    ///
    /// # Returns
    /// - `Some((segment_id, offset))` if found and key verified
    /// - `None` if not found or key verification failed
    pub fn get<P: Pool>(&self, key: &[u8], segments: &P) -> Option<(u32, u32)> {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Search primary bucket first
        if let Some(result) = self.search_bucket_for_get(primary, tag, key, segments) {
            return Some(result);
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && let Some(result) = self.search_bucket_for_get(secondary, tag, key, segments)
        {
            return Some(result);
        }

        None
    }

    /// Search a single bucket for a key during get operation.
    /// Updates frequency on hit.
    fn search_bucket_for_get<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> Option<(u32, u32)> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            // Skip empty slots and ghost entries
            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            // Check if tag matches
            if Hashbucket::item_tag(packed) == tag {
                let segment_id = Hashbucket::item_segment_id(packed);
                let offset = Hashbucket::item_offset(packed);

                // Verify the key actually matches by reading from segment
                if self.verify_key(key, segment_id, offset, segments, false) {
                    // Update frequency using ASFC algorithm for frequency-based eviction
                    // Skip update if frequency is already saturated (127) to avoid
                    // unnecessary RNG calls and CAS contention on hot items
                    let freq = Hashbucket::item_freq(packed);
                    if freq < 127
                        && let Some(new_packed) = Hashbucket::try_update_freq(packed, freq)
                    {
                        // Best-effort update - if CAS fails, another thread updated it
                        let _ = slot.compare_exchange(
                            packed,
                            new_packed,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    }

                    return Some((segment_id, offset));
                }
                // Tag matched but key didn't - continue searching for another match
            }
        }

        None
    }

    /// Check if a key has a ghost entry and return its frequency.
    ///
    /// Ghost entries track the frequency of recently evicted items. If a ghost
    /// exists with high frequency (> 1), it indicates the item was accessed
    /// between eviction and re-insertion, suggesting it should go directly
    /// to the main cache (bypassing the small queue).
    ///
    /// Checks both primary and secondary buckets for ghosts.
    ///
    /// Returns `Some(freq)` if a ghost entry exists for this key, `None` otherwise.
    pub fn get_ghost_freq(&self, key: &[u8]) -> Option<u8> {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if let Some(freq) = self.search_bucket_for_ghost(primary, tag) {
            return Some(freq);
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && let Some(freq) = self.search_bucket_for_ghost(secondary, tag)
        {
            return Some(freq);
        }

        None
    }

    /// Search a single bucket for a ghost entry with matching tag.
    fn search_bucket_for_ghost(&self, bucket_index: usize, tag: u16) -> Option<u8> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            // Skip empty slots and non-ghost entries
            if packed == 0 || !Hashbucket::is_ghost(packed) {
                continue;
            }

            // Check if tag matches
            if Hashbucket::item_tag(packed) == tag {
                return Some(Hashbucket::item_freq(packed));
            }
        }

        None
    }

    /// Check if a key exists in the cache without incrementing its frequency.
    ///
    /// This is useful for implementing conditional operations like ADD (only write if key
    /// doesn't exist) and REPLACE (only write if key exists) where checking existence
    /// should not affect the item's "hotness".
    ///
    /// Checks both primary and secondary buckets.
    ///
    /// # Returns
    /// `true` if the key exists in the cache, `false` otherwise.
    pub fn contains<P: Pool>(&self, key: &[u8], segments: &P) -> bool {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if self.search_bucket_for_contains(primary, tag, key, segments) {
            return true;
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && self.search_bucket_for_contains(secondary, tag, key, segments)
        {
            return true;
        }

        false
    }

    /// Get the frequency of an item by key without incrementing it.
    ///
    /// This is useful for implementing REPLACE semantics where we need to know
    /// the item's current frequency to decide whether to route to small queue
    /// or main cache.
    ///
    /// # Returns
    /// `Some(frequency)` if the key exists, `None` if not found.
    pub fn get_frequency<P: Pool>(&self, key: &[u8], segments: &P) -> Option<u8> {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if let Some(freq) = self.search_bucket_for_frequency_by_key(primary, tag, key, segments) {
            return Some(freq);
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && let Some(freq) = self.search_bucket_for_frequency_by_key(secondary, tag, key, segments)
        {
            return Some(freq);
        }

        None
    }

    /// Search a single bucket for key existence (without updating frequency).
    fn search_bucket_for_contains<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            // Skip empty slots and ghost entries
            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            // Check if tag matches
            if Hashbucket::item_tag(packed) == tag {
                let segment_id = Hashbucket::item_segment_id(packed);
                let offset = Hashbucket::item_offset(packed);

                // Verify the key actually matches by reading from segment
                // Do NOT update frequency - this is just an existence check
                if self.verify_key(key, segment_id, offset, segments, false) {
                    return true;
                }
                // Tag matched but key didn't - continue searching for another match
            }
        }

        false
    }

    /// Search a single bucket for frequency by key (without updating frequency).
    fn search_bucket_for_frequency_by_key<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> Option<u8> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            // Skip empty slots and ghost entries
            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            // Check if tag matches
            if Hashbucket::item_tag(packed) == tag {
                let segment_id = Hashbucket::item_segment_id(packed);
                let offset = Hashbucket::item_offset(packed);

                // Verify the key actually matches by reading from segment
                // Do NOT update frequency - this is just a lookup
                if self.verify_key(key, segment_id, offset, segments, false) {
                    return Some(Hashbucket::item_freq(packed));
                }
                // Tag matched but key didn't - continue searching for another match
            }
        }

        None
    }

    /// Get the frequency of an item at a known location.
    ///
    /// This is used during merge eviction to look up item frequencies for pruning decisions.
    /// The caller must provide the correct segment_id and offset for the key.
    ///
    /// Checks both primary and secondary buckets.
    ///
    /// # Returns
    /// `Some(frequency)` if the item is found, `None` if not in the hashtable.
    pub fn get_item_frequency(&self, key: &[u8], segment_id: u32, offset: u32) -> Option<u8> {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if let Some(freq) = self.search_bucket_for_frequency(primary, tag, segment_id, offset) {
            return Some(freq);
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && let Some(freq) = self.search_bucket_for_frequency(secondary, tag, segment_id, offset)
        {
            return Some(freq);
        }

        None
    }

    /// Search a single bucket for an item's frequency by location.
    fn search_bucket_for_frequency(
        &self,
        bucket_index: usize,
        tag: u16,
        segment_id: u32,
        offset: u32,
    ) -> Option<u8> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            // Skip empty slots and ghost entries
            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            if Hashbucket::item_tag(packed) == tag
                && Hashbucket::item_segment_id(packed) == segment_id
                && Hashbucket::item_offset(packed) == offset
            {
                return Some(Hashbucket::item_freq(packed));
            }
        }

        None
    }

    /// Verify that a key matches the item at the given segment/offset.
    ///
    /// This is a helper to check that a tag match is actually the right key
    /// (since tags are only 12 bits and can collide).
    ///
    /// # Parameters
    /// * `allow_deleted` - If true, matches deleted items (used during slot reservation);
    ///   if false, returns false for deleted items (used during get operations)
    fn verify_key<P: Pool>(&self, key: &[u8], segment_id: u32, offset: u32, pool: &P, allow_deleted: bool) -> bool {
        match pool.get(segment_id) {
            Some(segment) => segment.verify_key_at_offset(offset, key, allow_deleted),
            None => false,
        }
    }

    /// Link an item into the hashtable using two-choice hashing.
    ///
    /// Two-choice hashing: compute two bucket indices from the hash,
    /// insert into the bucket with fewer entries. This achieves higher
    /// fill rates (~95%) compared to single-choice (~85%).
    ///
    /// # Returns
    /// - `Ok(None)` if item was successfully linked with no replacement
    /// - `Ok(Some((old_seg_id, old_offset)))` if successfully linked and replaced existing item
    /// - `Err(())` if insertion failed (both buckets full)
    pub fn link_item<P: Pool>(
        &self,
        key: &[u8],
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<Option<(u32, u32)>, ()> {
        // Validate pool_id, segment_id and offset fit in their bit fields
        if pool_id > 3 {
            panic!("pool_id {} exceeds 2-bit limit", pool_id);
        }
        if segment_id > 0x3FFFFF {
            panic!("segment_id {} exceeds 22-bit limit", segment_id);
        }
        if offset > 0xFFFFF << 3 {
            panic!("offset {} exceeds limit (max {})", offset, 0xFFFFF << 3);
        }
        if offset & 0x7 != 0 {
            panic!("offset {} is not 8-byte aligned", offset);
        }

        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Phase 1: Check bucket(s) for existing key (replace if found)
        if let Some(result) = self.try_replace_existing_in_bucket(
            primary,
            tag,
            key,
            pool_id,
            segment_id,
            offset,
            segments,
            metrics,
        ) {
            return result;
        }
        if self.two_choice
            && secondary != primary
            && let Some(result) = self.try_replace_existing_in_bucket(
                secondary,
                tag,
                key,
                pool_id,
                segment_id,
                offset,
                segments,
                metrics,
            )
        {
            return result;
        }

        // Phase 2: Check bucket(s) for matching ghost (inherit frequency)
        if let Some(result) = self.try_replace_ghost_in_bucket(
            primary,
            tag,
            pool_id,
            segment_id,
            offset,
            metrics,
        ) {
            return result;
        }
        if self.two_choice
            && secondary != primary
            && let Some(result) = self.try_replace_ghost_in_bucket(
                secondary,
                tag,
                pool_id,
                segment_id,
                offset,
                metrics,
            )
        {
            return result;
        }

        // Phase 3: Insert into empty slot
        let new_packed = Hashbucket::pack_item(tag, 1, pool_id, segment_id, offset);

        if self.two_choice && secondary != primary {
            // Two-choice: insert into bucket with fewer entries
            let primary_count = self.count_occupied(primary);
            let secondary_count = self.count_occupied(secondary);

            let (first, second) = if primary_count <= secondary_count {
                (primary, secondary)
            } else {
                (secondary, primary)
            };

            if let Some(result) =
                self.try_insert_empty_slot(first, tag, new_packed, key, segments, metrics)
            {
                return result;
            }
            if let Some(result) =
                self.try_insert_empty_slot(second, tag, new_packed, key, segments, metrics)
            {
                return result;
            }

            // Try to evict ghost in either bucket
            if let Some(result) = self.try_evict_ghost_in_bucket(first, new_packed, metrics) {
                return result;
            }
            if let Some(result) = self.try_evict_ghost_in_bucket(second, new_packed, metrics) {
                return result;
            }
        } else {
            // Single-choice: only use primary bucket
            if let Some(result) =
                self.try_insert_empty_slot(primary, tag, new_packed, key, segments, metrics)
            {
                return result;
            }
            if let Some(result) = self.try_evict_ghost_in_bucket(primary, new_packed, metrics) {
                return result;
            }
        }

        // Bucket(s) full with real items
        metrics.hashtable_full.increment();
        Err(())
    }

    /// Link an item into the hashtable only if the key does not already exist.
    ///
    /// This implements atomic ADD semantics: the operation succeeds only if no
    /// existing entry for this key is found. Two concurrent ADDs for the same key
    /// will result in exactly one success and one KeyExists error.
    ///
    /// # Returns
    /// - `Ok(())` if item was successfully linked (key was not present)
    /// - `Err(KeyExists)` if the key already exists in the hashtable
    /// - `Err(HashTableFull)` if insertion failed (buckets full)
    pub fn link_item_if_absent<P: Pool>(
        &self,
        key: &[u8],
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(), crate::CacheError> {
        // Validate pool_id, segment_id and offset fit in their bit fields
        if pool_id > 3 {
            panic!("pool_id {} exceeds 2-bit limit", pool_id);
        }
        if segment_id > 0x3FFFFF {
            panic!("segment_id {} exceeds 22-bit limit", segment_id);
        }
        if offset > 0xFFFFF << 3 {
            panic!("offset {} exceeds limit (max {})", offset, 0xFFFFF << 3);
        }
        if offset & 0x7 != 0 {
            panic!("offset {} is not 8-byte aligned", offset);
        }

        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Phase 1: Check bucket(s) for existing key - FAIL if found (ADD semantics)
        if self.check_key_exists_in_bucket(primary, tag, key, segments) {
            return Err(crate::CacheError::KeyExists);
        }
        if self.two_choice
            && secondary != primary
            && self.check_key_exists_in_bucket(secondary, tag, key, segments)
        {
            return Err(crate::CacheError::KeyExists);
        }

        // Phase 2: Check bucket(s) for matching ghost (inherit frequency)
        if let Some(result) = self.try_replace_ghost_in_bucket(
            primary,
            tag,
            pool_id,
            segment_id,
            offset,
            metrics,
        ) {
            return result.map(|_| ()).map_err(|_| crate::CacheError::HashTableFull);
        }
        if self.two_choice
            && secondary != primary
            && let Some(result) = self.try_replace_ghost_in_bucket(
                secondary,
                tag,
                pool_id,
                segment_id,
                offset,
                metrics,
            )
        {
            return result.map(|_| ()).map_err(|_| crate::CacheError::HashTableFull);
        }

        // Phase 3: Insert into empty slot
        let new_packed = Hashbucket::pack_item(tag, 1, pool_id, segment_id, offset);

        if self.two_choice && secondary != primary {
            let primary_count = self.count_occupied(primary);
            let secondary_count = self.count_occupied(secondary);

            let (first, second) = if primary_count <= secondary_count {
                (primary, secondary)
            } else {
                (secondary, primary)
            };

            // Try empty slot, but re-check for key existence after CAS
            if let Some(result) = self.try_insert_empty_slot_if_absent(
                first, tag, new_packed, key, segments, metrics,
            ) {
                return result;
            }
            if let Some(result) = self.try_insert_empty_slot_if_absent(
                second, tag, new_packed, key, segments, metrics,
            ) {
                return result;
            }

            // Try to evict ghost in either bucket
            if let Some(result) = self.try_evict_ghost_in_bucket(first, new_packed, metrics) {
                return result.map(|_| ()).map_err(|_| crate::CacheError::HashTableFull);
            }
            if let Some(result) = self.try_evict_ghost_in_bucket(second, new_packed, metrics) {
                return result.map(|_| ()).map_err(|_| crate::CacheError::HashTableFull);
            }
        } else {
            if let Some(result) = self.try_insert_empty_slot_if_absent(
                primary, tag, new_packed, key, segments, metrics,
            ) {
                return result;
            }
            if let Some(result) = self.try_evict_ghost_in_bucket(primary, new_packed, metrics) {
                return result.map(|_| ()).map_err(|_| crate::CacheError::HashTableFull);
            }
        }

        // Bucket(s) full with real items
        metrics.hashtable_full.increment();
        Err(crate::CacheError::HashTableFull)
    }

    /// Link an item into the hashtable only if the key already exists.
    ///
    /// This implements atomic REPLACE semantics: the operation succeeds only if
    /// an existing entry for this key is found and replaced. Two concurrent
    /// REPLACEs for the same key will both succeed (last one wins).
    ///
    /// # Returns
    /// - `Ok(Some((old_seg_id, old_offset)))` if key existed and was replaced
    /// - `Err(KeyNotFound)` if the key does not exist in the hashtable
    pub fn link_item_if_present<P: Pool>(
        &self,
        key: &[u8],
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Result<(u32, u32), crate::CacheError> {
        // Validate pool_id, segment_id and offset fit in their bit fields
        if pool_id > 3 {
            panic!("pool_id {} exceeds 2-bit limit", pool_id);
        }
        if segment_id > 0x3FFFFF {
            panic!("segment_id {} exceeds 22-bit limit", segment_id);
        }
        if offset > 0xFFFFF << 3 {
            panic!("offset {} exceeds limit (max {})", offset, 0xFFFFF << 3);
        }
        if offset & 0x7 != 0 {
            panic!("offset {} is not 8-byte aligned", offset);
        }

        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Only check for existing key - do NOT insert into empty slots or ghosts
        if let Some(result) = self.try_replace_existing_for_replace(
            primary,
            tag,
            key,
            pool_id,
            segment_id,
            offset,
            segments,
            metrics,
        ) {
            return result;
        }
        if self.two_choice
            && secondary != primary
            && let Some(result) = self.try_replace_existing_for_replace(
                secondary,
                tag,
                key,
                pool_id,
                segment_id,
                offset,
                segments,
                metrics,
            )
        {
            return result;
        }

        // Key not found - REPLACE fails
        Err(crate::CacheError::KeyNotFound)
    }

    /// Check if a key exists in a bucket (for ADD operation).
    fn check_key_exists_in_bucket<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if current == 0 || Hashbucket::is_ghost(current) {
                continue;
            }

            if Hashbucket::item_tag(current) == tag {
                let seg_id = Hashbucket::item_segment_id(current);
                let off = Hashbucket::item_offset(current);

                if self.verify_key(key, seg_id, off, segments, false) {
                    return true;
                }
            }
        }

        false
    }

    /// Try to insert into an empty slot, checking for key existence after CAS.
    /// Returns Err(KeyExists) if another thread inserted the same key.
    fn try_insert_empty_slot_if_absent<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        new_packed: u64,
        key: &[u8],
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<(), crate::CacheError>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if current == 0 {
                match slot.compare_exchange(0, new_packed, Ordering::Release, Ordering::Acquire) {
                    Ok(_) => {
                        metrics.hashtable_link.increment();

                        // Check for duplicates - if found, another thread inserted the same key
                        // We need to back out our insertion
                        if self.check_for_duplicate_after_insert(bucket_index, slot, tag, key, segments) {
                            // Another entry exists for this key - remove ours and return KeyExists
                            let _ = slot.compare_exchange(
                                new_packed,
                                0,
                                Ordering::Release,
                                Ordering::Relaxed,
                            );
                            return Some(Err(crate::CacheError::KeyExists));
                        }

                        return Some(Ok(()));
                    }
                    Err(new_current) => {
                        // Check if another thread inserted our key
                        if new_current != 0
                            && !Hashbucket::is_ghost(new_current)
                            && Hashbucket::item_tag(new_current) == tag
                        {
                            let new_seg_id = Hashbucket::item_segment_id(new_current);
                            let new_offset = Hashbucket::item_offset(new_current);
                            if self.verify_key(key, new_seg_id, new_offset, segments, false) {
                                // Another thread inserted our key - return KeyExists
                                return Some(Err(crate::CacheError::KeyExists));
                            }
                        }
                        // Slot taken by different item, continue searching
                        continue;
                    }
                }
            }
        }

        None
    }

    /// Check if there's a duplicate entry for this key after we inserted.
    fn check_for_duplicate_after_insert<P: Pool>(
        &self,
        bucket_index: usize,
        our_slot: &AtomicU64,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for other_slot in &bucket.items {
            if std::ptr::eq(our_slot, other_slot) {
                continue;
            }

            let other_packed = other_slot.load(Ordering::Acquire);
            if other_packed == 0 || Hashbucket::is_ghost(other_packed) {
                continue;
            }

            if Hashbucket::item_tag(other_packed) == tag {
                let other_seg_id = Hashbucket::item_segment_id(other_packed);
                let other_offset = Hashbucket::item_offset(other_packed);

                if self.verify_key(key, other_seg_id, other_offset, segments, false) {
                    // Found another entry for this key
                    return true;
                }
            }
        }

        // Also check secondary bucket if two-choice is enabled
        if self.two_choice {
            let hash = self.hash_builder.hash_one(key);
            let (primary, secondary) = self.bucket_indices(hash);
            let other_bucket_index = if bucket_index == primary { secondary } else { primary };

            if other_bucket_index != bucket_index {
                let other_bucket = self.bucket(other_bucket_index);
                for slot in &other_bucket.items {
                    let packed = slot.load(Ordering::Acquire);
                    if packed == 0 || Hashbucket::is_ghost(packed) {
                        continue;
                    }

                    if Hashbucket::item_tag(packed) == tag {
                        let seg_id = Hashbucket::item_segment_id(packed);
                        let off = Hashbucket::item_offset(packed);

                        if self.verify_key(key, seg_id, off, segments, false) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Try to replace an existing key in a bucket for REPLACE operation.
    /// Returns the old location on success.
    fn try_replace_existing_for_replace<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<(u32, u32), crate::CacheError>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if current == 0 || Hashbucket::is_ghost(current) {
                continue;
            }

            if Hashbucket::item_tag(current) == tag {
                let old_segment_id = Hashbucket::item_segment_id(current);
                let old_offset = Hashbucket::item_offset(current);

                if self.verify_key(key, old_segment_id, old_offset, segments, false) {
                    // Found matching key - replace it, preserving frequency
                    // Frequency tracks how often the KEY is accessed (reads AND writes).
                    let old_freq = Hashbucket::item_freq(current);
                    let new_packed = Hashbucket::pack_item(tag, old_freq, pool_id, segment_id, offset);
                    match slot.compare_exchange(
                        current,
                        new_packed,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            metrics.hashtable_link.increment();
                            return Some(Ok((old_segment_id, old_offset)));
                        }
                        Err(_) => {
                            // CAS failed - retry the whole operation
                            return Some(self.link_item_if_present(
                                key, pool_id, segment_id, offset, segments, metrics,
                            ));
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to replace an existing key in a bucket.
    /// Returns Some(result) if replacement succeeded or needs retry, None to continue.
    fn try_replace_existing_in_bucket<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        key: &[u8],
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<Option<(u32, u32)>, ()>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if current == 0 || Hashbucket::is_ghost(current) {
                continue;
            }

            if Hashbucket::item_tag(current) == tag {
                let old_segment_id = Hashbucket::item_segment_id(current);
                let old_offset = Hashbucket::item_offset(current);

                if self.verify_key(key, old_segment_id, old_offset, segments, false) {
                    // Found matching key - replace it
                    let new_packed = Hashbucket::pack_item(tag, 1, pool_id, segment_id, offset);
                    match slot.compare_exchange(
                        current,
                        new_packed,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            metrics.hashtable_link.increment();
                            return Some(Ok(Some((old_segment_id, old_offset))));
                        }
                        Err(_) => {
                            // CAS failed - let outer function retry
                            return Some(self.link_item(
                                key, pool_id, segment_id, offset, segments, metrics,
                            ));
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to replace a matching ghost entry in a bucket.
    fn try_replace_ghost_in_bucket(
        &self,
        bucket_index: usize,
        tag: u16,
        pool_id: u8,
        segment_id: u32,
        offset: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<Option<(u32, u32)>, ()>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if Hashbucket::is_ghost(current) && Hashbucket::item_tag(current) == tag {
                let ghost_freq = Hashbucket::item_freq(current);
                // Inherit the ghost's frequency (frequency only increases on reads)
                let inherited_freq = ghost_freq.max(1);
                let new_packed =
                    Hashbucket::pack_item(tag, inherited_freq, pool_id, segment_id, offset);

                match slot.compare_exchange(
                    current,
                    new_packed,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        metrics.hashtable_link.increment();
                        metrics.ghost_hit.increment();
                        return Some(Ok(None));
                    }
                    Err(_) => {
                        // Ghost changed - continue searching
                        continue;
                    }
                }
            }
        }

        None
    }

    /// Try to insert into an empty slot in a bucket.
    fn try_insert_empty_slot<P: Pool>(
        &self,
        bucket_index: usize,
        tag: u16,
        new_packed: u64,
        key: &[u8],
        segments: &P,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<Option<(u32, u32)>, ()>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if current == 0 {
                match slot.compare_exchange(0, new_packed, Ordering::Release, Ordering::Acquire) {
                    Ok(_) => {
                        metrics.hashtable_link.increment();

                        // Scan for duplicates in this bucket
                        if let Some(dup) =
                            self.scan_and_remove_duplicates(bucket_index, slot, tag, key, segments)
                        {
                            return Some(Ok(Some(dup)));
                        }

                        return Some(Ok(None));
                    }
                    Err(new_current) => {
                        // Check if another thread inserted our key
                        if new_current != 0
                            && !Hashbucket::is_ghost(new_current)
                            && Hashbucket::item_tag(new_current) == tag
                        {
                            let new_seg_id = Hashbucket::item_segment_id(new_current);
                            let new_offset = Hashbucket::item_offset(new_current);
                            if self.verify_key(key, new_seg_id, new_offset, segments, false) {
                                // Another thread inserted our key - need to retry to replace
                                let pool_id = Hashbucket::item_pool_id(new_packed);
                                let segment_id = Hashbucket::item_segment_id(new_packed);
                                let offset = Hashbucket::item_offset(new_packed);
                                return Some(self.link_item(
                                    key, pool_id, segment_id, offset, segments, metrics,
                                ));
                            }
                        }
                        // Slot taken by different item, continue searching
                        continue;
                    }
                }
            }
        }

        None
    }

    /// Scan a bucket for duplicate entries and remove them.
    fn scan_and_remove_duplicates<P: Pool>(
        &self,
        bucket_index: usize,
        exclude_slot: &AtomicU64,
        tag: u16,
        key: &[u8],
        segments: &P,
    ) -> Option<(u32, u32)> {
        let bucket = self.bucket(bucket_index);

        for other_slot in &bucket.items {
            if std::ptr::eq(exclude_slot, other_slot) {
                continue;
            }

            let other_packed = other_slot.load(Ordering::Acquire);
            if other_packed == 0 || Hashbucket::is_ghost(other_packed) {
                continue;
            }

            if Hashbucket::item_tag(other_packed) == tag {
                let other_seg_id = Hashbucket::item_segment_id(other_packed);
                let other_offset = Hashbucket::item_offset(other_packed);

                if self.verify_key(key, other_seg_id, other_offset, segments, false) {
                    // Found duplicate - try to unlink it
                    loop {
                        let current = other_slot.load(Ordering::Acquire);
                        if current == 0
                            || Hashbucket::item_tag(current) != tag
                            || Hashbucket::item_segment_id(current) != other_seg_id
                            || Hashbucket::item_offset(current) != other_offset
                        {
                            break;
                        }

                        match other_slot.compare_exchange(
                            current,
                            0,
                            Ordering::Release,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => return Some((other_seg_id, other_offset)),
                            Err(_) => continue,
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to evict any ghost entry in a bucket.
    fn try_evict_ghost_in_bucket(
        &self,
        bucket_index: usize,
        new_packed: u64,
        metrics: &crate::metrics::CacheMetrics,
    ) -> Option<Result<Option<(u32, u32)>, ()>> {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let current = slot.load(Ordering::Acquire);

            if Hashbucket::is_ghost(current) {
                match slot.compare_exchange(
                    current,
                    new_packed,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        metrics.hashtable_link.increment();
                        metrics.ghost_evict.increment();
                        return Some(Ok(None));
                    }
                    Err(_) => continue,
                }
            }
        }

        None
    }

    /// Unlink an item from the hash table.
    ///
    /// Checks both primary and secondary buckets (two-choice hashing).
    ///
    /// # Async Compatibility
    ///
    /// This function is designed to be async-friendly by using minimal backoff.
    /// However, for high-contention scenarios in async contexts, consider:
    /// 1. Using tokio::task::yield_now() between retries
    /// 2. Implementing an async variant that yields to the executor
    /// 3. Using a different strategy like message passing for deletions
    ///
    /// The current implementation uses bounded retries with minimal spin hints
    /// to avoid blocking the async executor thread.
    ///
    /// # Loom Test Coverage
    /// - `hashtable_unlink_concurrent` - Two threads unlinking (empty table)
    /// - Called by `evict_and_clear` which is tested indirectly via TTL bucket eviction
    pub fn unlink_item(
        &self,
        key: &[u8],
        segment_id: u32,
        offset: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if self.try_unlink_in_bucket(primary, tag, segment_id, offset, metrics) {
            return true;
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && self.try_unlink_in_bucket(secondary, tag, segment_id, offset, metrics)
        {
            return true;
        }

        metrics.item_unlink_not_found.increment();
        false
    }

    /// Try to unlink an item in a specific bucket.
    fn try_unlink_in_bucket(
        &self,
        bucket_index: usize,
        tag: u16,
        segment_id: u32,
        offset: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            if Hashbucket::item_tag(packed) == tag
                && Hashbucket::item_segment_id(packed) == segment_id
                && Hashbucket::item_offset(packed) == offset
            {
                let result = retry_cas_u64(
                    slot,
                    |current_packed| {
                        if current_packed == 0
                            || Hashbucket::is_ghost(current_packed)
                            || Hashbucket::item_tag(current_packed) != tag
                            || Hashbucket::item_segment_id(current_packed) != segment_id
                            || Hashbucket::item_offset(current_packed) != offset
                        {
                            return None;
                        }
                        Some((0, true))
                    },
                    CasRetryConfig::default(),
                    metrics,
                );

                if let CasResult::Success(true) = result {
                    metrics.item_unlink.increment();
                    return true;
                }
            }
        }

        false
    }

    /// Unlink an item from the hashtable, converting it to a ghost entry.
    ///
    /// Ghost entries preserve the tag and frequency but have no valid location.
    /// This is used during eviction to track recently-evicted items and their
    /// frequency history, which helps detect thrashing patterns.
    ///
    /// Checks both primary and secondary buckets (two-choice hashing).
    ///
    /// # Parameters
    /// - `key`: The key to unlink (used for hashing to find bucket)
    /// - `segment_id`: The segment ID (must match for CAS to succeed)
    /// - `offset`: The offset within the segment (must match for CAS to succeed)
    /// - `metrics`: Cache metrics for tracking operations
    ///
    /// # Returns
    /// `true` if the item was found and converted to ghost, `false` otherwise.
    pub fn unlink_item_to_ghost(
        &self,
        key: &[u8],
        segment_id: u32,
        offset: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool {
        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if self.try_unlink_to_ghost_in_bucket(primary, tag, segment_id, offset, metrics) {
            return true;
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && self.try_unlink_to_ghost_in_bucket(secondary, tag, segment_id, offset, metrics)
        {
            return true;
        }

        metrics.item_unlink_not_found.increment();
        false
    }

    /// Try to convert an item to ghost in a specific bucket.
    fn try_unlink_to_ghost_in_bucket(
        &self,
        bucket_index: usize,
        tag: u16,
        segment_id: u32,
        offset: u32,
        metrics: &crate::metrics::CacheMetrics,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            if Hashbucket::item_tag(packed) == tag
                && Hashbucket::item_segment_id(packed) == segment_id
                && Hashbucket::item_offset(packed) == offset
            {
                let result = retry_cas_u64(
                    slot,
                    |current_packed| {
                        if current_packed == 0
                            || Hashbucket::is_ghost(current_packed)
                            || Hashbucket::item_tag(current_packed) != tag
                            || Hashbucket::item_segment_id(current_packed) != segment_id
                            || Hashbucket::item_offset(current_packed) != offset
                        {
                            return None;
                        }
                        Some((Hashbucket::to_ghost(current_packed), true))
                    },
                    CasRetryConfig::default(),
                    metrics,
                );

                if let CasResult::Success(true) = result {
                    metrics.item_unlink.increment();
                    metrics.ghost_create.increment();
                    return true;
                }
            }
        }

        false
    }

    /// Update the location of an item in the hashtable without changing its frequency.
    ///
    /// Used during segment compaction where an item is moved within the same segment.
    /// The tag and frequency are preserved, only the offset is updated.
    ///
    /// # Parameters
    /// - `key`: The key to update
    /// - `segment_id`: The segment ID (must match existing entry)
    /// - `old_offset`: The current offset (must match for CAS to succeed)
    /// - `new_offset`: The new offset within the segment (must fit in 20 bits)
    ///
    /// # Returns
    /// `true` if the update succeeded, `false` if item not found or changed
    pub fn update_item_location(&self, key: &[u8], pool_id: u8, segment_id: u32, old_offset: u32, new_offset: u32) -> bool {
        self.relink_item(key, pool_id, segment_id, old_offset, pool_id, segment_id, new_offset, true)
    }

    /// Relink an item to a new location, optionally preserving frequency.
    ///
    /// This is more efficient than `link_item` when the old location is known,
    /// as it matches on (tag, segment_id, offset) without reading the segment.
    ///
    /// Checks both primary and secondary buckets (two-choice hashing).
    ///
    /// # Parameters
    /// - `key`: The key to relink (used for hashing to find bucket)
    /// - `old_segment_id`: The current segment ID
    /// - `old_offset`: The current offset
    /// - `new_pool_id`: The new pool ID
    /// - `new_segment_id`: The new segment ID
    /// - `new_offset`: The new offset
    /// - `preserve_freq`: If true, keeps existing frequency; if false, resets to 1
    ///
    /// # Returns
    /// `true` if the relink succeeded, `false` if item not found or changed
    #[allow(clippy::too_many_arguments)]
    pub fn relink_item(
        &self,
        key: &[u8],
        old_pool_id: u8,
        old_segment_id: u32,
        old_offset: u32,
        new_pool_id: u8,
        new_segment_id: u32,
        new_offset: u32,
        preserve_freq: bool,
    ) -> bool {
        // Validate new pool_id fits in 2 bits
        if new_pool_id > 3 {
            return false;
        }
        // Validate new segment_id fits in 22 bits
        if new_segment_id > 0x3FFFFF {
            return false;
        }
        // Validate new offset fits (stored as offset/8 in 20 bits, so max is 0xFFFFF * 8)
        if new_offset > 0xFFFFF << 3 {
            return false;
        }
        // Offset must be 8-byte aligned
        if new_offset & 0x7 != 0 {
            return false;
        }

        let hash = self.hash_builder.hash_one(key);
        let tag = ((hash >> 32) & 0xFFF) as u16;
        let (primary, secondary) = self.bucket_indices(hash);

        // Check primary bucket
        if self.try_relink_in_bucket(
            primary,
            tag,
            old_pool_id,
            old_segment_id,
            old_offset,
            new_pool_id,
            new_segment_id,
            new_offset,
            preserve_freq,
        ) {
            return true;
        }

        // Check secondary bucket only if two-choice hashing is enabled
        if self.two_choice
            && secondary != primary
            && self.try_relink_in_bucket(
                secondary,
                tag,
                old_pool_id,
                old_segment_id,
                old_offset,
                new_pool_id,
                new_segment_id,
                new_offset,
                preserve_freq,
            )
        {
            return true;
        }

        false
    }

    /// Try to relink an item in a specific bucket.
    #[allow(clippy::too_many_arguments)]
    fn try_relink_in_bucket(
        &self,
        bucket_index: usize,
        tag: u16,
        old_pool_id: u8,
        old_segment_id: u32,
        old_offset: u32,
        new_pool_id: u8,
        new_segment_id: u32,
        new_offset: u32,
        preserve_freq: bool,
    ) -> bool {
        let bucket = self.bucket(bucket_index);

        for slot in &bucket.items {
            let packed = slot.load(Ordering::Acquire);

            if packed == 0 || Hashbucket::is_ghost(packed) {
                continue;
            }

            if Hashbucket::item_tag(packed) == tag
                && Hashbucket::item_pool_id(packed) == old_pool_id
                && Hashbucket::item_segment_id(packed) == old_segment_id
                && Hashbucket::item_offset(packed) == old_offset
            {
                let freq = if preserve_freq {
                    Hashbucket::item_freq(packed)
                } else {
                    1
                };
                let new_packed =
                    Hashbucket::pack_item(tag, freq, new_pool_id, new_segment_id, new_offset);

                if slot
                    .compare_exchange(packed, new_packed, Ordering::Release, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
            }
        }

        false
    }
}

pub struct Hashbucket {
    pub(crate) info: AtomicU64,
    pub(crate) items: [AtomicU64; 7],
}

impl Hashbucket {
    #[allow(dead_code)]
    pub fn cas(&self) -> u32 {
        (self.info.load(Ordering::Acquire) >> 32) as u32
    }

    #[allow(dead_code)]
    pub fn timestamp(&self) -> u16 {
        self.info.load(Ordering::Acquire) as u16
    }

    /// Get a reference to the item slots
    #[allow(dead_code)]
    pub fn items(&self) -> &[AtomicU64; 7] {
        &self.items
    }

    /// Update the CAS value for this bucket
    #[allow(dead_code)]
    pub fn update_cas(&self, cas: u32, metrics: &crate::metrics::CacheMetrics) {
        let result = retry_cas_u64(
            &self.info,
            |current| Some((((cas as u64) << 32) | (current & 0xFFFFFFFF), ())),
            CasRetryConfig::default(),
            metrics,
        );

        assert!(
            matches!(result, CasResult::Success(())),
            "Failed to update CAS"
        );
    }

    /// Pack item information into a u64
    /// Layout: [12 bits tag][8 bits freq][2 bits pool_id][22 bits segment_id][20 bits offset/8]
    ///
    /// Pool ID (2 bits): Supports up to 4 segment pools (e.g., hot, warm, cold, frozen)
    /// Segment ID (22 bits): 4M segments per pool × 1MB = 4TB per pool
    /// Offset (20 bits): Stored as offset/8, supports up to ~8MB per segment
    ///
    /// Since items are 8-byte aligned, we store offset/8 to get 8x more addressable space.
    pub fn pack_item(tag: u16, freq: u8, pool_id: u8, segment_id: u32, offset: u32) -> u64 {
        // Offset must be 8-byte aligned
        debug_assert!(offset & 0x7 == 0, "offset {} is not 8-byte aligned", offset);
        debug_assert!(pool_id <= 3, "pool_id {} exceeds 2-bit limit", pool_id);
        debug_assert!(segment_id <= 0x3FFFFF, "segment_id {} exceeds 22-bit limit", segment_id);
        let tag_64 = (tag as u64 & 0xFFF) << 52;
        let freq_64 = (freq as u64 & 0xFF) << 44;
        let pool_64 = (pool_id as u64 & 0x3) << 42;
        let seg_64 = (segment_id as u64 & 0x3FFFFF) << 20;
        let off_64 = (offset >> 3) as u64 & 0xFFFFF; // Store offset/8
        tag_64 | freq_64 | pool_64 | seg_64 | off_64
    }

    /// Extract tag from packed item (12 bits)
    pub fn item_tag(packed: u64) -> u16 {
        (packed >> 52) as u16
    }

    /// Extract pool ID from packed item (2 bits)
    pub fn item_pool_id(packed: u64) -> u8 {
        ((packed >> 42) & 0x3) as u8
    }

    /// Extract segment ID from packed item (22 bits)
    pub fn item_segment_id(packed: u64) -> u32 {
        ((packed >> 20) & 0x3FFFFF) as u32
    }

    /// Extract offset from packed item (20 bits stored, multiplied by 8)
    ///
    /// Since we store offset/8, we multiply by 8 to get the actual offset.
    pub fn item_offset(packed: u64) -> u32 {
        ((packed & 0xFFFFF) << 3) as u32 // Return offset * 8
    }

    /// Extract frequency from packed item (8 bits)
    pub(crate) fn item_freq(packed: u64) -> u8 {
        ((packed >> 44) & 0xFF) as u8
    }

    /// Try to update frequency using ASFC algorithm.
    /// Returns Some(new_packed) if frequency should be incremented, None otherwise.
    ///
    /// This avoids RNG calls for hot items by using probabilistic increment:
    /// - Cold items (freq <= 16): Always increment
    /// - Hot items (freq > 16): Increment with probability 1/freq
    ///
    /// The caller should check freq < 127 before calling to avoid unnecessary work.
    pub(crate) fn try_update_freq(packed: u64, freq: u8) -> Option<u64> {
        // ASFC: Adaptive Software Frequency Counter
        // Caller should ensure freq < 127, but double-check
        if freq >= 127 {
            return None;
        }

        // Probabilistic increment:
        // - Always increment if freq <= 16 (cold items)
        // - For freq > 16, increment with probability 1/freq (hot items increment slower)
        let should_increment = if freq <= 16 {
            true
        } else {
            // Use thread-local RNG for better performance
            #[cfg(not(feature = "loom"))]
            let rand = {
                use rand::Rng;
                rand::rng().random::<u64>()
            };

            #[cfg(feature = "loom")]
            let rand = 0u64; // Deterministic for loom tests

            rand % (freq as u64) == 0
        };

        if should_increment {
            let new_freq = freq + 1;
            // Clear old freq bits and set new freq
            let freq_mask = 0xFF_u64 << 44;
            Some((packed & !freq_mask) | ((new_freq as u64) << 44))
        } else {
            None
        }
    }

    /// Sentinel segment ID used to mark ghost entries.
    /// Ghost entries preserve tag and frequency but have no valid location.
    /// This is the maximum 22-bit value, which cannot be a valid segment ID.
    pub const GHOST_SEGMENT_ID: u32 = 0x3FFFFF;

    /// Check if a packed item is a ghost entry.
    /// Ghost entries have segment_id == GHOST_SEGMENT_ID.
    pub fn is_ghost(packed: u64) -> bool {
        packed != 0 && Self::item_segment_id(packed) == Self::GHOST_SEGMENT_ID
    }

    /// Pack a ghost entry preserving tag and frequency.
    /// Ghost entries use GHOST_SEGMENT_ID as sentinel, with pool_id=0 and offset=0.
    pub fn pack_ghost(tag: u16, freq: u8) -> u64 {
        let tag_64 = (tag as u64 & 0xFFF) << 52;
        let freq_64 = (freq as u64 & 0xFF) << 44;
        // pool_id = 0, segment_id = GHOST_SEGMENT_ID, offset = 0
        let seg_64 = (Self::GHOST_SEGMENT_ID as u64 & 0x3FFFFF) << 20;
        tag_64 | freq_64 | seg_64
    }

    /// Convert a live item to a ghost entry, preserving tag and frequency.
    pub fn to_ghost(packed: u64) -> u64 {
        let tag = Self::item_tag(packed);
        let freq = Self::item_freq(packed);
        Self::pack_ghost(tag, freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_item_basic() {
        let tag = 0xABC;
        let freq = 42;
        let pool_id = 2;
        let segment_id = 0x123456;
        let offset = 0x100; // Must be 8-byte aligned

        let packed = Hashbucket::pack_item(tag, freq, pool_id, segment_id, offset);

        assert_eq!(Hashbucket::item_tag(packed), tag);
        assert_eq!(Hashbucket::item_freq(packed), freq);
        assert_eq!(Hashbucket::item_pool_id(packed), pool_id);
        assert_eq!(Hashbucket::item_segment_id(packed), segment_id);
        assert_eq!(Hashbucket::item_offset(packed), offset);
    }

    #[test]
    fn test_pack_item_max_values() {
        let tag = 0xFFF; // 12 bits max
        let freq = 0xFF; // 8 bits max
        let pool_id = 3; // 2 bits max
        let segment_id = 0x3FFFFF; // 22 bits max (but this is GHOST_SEGMENT_ID)
        let offset = 0xFFFFF << 3; // 20 bits * 8 max

        // Use segment_id - 1 to avoid GHOST_SEGMENT_ID
        let packed = Hashbucket::pack_item(tag, freq, pool_id, segment_id - 1, offset);

        assert_eq!(Hashbucket::item_tag(packed), tag);
        assert_eq!(Hashbucket::item_freq(packed), freq);
        assert_eq!(Hashbucket::item_pool_id(packed), pool_id);
        assert_eq!(Hashbucket::item_segment_id(packed), segment_id - 1);
        assert_eq!(Hashbucket::item_offset(packed), offset);
    }

    #[test]
    fn test_ghost_entry_pack_unpack() {
        let tag = 0x123;
        let freq = 50;

        let ghost = Hashbucket::pack_ghost(tag, freq);

        assert!(Hashbucket::is_ghost(ghost));
        assert_eq!(Hashbucket::item_tag(ghost), tag);
        assert_eq!(Hashbucket::item_freq(ghost), freq);
        assert_eq!(Hashbucket::item_segment_id(ghost), Hashbucket::GHOST_SEGMENT_ID);
    }

    #[test]
    fn test_to_ghost_preserves_tag_and_freq() {
        let tag = 0xABC;
        let freq = 75;
        let pool_id = 1;
        let segment_id = 0x5000;
        let offset = 0x200;

        let packed = Hashbucket::pack_item(tag, freq, pool_id, segment_id, offset);
        assert!(!Hashbucket::is_ghost(packed));

        let ghost = Hashbucket::to_ghost(packed);
        assert!(Hashbucket::is_ghost(ghost));
        assert_eq!(Hashbucket::item_tag(ghost), tag);
        assert_eq!(Hashbucket::item_freq(ghost), freq);
    }

    #[test]
    fn test_is_ghost_empty_slot() {
        // Empty slot (0) should not be considered a ghost
        assert!(!Hashbucket::is_ghost(0));
    }

    #[test]
    fn test_is_ghost_real_item() {
        let packed = Hashbucket::pack_item(0x123, 1, 0, 0x1000, 0x100);
        assert!(!Hashbucket::is_ghost(packed));
    }

    #[test]
    fn test_hashtable_new() {
        let ht = Hashtable::new(4).unwrap(); // 2^4 = 16 buckets
        assert_eq!(ht.mask, 15); // 16 - 1
        assert_eq!(ht.num_buckets, 16);
    }

    #[test]
    fn test_hashtable_new_power_range() {
        // Test minimum power (4 = 16 buckets)
        let ht_min = Hashtable::new(4).unwrap();
        assert_eq!(ht_min.num_buckets, 16);

        // Test larger power
        let ht_large = Hashtable::new(10).unwrap(); // 1024 buckets
        assert_eq!(ht_large.num_buckets, 1024);
    }

    #[test]
    fn test_try_update_freq_probabilistic() {
        // At low frequency, updates should happen more often
        let packed = Hashbucket::pack_item(0x123, 1, 0, 0x1000, 0x100);

        // With freq=1, probability is 1/1 = 100%, so should always update
        let result = Hashbucket::try_update_freq(packed, 1);
        assert!(result.is_some());

        if let Some(new_packed) = result {
            assert_eq!(Hashbucket::item_freq(new_packed), 2);
            // Tag, pool_id, segment_id, offset should be preserved
            assert_eq!(Hashbucket::item_tag(new_packed), 0x123);
            assert_eq!(Hashbucket::item_pool_id(new_packed), 0);
            assert_eq!(Hashbucket::item_segment_id(new_packed), 0x1000);
            assert_eq!(Hashbucket::item_offset(new_packed), 0x100);
        }
    }

    #[test]
    fn test_try_update_freq_max() {
        // At max frequency (127), updates should not happen
        let packed = Hashbucket::pack_item(0x123, 127, 0, 0x1000, 0x100);
        let result = Hashbucket::try_update_freq(packed, 127);
        assert!(result.is_none());
    }

    /// Test maximum fill rate with two-choice hashing
    ///
    /// This test measures how many items can be inserted before the hashtable
    /// becomes "full" (insertion failures start occurring).
    ///
    /// Two-choice hashing achieves ~95% fill rate vs ~85% for single-choice.
    #[test]
    fn test_fill_rate_two_choice() {
        use crate::pool::MemoryPoolBuilder;
        use crate::segment::Segment;
        use crate::pool::Pool;

        let power: u8 = 12; // 4K buckets = 28K slots
        let num_buckets = 1usize << power;
        let num_slots = num_buckets * 7;

        // Create pool for item storage
        let pool = MemoryPoolBuilder::new(0)
            .segment_size(1024 * 1024)
            .heap_size(64 * 1024 * 1024)
            .small_queue_percent(0)
            .build()
            .expect("Failed to create pool");

        // Use two-choice hashing for this test
        let hashtable = Hashtable::with_two_choice(power, true).unwrap();
        let metrics = crate::metrics::CacheMetrics::new();

        // Insert items until we hit many failures
        let mut success = 0;
        let mut fail = 0;
        let mut current_segment = pool.reserve_main_cache(&metrics).expect("Failed to reserve segment");

        // Try to insert 100% of slots
        for i in 0..num_slots {
            let key = format!("key_{:016x}", i).into_bytes();
            let value = format!("value_{:016x}", i).into_bytes();

            let segment = pool.get(current_segment).expect("Failed to get segment");
            let offset = match segment.append_item(&key, &value, &[], &metrics) {
                Some(offset) => offset,
                None => {
                    current_segment = pool.reserve_main_cache(&metrics)
                        .expect("Failed to reserve segment");
                    let segment = pool.get(current_segment).expect("Failed to get segment");
                    segment.append_item(&key, &value, &[], &metrics)
                        .expect("Failed to append to fresh segment")
                }
            };

            if hashtable.link_item(&key, 0, current_segment, offset, &pool, &metrics).is_ok() {
                success += 1;
            } else {
                fail += 1;
            }
        }

        let fill_percent = (success as f64 / num_slots as f64) * 100.0;
        println!("\n=== Two-Choice Hashing Fill Rate ===");
        println!("Buckets: {}", num_buckets);
        println!("Slots: {} (7 per bucket)", num_slots);
        println!("Successful inserts: {}", success);
        println!("Failed inserts: {}", fail);
        println!("Fill rate: {:.2}%", fill_percent);
        println!("=====================================\n");

        // Expect at least 90% fill rate with two-choice hashing
        assert!(fill_percent >= 90.0, "Expected at least 90% fill rate, got {:.2}%", fill_percent);
    }

    /// Test fill rate with single-choice hashing (default mode)
    #[test]
    fn test_fill_rate_single_choice() {
        use crate::pool::MemoryPoolBuilder;
        use crate::segment::Segment;
        use crate::pool::Pool;

        let power: u8 = 12;
        let num_buckets = 1usize << power;
        let num_slots = num_buckets * 7;

        let pool = MemoryPoolBuilder::new(0)
            .segment_size(1024 * 1024)
            .heap_size(64 * 1024 * 1024)
            .small_queue_percent(0)
            .build()
            .expect("Failed to create pool");

        // Default: single-choice hashing
        let hashtable = Hashtable::new(power).unwrap();
        let metrics = crate::metrics::CacheMetrics::new();

        let mut success = 0;
        let mut current_segment = pool.reserve_main_cache(&metrics).expect("Failed to reserve segment");

        for i in 0..num_slots {
            let key = format!("key_{:016x}", i).into_bytes();
            let value = format!("value_{:016x}", i).into_bytes();

            let segment = pool.get(current_segment).expect("Failed to get segment");
            let offset = match segment.append_item(&key, &value, &[], &metrics) {
                Some(offset) => offset,
                None => {
                    current_segment = pool.reserve_main_cache(&metrics)
                        .expect("Failed to reserve segment");
                    let segment = pool.get(current_segment).expect("Failed to get segment");
                    segment.append_item(&key, &value, &[], &metrics)
                        .expect("Failed to append to fresh segment")
                }
            };

            if hashtable.link_item(&key, 0, current_segment, offset, &pool, &metrics).is_ok() {
                success += 1;
            }
        }

        let fill_percent = (success as f64 / num_slots as f64) * 100.0;

        // Single-choice should achieve at least 80% fill rate
        assert!(fill_percent >= 80.0, "Expected at least 80% fill rate, got {:.2}%", fill_percent);
        // But less than two-choice (typically ~85% vs ~95%)
        assert!(fill_percent < 92.0, "Single-choice should be below 92%, got {:.2}%", fill_percent);
    }
}
