use crate::segment::Segment;

/// Item header for main cache segments (TTL inherited from segment)
///
/// Layout with validation feature:
/// ```text
/// [0..4] len: [key_len (8 bits)][value_len (24 bits)]
/// [4]    flags: [is_numeric (1)][is_deleted (1)][optional_len (6)]
/// [5]    MAGIC0 (0xCA)
/// [6]    MAGIC1 (0xCE)
/// [7]    checksum
/// ```
///
/// Layout without validation:
/// ```text
/// [0..4] len: [key_len (8 bits)][value_len (24 bits)]
/// [4]    flags: [is_numeric (1)][is_deleted (1)][optional_len (6)]
/// ```
#[derive(Debug)]
pub struct ItemHeader {
    key_len: u8,
    optional_len: u8,
    is_deleted: bool,
    is_numeric: bool,
    value_len: u32,
}

impl ItemHeader {
    /// Magic bytes to detect valid headers (0xCA 0xCE = "CACHE")
    pub const MAGIC0: u8 = 0xCA;
    pub const MAGIC1: u8 = 0xCE;

    /// Create a new ItemHeader
    pub fn new(
        key_len: u8,
        optional_len: u8,
        value_len: u32,
        is_deleted: bool,
        is_numeric: bool,
    ) -> Self {
        Self {
            key_len,
            optional_len,
            is_deleted,
            is_numeric,
            value_len,
        }
    }

    /// The packed length of the item header
    #[cfg(feature = "validation")]
    pub const SIZE: usize = 8; // With magic bytes and checksum

    #[cfg(not(feature = "validation"))]
    pub const SIZE: usize = 5; // Original compact format

    /// Maximum key length (8 bits)
    pub const MAX_KEY_LEN: usize = 0xFF;

    /// Maximum optional metadata length (6 bits in flags)
    pub const MAX_OPTIONAL_LEN: usize = 0x3F;

    /// Maximum value length (24 bits)
    pub const MAX_VALUE_LEN: usize = (1 << 24) - 1;

    /// Minimum valid item size for bounds checking
    pub const MIN_ITEM_SIZE: usize = Self::SIZE;

    /// Compute a simple checksum for validation (excluding is_deleted flag)
    /// Uses XOR of immutable fields only - is_deleted is mutable via atomic update
    fn compute_checksum(key_len: u8, optional_len: u8, value_len: u32, is_numeric: bool) -> u8 {
        let mut checksum = Self::MAGIC0;
        checksum ^= Self::MAGIC1;
        checksum ^= key_len;
        checksum ^= optional_len;
        checksum ^= (value_len & 0xFF) as u8;
        checksum ^= ((value_len >> 8) & 0xFF) as u8;
        checksum ^= ((value_len >> 16) & 0xFF) as u8;
        if is_numeric { checksum ^= 0x55; }
        checksum
    }

    /// Try to parse header from bytes, returning None if validation fails.
    /// This is used in paths where invalid headers are expected (e.g., TOCTOU races).
    pub fn try_from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < ItemHeader::SIZE {
            return None;
        }

        // Parse header fields first
        let len = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        let flags = data[4];

        let key_len = len as u8;
        let value_len = len >> 8;
        let optional_len = flags & 0x3F;
        let is_deleted = (flags & 0x40) != 0;
        let is_numeric = (flags & 0x80) != 0;

        #[cfg(feature = "validation")]
        {
            // Validate magic bytes
            if data[5] != Self::MAGIC0 || data[6] != Self::MAGIC1 {
                return None;
            }

            // Validate checksum
            let stored_checksum = data[7];
            let computed_checksum = Self::compute_checksum(key_len, optional_len, value_len, is_numeric);
            if stored_checksum != computed_checksum {
                return None;
            }
        }

        Some(Self {
            key_len,
            optional_len,
            is_deleted,
            is_numeric,
            value_len,
        })
    }

    /// Parse header from bytes, panicking with context if validation fails.
    /// Use this for internal operations (compact, prune, etc.) where corruption
    /// indicates a serious bug that should crash rather than leave partial state.
    pub fn from_bytes_with_context(data: &[u8], segment_id: u32, offset: u32, operation: &str) -> Self {
        match Self::try_from_bytes(data) {
            Some(header) => header,
            None => {
                #[cfg(feature = "validation")]
                {
                    let magic0 = data.get(5).copied().unwrap_or(0);
                    let magic1 = data.get(6).copied().unwrap_or(0);
                    panic!(
                        "CORRUPTION in {}: Invalid item header at segment {} offset {}. \
                         Expected magic [0xCA, 0xCE], got [0x{:02X}, 0x{:02X}]. \
                         Raw header bytes: {:02X?}",
                        operation, segment_id, offset, magic0, magic1,
                        &data[..ItemHeader::SIZE.min(data.len())]
                    );
                }
                #[cfg(not(feature = "validation"))]
                {
                    panic!(
                        "CORRUPTION in {}: Invalid item header at segment {} offset {}. \
                         Raw header bytes: {:02X?}",
                        operation, segment_id, offset,
                        &data[..ItemHeader::SIZE.min(data.len())]
                    );
                }
            }
        }
    }

    pub fn to_bytes(&self, data: &mut [u8]) {
        debug_assert!(data.len() >= ItemHeader::SIZE);

        // Write original 5-byte format in bytes 0-4 (compatible with atomic flag updates)
        let len = (self.value_len << 8) | (self.key_len as u32);
        let mut flags = self.optional_len;

        if self.is_deleted {
            flags |= 0x40;
        }

        if self.is_numeric {
            flags |= 0x80;
        }

        data[0..4].copy_from_slice(&len.to_ne_bytes());
        data[4] = flags;

        #[cfg(feature = "validation")]
        {
            // Write magic bytes and checksum at end (bytes 5-7)
            data[5] = Self::MAGIC0;
            data[6] = Self::MAGIC1;
            data[7] = Self::compute_checksum(self.key_len, self.optional_len, self.value_len, self.is_numeric);
        }
    }

    /// Calculate the padded size for this item, rounded to 8-byte boundary
    pub fn padded_size(&self) -> usize {
        let size = Self::SIZE
            .checked_add(self.optional_len as usize)
            .and_then(|s| s.checked_add(self.key_len as usize))
            .and_then(|s| s.checked_add(self.value_len as usize))
            .and_then(|s| s.checked_add(7))  // Add 7 for 8-byte alignment
            .expect("Item size overflow");

        size & !7  // Round up to 8-byte boundary
    }

    // Getter methods for accessing private fields
    pub fn is_deleted(&self) -> bool {
        self.is_deleted
    }

    pub fn key_len(&self) -> u8 {
        self.key_len
    }

    pub fn optional_len(&self) -> u8 {
        self.optional_len
    }

    pub fn value_len(&self) -> u32 {
        self.value_len
    }
}

/// Item header for small queue segments (per-item TTL)
///
/// Small queue items need their own TTL since they're in a FIFO queue
/// without TTL bucket organization. The expire_at field stores a coarse
/// timestamp (seconds since cache start) for efficient expiration checking.
///
/// Layout with validation feature:
/// ```text
/// [0..4]  len: [key_len (8 bits)][value_len (24 bits)]
/// [4]     flags: [is_numeric (1)][is_deleted (1)][optional_len (6)]
/// [5..9]  expire_at: u32 (coarse seconds, ~136 years range)
/// [9]     MAGIC0 (0xCA)
/// [10]    MAGIC1 (0xCE)
/// [11]    checksum
/// ```
///
/// Layout without validation:
/// ```text
/// [0..4]  len: [key_len (8 bits)][value_len (24 bits)]
/// [4]     flags: [is_numeric (1)][is_deleted (1)][optional_len (6)]
/// [5..9]  expire_at: u32 (coarse seconds)
/// ```
#[derive(Debug)]
pub struct SmallQueueItemHeader {
    key_len: u8,
    optional_len: u8,
    is_deleted: bool,
    is_numeric: bool,
    value_len: u32,
    /// Expiration time as coarse seconds (relative to cache epoch)
    expire_at: u32,
}

impl SmallQueueItemHeader {
    /// Magic bytes to detect valid headers (0xCA 0xCE = "CACHE")
    pub const MAGIC0: u8 = 0xCA;
    pub const MAGIC1: u8 = 0xCE;

    /// Create a new SmallQueueItemHeader
    pub fn new(
        key_len: u8,
        optional_len: u8,
        value_len: u32,
        is_deleted: bool,
        is_numeric: bool,
        expire_at: u32,
    ) -> Self {
        Self {
            key_len,
            optional_len,
            is_deleted,
            is_numeric,
            value_len,
            expire_at,
        }
    }

    /// The packed length of the item header
    #[cfg(feature = "validation")]
    pub const SIZE: usize = 12; // 5 base + 4 expire_at + 3 validation

    #[cfg(not(feature = "validation"))]
    pub const SIZE: usize = 9; // 5 base + 4 expire_at

    /// Maximum key length (8 bits)
    pub const MAX_KEY_LEN: usize = 0xFF;

    /// Maximum optional metadata length (6 bits in flags)
    pub const MAX_OPTIONAL_LEN: usize = 0x3F;

    /// Maximum value length (24 bits)
    pub const MAX_VALUE_LEN: usize = (1 << 24) - 1;

    /// Compute a simple checksum for validation (excluding is_deleted flag)
    fn compute_checksum(key_len: u8, optional_len: u8, value_len: u32, is_numeric: bool, expire_at: u32) -> u8 {
        let mut checksum = Self::MAGIC0;
        checksum ^= Self::MAGIC1;
        checksum ^= key_len;
        checksum ^= optional_len;
        checksum ^= (value_len & 0xFF) as u8;
        checksum ^= ((value_len >> 8) & 0xFF) as u8;
        checksum ^= ((value_len >> 16) & 0xFF) as u8;
        checksum ^= (expire_at & 0xFF) as u8;
        checksum ^= ((expire_at >> 8) & 0xFF) as u8;
        checksum ^= ((expire_at >> 16) & 0xFF) as u8;
        checksum ^= ((expire_at >> 24) & 0xFF) as u8;
        if is_numeric { checksum ^= 0x55; }
        checksum
    }

    /// Try to parse header from bytes, returning None if validation fails
    pub fn try_from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SIZE {
            return None;
        }

        let len = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        let flags = data[4];
        let expire_at = u32::from_ne_bytes([data[5], data[6], data[7], data[8]]);

        let key_len = len as u8;
        let value_len = len >> 8;
        let optional_len = flags & 0x3F;
        let is_deleted = (flags & 0x40) != 0;
        let is_numeric = (flags & 0x80) != 0;

        #[cfg(feature = "validation")]
        {
            if data[9] != Self::MAGIC0 || data[10] != Self::MAGIC1 {
                return None;
            }

            let stored_checksum = data[11];
            let computed_checksum = Self::compute_checksum(key_len, optional_len, value_len, is_numeric, expire_at);
            if stored_checksum != computed_checksum {
                return None;
            }
        }

        Some(Self {
            key_len,
            optional_len,
            is_deleted,
            is_numeric,
            value_len,
            expire_at,
        })
    }

    pub fn to_bytes(&self, data: &mut [u8]) {
        debug_assert!(data.len() >= Self::SIZE);

        let len = (self.value_len << 8) | (self.key_len as u32);
        let mut flags = self.optional_len;

        if self.is_deleted {
            flags |= 0x40;
        }

        if self.is_numeric {
            flags |= 0x80;
        }

        data[0..4].copy_from_slice(&len.to_ne_bytes());
        data[4] = flags;
        data[5..9].copy_from_slice(&self.expire_at.to_ne_bytes());

        #[cfg(feature = "validation")]
        {
            data[9] = Self::MAGIC0;
            data[10] = Self::MAGIC1;
            data[11] = Self::compute_checksum(self.key_len, self.optional_len, self.value_len, self.is_numeric, self.expire_at);
        }
    }

    /// Calculate the padded size for this item, rounded to 8-byte boundary
    pub fn padded_size(&self) -> usize {
        let size = Self::SIZE
            .checked_add(self.optional_len as usize)
            .and_then(|s| s.checked_add(self.key_len as usize))
            .and_then(|s| s.checked_add(self.value_len as usize))
            .and_then(|s| s.checked_add(7))
            .expect("Item size overflow");

        size & !7
    }

    pub fn is_deleted(&self) -> bool {
        self.is_deleted
    }

    pub fn key_len(&self) -> u8 {
        self.key_len
    }

    pub fn optional_len(&self) -> u8 {
        self.optional_len
    }

    pub fn value_len(&self) -> u32 {
        self.value_len
    }

    pub fn expire_at(&self) -> u32 {
        self.expire_at
    }

    /// Check if this item has expired given the current coarse time
    pub fn is_expired(&self, now: u32) -> bool {
        now >= self.expire_at
    }
}

/// A guard that holds a reference to an item's data in a segment.
///
/// This is a zero-copy, zero-allocation view into the segment's data.
/// The segment's reference count is held while this guard exists,
/// preventing eviction or clearing of the segment.
///
/// The guard is dropped automatically when it goes out of scope, at which
/// point the segment's reference count is decremented.
///
/// # Examples
///
/// ```ignore
/// let guard = cache.get(b"key")?;
/// // Zero-copy access to item data
/// let value = guard.value();
/// // Can serialize directly to socket, etc.
/// socket.write_all(value)?;
/// // Guard dropped here, ref_count decremented
/// ```
pub struct ItemGuard<'a, S> where S: Segment + ?Sized {
    segment: &'a S,
    key: &'a [u8],
    value: &'a [u8],
    optional: &'a [u8],
}

impl<'a, S> ItemGuard<'a, S> where S: Segment {
    /// Create a new ItemGuard
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The segment's ref_count has been incremented
    /// - The slice references are valid and point into the segment's data
    pub(crate) fn new(
        segment: &'a S,
        key: &'a [u8],
        value: &'a [u8],
        optional: &'a [u8],
    ) -> Self {
        Self {
            segment,
            key,
            value,
            optional,
        }
    }

    /// Get the item's key
    pub fn key(&self) -> &[u8] {
        self.key
    }

    /// Get the item's value
    pub fn value(&self) -> &[u8] {
        self.value
    }

    /// Get the item's optional metadata
    pub fn optional(&self) -> &[u8] {
        self.optional
    }
}

impl<S: ?Sized + Segment> Drop for ItemGuard<'_, S> {
    fn drop(&mut self) {
        self.segment.decr_ref_count();
    }
}
