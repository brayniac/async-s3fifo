// Atomic wrappers that switch between std and loom based on feature flag

#[cfg(not(feature = "loom"))]
pub(crate) use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering, fence};

#[cfg(feature = "loom")]
pub(crate) use loom::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering, fence};

// Helper for spin loop hints
#[cfg(not(feature = "loom"))]
#[inline]
pub(crate) fn spin_loop() {
    std::hint::spin_loop();
}

#[cfg(feature = "loom")]
#[inline]
pub(crate) fn spin_loop() {
    loom::thread::yield_now();
}
