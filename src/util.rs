use crate::segment::Metadata;
use crate::segment::State;
use crate::sync::{AtomicU32, AtomicU64, Ordering, spin_loop};
// use crate::segments::{SegmentState};

/// Result of a CAS retry operation
#[derive(Debug, PartialEq)]
pub enum CasResult<T> {
    /// Operation succeeded with the given result
    Success(T),
    /// Operation failed after all retries, with the final observed value
    Failed(u64),
    /// Operation was aborted due to a condition change
    Aborted,
}

/// Configuration for CAS retry operations
#[derive(Clone, Copy)]
pub struct CasRetryConfig {
    pub max_attempts: u32,
    pub early_spin_threshold: u32,
}

impl Default for CasRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 16,            // Reasonable attempts before giving up
            early_spin_threshold: 16,    // Never yield - pure spinning with exponential backoff
        }
    }
}

/// Retry a CAS operation with configurable backoff strategy.
///
/// This encapsulates the common pattern of:
/// 1. Load current value
/// 2. Check if operation should proceed
/// 3. Attempt CAS
/// 4. Handle failure with appropriate backoff
///
/// The closure should return:
/// - `Some(new_value)` to attempt CAS with that value
/// - `None` to abort (e.g., if conditions changed)
#[inline]
pub(crate) fn retry_cas_u64<T, F>(
    atomic: &AtomicU64,
    mut operation: F,
    config: CasRetryConfig,
    metrics: &crate::metrics::CacheMetrics,
) -> CasResult<T>
where
    F: FnMut(u64) -> Option<(u64, T)>, // (new_value, result_on_success)
{
    let mut current = atomic.load(Ordering::Acquire);

    for attempt in 0..config.max_attempts {
        // Track retry attempts (skip first attempt)
        if attempt > 0 {
            metrics.cas_retry.increment();
        }

        // Let the operation decide if it should proceed and what value to use
        let (new_value, success_result) = match operation(current) {
            Some(result) => result,
            None => {
                metrics.cas_abort.increment();
                return CasResult::Aborted;
            }
        };

        // Attempt the CAS
        match atomic.compare_exchange_weak(current, new_value, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => return CasResult::Success(success_result),
            Err(actual) => {
                current = actual;

                // Progressive backoff strategy
                if attempt < config.early_spin_threshold {
                    // Exponential backoff with reasonable cap
                    // Attempt 0: 1 spin, 1: 2 spins, 2: 4 spins, 3: 8 spins, 4+: 16 spins
                    let spin_count = 1u32 << attempt.min(6); // Up to 64 spins
                    for _ in 0..spin_count {
                        spin_loop();
                    }
                } else {
                    // Very late attempts: aggressive spinning instead of yield (async compatible)
                    for _ in 0..128 {
                        spin_loop();
                    }
                }
            }
        }
    }

    CasResult::Failed(current)
}

/// Retry a CAS operation on AtomicU32 with configurable backoff strategy.
#[inline]
pub(crate) fn retry_cas_u32<T, F>(
    atomic: &AtomicU32,
    mut operation: F,
    config: CasRetryConfig,
    metrics: &crate::metrics::CacheMetrics,
) -> CasResult<T>
where
    F: FnMut(u32) -> Option<(u32, T)>, // (new_value, result_on_success)
{
    let mut current = atomic.load(Ordering::Acquire);

    for attempt in 0..config.max_attempts {
        // Track retry attempts (skip first attempt)
        if attempt > 0 {
            metrics.cas_retry.increment();
        }

        // Let the operation decide if it should proceed and what value to use
        let (new_value, success_result) = match operation(current) {
            Some(result) => result,
            None => {
                metrics.cas_abort.increment();
                return CasResult::Aborted;
            }
        };

        // Attempt the CAS
        match atomic.compare_exchange_weak(current, new_value, Ordering::Release, Ordering::Acquire)
        {
            Ok(_) => return CasResult::Success(success_result),
            Err(actual) => {
                current = actual;

                // Progressive backoff strategy
                if attempt < config.early_spin_threshold {
                    // Exponential backoff with reasonable cap
                    // Attempt 0: 1 spin, 1: 2 spins, 2: 4 spins, 3: 8 spins, 4+: 16 spins
                    let spin_count = 1u32 << attempt.min(6); // Up to 64 spins
                    for _ in 0..spin_count {
                        spin_loop();
                    }
                } else {
                    // Very late attempts: aggressive spinning instead of yield (async compatible)
                    for _ in 0..128 {
                        spin_loop();
                    }
                }
            }
        }
    }

    CasResult::Failed(current as u64)
}

/// Specialized retry for packed metadata operations
#[inline]
pub(crate) fn retry_cas_metadata<F>(
    atomic: &AtomicU64,
    expected_state: State,
    operation: F,
    config: CasRetryConfig,
    metrics: &crate::metrics::CacheMetrics,
) -> CasResult<()>
where
    F: Fn(Metadata) -> Option<Metadata>,
{
    retry_cas_u64(
        atomic,
        |current_packed| {
            let current_meta = Metadata::unpack(current_packed);

            // Only proceed if we're in the expected state
            if current_meta.state != expected_state {
                return None;
            }

            // Let the operation transform the metadata
            let new_meta = operation(current_meta)?;
            Some((new_meta.pack(), ()))
        },
        config,
        metrics,
    )
}
