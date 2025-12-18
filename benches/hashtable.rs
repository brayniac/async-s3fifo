//! Hashtable benchmarks for comparing single-choice vs two-choice hashing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use s3::hashtable::Hashtable;
use s3::pool::Pool;
use s3::segment::Segment;
use s3::{CacheMetrics, MemoryPoolBuilder};

/// Generate a key for the given index
fn generate_key(index: usize) -> Vec<u8> {
    format!("key_{:016x}", index).into_bytes()
}

/// Generate a value for the given index
fn generate_value(index: usize) -> Vec<u8> {
    format!("value_{:016x}", index).into_bytes()
}

/// Benchmark hashtable insertion at various occupancy levels
fn bench_hashtable_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtable_insert");

    let power: u8 = 14; // 16K buckets = 112K slots
    let num_buckets = 1usize << power;
    let num_slots = num_buckets * 7;

    // Test both hashing modes
    for two_choice in [false, true] {
        let mode_name = if two_choice { "two_choice" } else { "single_choice" };

        // Test at different occupancy levels
        for occupancy_target in [50, 70, 80] {
            let num_items = (num_slots * occupancy_target) / 100;

            group.throughput(Throughput::Elements(num_items as u64));

            group.bench_with_input(
                BenchmarkId::new(mode_name, format!("{}%", occupancy_target)),
                &(num_items, two_choice),
                |b, &(num_items, two_choice)| {
                    b.iter_batched(
                        || {
                            // Setup: create pool and hashtable
                            let pool = MemoryPoolBuilder::new(0)
                                .segment_size(1024 * 1024)
                                .heap_size(128 * 1024 * 1024)
                                .small_queue_percent(0)
                                .build()
                                .expect("Failed to create pool");

                            let hashtable = Hashtable::with_two_choice(power, two_choice);
                            let metrics = CacheMetrics::new();

                            // Pre-allocate segments and write items to them
                            let mut items = Vec::with_capacity(num_items);
                            let mut current_segment = pool
                                .reserve_main_cache(&metrics)
                                .expect("Failed to reserve segment");

                            for i in 0..num_items {
                                let key = generate_key(i);
                                let value = generate_value(i);
                                let optional: &[u8] = &[];

                                let segment =
                                    pool.get(current_segment).expect("Failed to get segment");

                                match segment.append_item(&key, &value, optional, &metrics) {
                                    Some(offset) => {
                                        items.push((key, current_segment, offset));
                                    }
                                    None => {
                                        current_segment = pool
                                            .reserve_main_cache(&metrics)
                                            .expect("Failed to reserve segment");
                                        let segment = pool
                                            .get(current_segment)
                                            .expect("Failed to get segment");
                                        let offset = segment
                                            .append_item(&key, &value, optional, &metrics)
                                            .expect("Failed to append to fresh segment");
                                        items.push((key, current_segment, offset));
                                    }
                                }
                            }

                            (pool, hashtable, metrics, items)
                        },
                        |(pool, hashtable, metrics, items)| {
                            let mut success_count = 0;
                            for (key, segment_id, offset) in items.iter() {
                                if hashtable
                                    .link_item(key, 0, *segment_id, *offset, &pool, &metrics)
                                    .is_ok()
                                {
                                    success_count += 1;
                                }
                            }
                            black_box(success_count)
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark hashtable lookups
fn bench_hashtable_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashtable_lookup");

    let power: u8 = 14;
    let occupancy = 70;
    let num_buckets = 1usize << power;
    let num_slots = num_buckets * 7;
    let num_items = (num_slots * occupancy) / 100;

    group.throughput(Throughput::Elements(num_items as u64));

    // Test both modes
    for two_choice in [false, true] {
        let mode_name = if two_choice {
            "two_choice_70pct"
        } else {
            "single_choice_70pct"
        };

        group.bench_function(mode_name, |b| {
            let pool = MemoryPoolBuilder::new(0)
                .segment_size(1024 * 1024)
                .heap_size(128 * 1024 * 1024)
                .small_queue_percent(0)
                .build()
                .expect("Failed to create pool");

            let hashtable = Hashtable::with_two_choice(power, two_choice);
            let metrics = CacheMetrics::new();

            let mut items = Vec::with_capacity(num_items);
            let mut current_segment = pool
                .reserve_main_cache(&metrics)
                .expect("Failed to reserve segment");

            for i in 0..num_items {
                let key = generate_key(i);
                let value = generate_value(i);
                let optional: &[u8] = &[];

                let segment = pool.get(current_segment).expect("Failed to get segment");
                match segment.append_item(&key, &value, optional, &metrics) {
                    Some(offset) => {
                        hashtable
                            .link_item(&key, 0, current_segment, offset, &pool, &metrics)
                            .ok();
                        items.push(key);
                    }
                    None => {
                        current_segment = pool
                            .reserve_main_cache(&metrics)
                            .expect("Failed to reserve segment");
                        let segment = pool.get(current_segment).expect("Failed to get segment");
                        let offset = segment
                            .append_item(&key, &value, optional, &metrics)
                            .expect("Failed to append to fresh segment");
                        hashtable
                            .link_item(&key, 0, current_segment, offset, &pool, &metrics)
                            .ok();
                        items.push(key);
                    }
                }
            }

            b.iter(|| {
                let mut found = 0;
                for key in items.iter() {
                    if hashtable.get(key, &pool).is_some() {
                        found += 1;
                    }
                }
                black_box(found)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hashtable_insert, bench_hashtable_lookup,);

criterion_main!(benches);
