//! Factorization benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_pattern_solver::{
    observer::ObservationCollector,
    pattern::{self, execution, formalization, recognition},
    types::Number,
};

fn benchmark_observation(c: &mut Criterion) {
    c.bench_function("observe_16bit_semiprime", |b| {
        let n = Number::from(10403u32); // 101 × 103
        let mut collector = ObservationCollector::new();

        b.iter(|| collector.observe_single(black_box(n.clone())));
    });

    c.bench_function("observe_32bit_semiprime", |b| {
        let n = Number::from(1073676287u64); // 32749 × 32771
        let mut collector = ObservationCollector::new();

        b.iter(|| collector.observe_single(black_box(n.clone())));
    });
}

fn benchmark_pattern_discovery(c: &mut Criterion) {
    let mut collector = ObservationCollector::new();
    let numbers: Vec<Number> = (0..100)
        .map(|i| {
            let p = 2 * i + 3;
            let q = 2 * i + 5;
            Number::from(p * q)
        })
        .collect();

    let observations = collector.observe_parallel(&numbers).unwrap();

    c.bench_function("pattern_discovery_100_observations", |b| {
        b.iter(|| pattern::Pattern::discover_from_observations(black_box(&observations)));
    });
}

fn benchmark_recognition(c: &mut Criterion) {
    // Setup patterns
    let mut collector = ObservationCollector::new();
    let training: Vec<Number> =
        vec![15, 21, 35, 77, 91, 143, 221, 323].into_iter().map(Number::from).collect();

    let observations = collector.observe_parallel(&training).unwrap();
    let patterns = pattern::Pattern::discover_from_observations(&observations).unwrap();

    c.bench_function("recognize_small_semiprime", |b| {
        let n = Number::from(391u32); // 17 × 23
        b.iter(|| recognition::recognize(black_box(n.clone()), black_box(&patterns)));
    });

    c.bench_function("recognize_medium_semiprime", |b| {
        let n = Number::from(1073741827u64); // Large prime product
        b.iter(|| recognition::recognize(black_box(n.clone()), black_box(&patterns)));
    });
}

fn benchmark_full_pipeline(c: &mut Criterion) {
    // Setup
    let mut collector = ObservationCollector::new();
    let training: Vec<Number> = (0..50).map(|i| Number::from((2 * i + 3) * (2 * i + 5))).collect();

    let observations = collector.observe_parallel(&training).unwrap();
    let patterns = pattern::Pattern::discover_from_observations(&observations).unwrap();

    let mut group = c.benchmark_group("full_pipeline");

    // Small semiprime
    group.bench_function("small_16bit", |b| {
        let n = Number::from(667u32); // 23 × 29
        b.iter(|| {
            let recognition = recognition::recognize(n.clone(), &patterns).unwrap();
            let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();
            execution::execute(black_box(formalized), black_box(&patterns))
        });
    });

    // Medium semiprime
    group.bench_function("medium_32bit", |b| {
        let n = Number::from(536870923u64); // 23117 × 23227
        b.iter(|| {
            let recognition = recognition::recognize(n.clone(), &patterns).unwrap();
            let formalized = formalization::formalize(recognition, &patterns, &[]).unwrap();
            execution::execute(black_box(formalized), black_box(&patterns))
        });
    });

    group.finish();
}

fn benchmark_parallel_observation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_observation");

    for size in &[10, 100, 1000] {
        group.bench_function(format!("observe_{}_numbers", size), |b| {
            let numbers: Vec<Number> =
                (0..*size).map(|i| Number::from((2 * i + 3) * (2 * i + 5))).collect();

            let mut collector = ObservationCollector::new();

            b.iter(|| collector.observe_parallel(black_box(&numbers)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_observation,
    benchmark_pattern_discovery,
    benchmark_recognition,
    benchmark_full_pipeline,
    benchmark_parallel_observation
);
criterion_main!(benches);
