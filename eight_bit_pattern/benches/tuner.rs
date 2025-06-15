//! Benchmark suite for the auto-tuner
//! 
//! Measures performance of the tuning process and factorization.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use eight_bit_pattern::{AutoTuner, TestCase, compute_basis, recognize_factors, TunerParams};
use num_bigint::BigInt;

fn bench_basis_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("basis_computation");
    
    for num_channels in [8, 16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_channels),
            num_channels,
            |b, &size| {
                let params = TunerParams::default();
                b.iter(|| {
                    compute_basis(black_box(size), &params)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization");
    
    // Pre-compute basis once
    let params = TunerParams::default();
    let basis = compute_basis(128, &params);
    
    // Test cases of different sizes
    let test_cases = vec![
        ("8-bit", "143"),       // 11 * 13
        ("10-bit", "1073"),     // 29 * 37
        ("12-bit", "4087"),     // 61 * 67
        ("16-bit", "62837"),    // 241 * 261
    ];
    
    for (name, n_str) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("recognize_factors", name),
            n_str,
            |b, n_str| {
                let n = n_str.parse::<BigInt>().unwrap();
                b.iter(|| {
                    recognize_factors(black_box(&n), &basis, &params)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_tuner_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tuner_optimization");
    
    // Create small test set
    let test_cases = vec![
        TestCase {
            n: BigInt::from(143),
            p: BigInt::from(11),
            q: BigInt::from(13),
            bit_length: 8,
        },
        TestCase {
            n: BigInt::from(15),
            p: BigInt::from(3),
            q: BigInt::from(5),
            bit_length: 4,
        },
        TestCase {
            n: BigInt::from(77),
            p: BigInt::from(7),
            q: BigInt::from(11),
            bit_length: 7,
        },
    ];
    
    group.bench_function("optimize_10_rounds", |b| {
        b.iter(|| {
            let mut tuner = AutoTuner::new();
            tuner.load_test_cases(test_cases.clone());
            tuner.optimize(black_box(10))
        });
    });
    
    group.finish();
}

fn bench_channel_operations(c: &mut Criterion) {
    use eight_bit_pattern::{decompose, reconstruct};
    
    let mut group = c.benchmark_group("channel_operations");
    
    // Different sized numbers
    let numbers = vec![
        ("64-bit", BigInt::from(u64::MAX)),
        ("128-bit", BigInt::from(u128::MAX)),
        ("256-bit", BigInt::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16
        ).unwrap()),
    ];
    
    for (name, n) in numbers {
        group.bench_with_input(
            BenchmarkId::new("decompose", name),
            &n,
            |b, n| {
                b.iter(|| {
                    decompose(black_box(n))
                });
            },
        );
        
        let channels = decompose(&n);
        group.bench_with_input(
            BenchmarkId::new("reconstruct", name),
            &channels,
            |b, channels| {
                b.iter(|| {
                    reconstruct(black_box(channels))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_basis_computation,
    bench_factorization,
    bench_tuner_optimization,
    bench_channel_operations
);
criterion_main!(benches);