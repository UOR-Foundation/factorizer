use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;

fn bench_simple_operations(c: &mut Criterion) {
    c.bench_function("gcd_small", |bench| {
        let a = Number::from(48u32);
        let b = Number::from(18u32);
        bench.iter(|| utils::gcd(black_box(&a), black_box(&b)));
    });

    c.bench_function("is_prime_small", |bench| {
        let n = Number::from(97u32);
        bench.iter(|| utils::is_probable_prime(black_box(&n), 10));
    });

    c.bench_function("integer_sqrt", |bench| {
        let n = Number::from(144u32);
        bench.iter(|| utils::integer_sqrt(black_box(&n)));
    });
}

criterion_group!(benches, bench_simple_operations);
criterion_main!(benches);