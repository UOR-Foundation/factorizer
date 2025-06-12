//! Demonstration of parallel processing and caching
//!
//! This example shows how The Pattern leverages multiple CPU cores
//! and caching to efficiently factor many numbers in parallel.

use rust_pattern_solver::pattern::cache::CacheConfig;
use rust_pattern_solver::pattern::parallel::{
    DistributionStrategy, ParallelConfig, ParallelExecutor,
};
use rust_pattern_solver::types::pattern::ScaleRange;
use rust_pattern_solver::types::{Number, Pattern, PatternKind};
use std::time::Instant;

fn main() {
    println!("Parallel Processing and Caching Demo");
    println!("===================================\n");

    // Configure parallel execution
    let config = ParallelConfig {
        num_threads: num_cpus::get(),
        batch_size: 50,
        show_progress: true,
        cache_config: CacheConfig {
            max_entries: 1000,
            ttl: std::time::Duration::from_secs(300), // 5 minutes
            persistent: false,
            cache_path: None,
            enable_stats: true,
        },
        distribution_strategy: DistributionStrategy::Dynamic,
    };

    println!("Configuration:");
    println!("  CPU cores: {}", config.num_threads);
    println!("  Distribution: {:?}", config.distribution_strategy);
    println!("  Cache size: {} entries", config.cache_config.max_entries);
    println!();

    // Create test patterns
    let patterns = create_test_patterns();

    // Test 1: Sequential vs Parallel comparison
    sequential_vs_parallel_test(&patterns);

    // Test 2: Cache effectiveness
    cache_effectiveness_test(&patterns);

    // Test 3: Different distribution strategies
    distribution_strategy_test(&patterns);
}

fn sequential_vs_parallel_test(patterns: &[Pattern]) {
    println!("Sequential vs Parallel Test");
    println!("==========================\n");

    // Generate test numbers
    let test_numbers: Vec<Number> = (100..200).map(|i| Number::from(i as u32)).collect();

    println!("Testing with {} numbers\n", test_numbers.len());

    // Sequential execution
    let seq_start = Instant::now();
    let seq_results: Vec<_> = test_numbers
        .iter()
        .map(|n| {
            crate::pattern::recognition::recognize(n.clone(), patterns).map(|r| r.pattern_type)
        })
        .collect();
    let seq_time = seq_start.elapsed();

    println!("Sequential execution:");
    println!("  Time: {:.3}s", seq_time.as_secs_f64());
    println!(
        "  Results: {} successful",
        seq_results.iter().filter(|r| r.is_ok()).count()
    );

    // Parallel execution
    let parallel_config = ParallelConfig {
        num_threads: num_cpus::get(),
        show_progress: false,
        ..Default::default()
    };

    let executor = ParallelExecutor::new(parallel_config);
    let par_start = Instant::now();
    let par_results = executor.execute_batch(&test_numbers, patterns);
    let par_time = par_start.elapsed();

    println!("\nParallel execution:");
    println!("  Time: {:.3}s", par_time.as_secs_f64());
    println!(
        "  Results: {} successful",
        par_results.iter().filter(|r| r.is_ok()).count()
    );
    println!(
        "  Speedup: {:.2}x\n",
        seq_time.as_secs_f64() / par_time.as_secs_f64()
    );

    let stats = executor.stats();
    println!("Execution statistics:");
    println!("  Total: {}", stats.total_numbers);
    println!("  Successful: {}", stats.successful);
    println!("  Failed: {}", stats.failed);
    println!(
        "  Avg time per number: {:.3}ms\n",
        stats.average_time_per_number.as_secs_f64() * 1000.0
    );
}

fn cache_effectiveness_test(patterns: &[Pattern]) {
    println!("\nCache Effectiveness Test");
    println!("=======================\n");

    // Create executor with caching
    let config = ParallelConfig {
        show_progress: false,
        cache_config: CacheConfig {
            max_entries: 100,
            enable_stats: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let executor = ParallelExecutor::new(config);

    // Test with repeated numbers
    let base_numbers: Vec<Number> = (1..=20).map(|i| Number::from((i * 11) as u32)).collect();

    // Run 3 times to test cache
    for round in 1..=3 {
        println!("Round {}: Processing {} numbers", round, base_numbers.len());

        let start = Instant::now();
        let _results = executor.execute_batch(&base_numbers, patterns);
        let elapsed = start.elapsed();

        let cache_stats = executor.cache_stats();
        println!("  Time: {:.3}s", elapsed.as_secs_f64());
        println!("  Cache hits: {}", cache_stats.hits);
        println!("  Cache misses: {}", cache_stats.misses);
        println!("  Hit rate: {:.1}%", cache_stats.hit_rate() * 100.0);

        if round == 1 {
            println!("  (First run - populating cache)");
        } else {
            println!("  (Using cached results)");
        }
        println!();
    }
}

fn distribution_strategy_test(patterns: &[Pattern]) {
    println!("\nDistribution Strategy Test");
    println!("=========================\n");

    // Generate numbers with varying complexity
    let mut test_numbers = Vec::new();

    // Small numbers (easy)
    for i in 10..20 {
        test_numbers.push(Number::from(i as u32));
    }

    // Medium numbers
    for i in 100..110 {
        test_numbers.push(Number::from(i as u32));
    }

    // Larger numbers (harder)
    for i in 1000..1010 {
        test_numbers.push(Number::from(i as u32));
    }

    println!(
        "Testing with {} numbers of varying complexity\n",
        test_numbers.len()
    );

    // Test each strategy
    for strategy in &[
        DistributionStrategy::Static,
        DistributionStrategy::Dynamic,
        DistributionStrategy::Adaptive,
    ] {
        let config = ParallelConfig {
            num_threads: 4,
            show_progress: false,
            distribution_strategy: *strategy,
            ..Default::default()
        };

        let executor = ParallelExecutor::new(config);

        let start = Instant::now();
        let results = executor.execute_batch(&test_numbers, patterns);
        let elapsed = start.elapsed();

        println!("{:?} strategy:", strategy);
        println!("  Time: {:.3}s", elapsed.as_secs_f64());
        println!(
            "  Successful: {}",
            results.iter().filter(|r| r.is_ok()).count()
        );

        let stats = executor.stats();
        println!(
            "  Avg time per number: {:.3}ms\n",
            stats.average_time_per_number.as_secs_f64() * 1000.0
        );
    }
}

fn create_test_patterns() -> Vec<Pattern> {
    vec![
        Pattern {
            id: "test_emergent".to_string(),
            kind: PatternKind::Emergent,
            frequency: 0.8,
            description: "Test emergent pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 1,
                max_bits: 100,
                unbounded: false,
            },
        },
        Pattern {
            id: "test_harmonic".to_string(),
            kind: PatternKind::Harmonic {
                base_frequency: 0.5,
                harmonics: vec![1.0, 0.5, 0.25],
            },
            frequency: 0.5,
            description: "Test harmonic pattern".to_string(),
            parameters: vec![],
            scale_range: ScaleRange {
                min_bits: 1,
                max_bits: 100,
                unbounded: false,
            },
        },
    ]
}

// Import the pattern modules
use rust_pattern_solver::pattern;
