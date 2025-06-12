//! Performance analysis for The Pattern
//!
//! This suite measures performance characteristics to understand
//! scaling behavior and identify optimization opportunities.

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::pattern::{self, Pattern};
use rust_pattern_solver::types::Number;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    bit_size: usize,
    observation_time: f64,
    pattern_discovery_time: f64,
    recognition_time: f64,
    formalization_time: f64,
    execution_time: f64,
    total_time: f64,
    success: bool,
}

#[test]
#[ignore] // Run with: cargo test --test performance_analysis -- --ignored --nocapture
fn test_performance_scaling() {
    println!("\n=== Performance Scaling Analysis ===\n");
    
    let mut metrics = Vec::new();
    
    // Test different bit sizes
    let bit_sizes = vec![8, 16, 24, 32, 40, 48, 56, 64];
    
    for &bits in &bit_sizes {
        println!("Testing {}-bit numbers...", bits);
        
        // Generate test number
        let half_bits = bits / 2;
        let p = Number::from(1u32) << (half_bits as u32 - 1);
        let p = &p + &Number::from(15u32);
        let q = Number::from(1u32) << (half_bits as u32 - 1);
        let q = &q + &Number::from(25u32);
        let n = &p * &q;
        
        let mut metric = PerformanceMetrics {
            bit_size: bits,
            observation_time: 0.0,
            pattern_discovery_time: 0.0,
            recognition_time: 0.0,
            formalization_time: 0.0,
            execution_time: 0.0,
            total_time: 0.0,
            success: false,
        };
        
        let total_start = Instant::now();
        
        // Generate training data
        let training = generate_training_data(bits);
        
        // Measure observation time
        let obs_start = Instant::now();
        let mut collector = ObservationCollector::new();
        let observations = match collector.observe_parallel(&training) {
            Ok(obs) => obs,
            Err(e) => {
                println!("  Observation failed: {}", e);
                continue;
            }
        };
        metric.observation_time = obs_start.elapsed().as_secs_f64();
        
        // Measure pattern discovery time
        let pattern_start = Instant::now();
        let patterns = match Pattern::discover_from_observations(&observations) {
            Ok(pats) => pats,
            Err(e) => {
                println!("  Pattern discovery failed: {}", e);
                continue;
            }
        };
        metric.pattern_discovery_time = pattern_start.elapsed().as_secs_f64();
        
        // Measure recognition time
        let rec_start = Instant::now();
        let recognition = match pattern::recognition::recognize(n.clone(), &patterns) {
            Ok(rec) => rec,
            Err(e) => {
                println!("  Recognition failed: {}", e);
                continue;
            }
        };
        metric.recognition_time = rec_start.elapsed().as_secs_f64();
        
        // Measure formalization time
        let form_start = Instant::now();
        let formalization = match pattern::formalization::formalize(recognition, &patterns, &[]) {
            Ok(form) => form,
            Err(e) => {
                println!("  Formalization failed: {}", e);
                continue;
            }
        };
        metric.formalization_time = form_start.elapsed().as_secs_f64();
        
        // Measure execution time
        let exec_start = Instant::now();
        match pattern::execution::execute(formalization, &patterns) {
            Ok(factors) => {
                metric.execution_time = exec_start.elapsed().as_secs_f64();
                metric.success = factors.p * factors.q == n;
                if metric.success {
                    println!("  ✓ Successfully factored");
                }
            }
            Err(e) => {
                println!("  Execution failed: {}", e);
            }
        }
        
        metric.total_time = total_start.elapsed().as_secs_f64();
        metrics.push(metric);
    }
    
    // Print summary table
    println!("\n{:<8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8}",
             "Bits", "Observe", "Discover", "Recognize", "Formalize", "Execute", "Total", "Success");
    println!("{:-<96}", "");
    
    for m in &metrics {
        println!("{:<8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>8}",
                 m.bit_size,
                 m.observation_time,
                 m.pattern_discovery_time,
                 m.recognition_time,
                 m.formalization_time,
                 m.execution_time,
                 m.total_time,
                 if m.success { "✓" } else { "✗" });
    }
    
    // Analyze scaling behavior
    analyze_scaling(&metrics);
}

fn generate_training_data(target_bits: usize) -> Vec<Number> {
    let mut training = Vec::new();
    
    // Base training set
    for i in 0..20 {
        let p = 2 * i + 3;
        let q = 2 * i + 5;
        training.push(Number::from(p * q));
    }
    
    // Add numbers at target scale
    if target_bits >= 16 {
        let half_bits = target_bits / 2;
        for i in 0..10 {
            let p = Number::from(1u32) << (half_bits as u32 - 2);
            let p = &p + &Number::from((2 * i + 3) as u32);
            let q = Number::from(1u32) << (half_bits as u32 - 2);
            let q = &q + &Number::from((2 * i + 5) as u32);
            training.push(&p * &q);
        }
    }
    
    training
}

fn analyze_scaling(metrics: &[PerformanceMetrics]) {
    println!("\n=== Scaling Analysis ===\n");
    
    // Calculate growth rates
    let successful_metrics: Vec<&PerformanceMetrics> = metrics.iter()
        .filter(|m| m.success)
        .collect();
    
    if successful_metrics.len() < 2 {
        println!("Not enough successful factorizations for scaling analysis");
        return;
    }
    
    // Analyze each phase
    let phases = vec![
        ("Observation", |m: &PerformanceMetrics| m.observation_time),
        ("Pattern Discovery", |m: &PerformanceMetrics| m.pattern_discovery_time),
        ("Recognition", |m: &PerformanceMetrics| m.recognition_time),
        ("Formalization", |m: &PerformanceMetrics| m.formalization_time),
        ("Execution", |m: &PerformanceMetrics| m.execution_time),
    ];
    
    for (phase_name, time_fn) in phases {
        let mut ratios = Vec::new();
        
        for i in 1..successful_metrics.len() {
            let prev = successful_metrics[i-1];
            let curr = successful_metrics[i];
            
            let bit_ratio = curr.bit_size as f64 / prev.bit_size as f64;
            let time_ratio = time_fn(curr) / time_fn(prev);
            
            if time_fn(prev) > 0.0 {
                ratios.push((bit_ratio, time_ratio));
            }
        }
        
        if !ratios.is_empty() {
            let avg_scaling = ratios.iter()
                .map(|(br, tr)| tr.ln() / br.ln())
                .sum::<f64>() / ratios.len() as f64;
            
            println!("{} scaling: O(n^{:.2})", phase_name, avg_scaling);
        }
    }
}

#[test]
#[ignore]
fn test_pattern_caching_effectiveness() {
    println!("\n=== Pattern Caching Effectiveness ===\n");
    
    // Generate test numbers
    let test_numbers = vec![
        Number::from(143u32),  // 11 × 13
        Number::from(143u32),  // Same number (should be cached)
        Number::from(323u32),  // 17 × 19
        Number::from(143u32),  // Repeated (should be cached)
        Number::from(667u32),  // 23 × 29
        Number::from(323u32),  // Repeated (should be cached)
    ];
    
    let mut collector = ObservationCollector::new();
    let training = vec![
        Number::from(15u32),
        Number::from(21u32),
        Number::from(35u32),
    ];
    
    let observations = collector.observe_parallel(&training).unwrap();
    let patterns = Pattern::discover_from_observations(&observations).unwrap();
    
    let mut timings = HashMap::new();
    
    for (i, n) in test_numbers.iter().enumerate() {
        let start = Instant::now();
        
        let recognition = pattern::recognition::recognize(n.clone(), &patterns).unwrap();
        let formalization = pattern::formalization::formalize(recognition, &patterns, &[]).unwrap();
        let _ = pattern::execution::execute(formalization, &patterns);
        
        let elapsed = start.elapsed().as_secs_f64();
        
        let entry = timings.entry(n.to_string()).or_insert(Vec::new());
        entry.push((i, elapsed));
        
        println!("Attempt {} for {}: {:.3}ms", i + 1, n, elapsed * 1000.0);
    }
    
    // Analyze cache effectiveness
    println!("\nCache effectiveness analysis:");
    for (num, times) in timings {
        if times.len() > 1 {
            let first_time = times[0].1;
            let subsequent_avg = times[1..].iter()
                .map(|(_, t)| t)
                .sum::<f64>() / (times.len() - 1) as f64;
            
            let speedup = first_time / subsequent_avg;
            println!("  {}: {:.1}x speedup from caching", num, speedup);
        }
    }
}

#[test]
#[ignore]
fn test_parallel_efficiency() {
    println!("\n=== Parallel Processing Efficiency ===\n");
    
    // Generate a large set of numbers to observe
    let mut numbers = Vec::new();
    for i in 0..100 {
        let p = 100 + i;
        let q = 100 + i + 2;
        numbers.push(Number::from(p * q));
    }
    
    let mut collector = ObservationCollector::new();
    
    // Test with different thread counts
    let thread_counts = vec![1, 2, 4, 8];
    let mut results = Vec::new();
    
    for &threads in &thread_counts {
        // Set thread pool size
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        
        let start = Instant::now();
        let _ = collector.observe_parallel(&numbers).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        
        results.push((threads, elapsed));
        println!("Threads: {}, Time: {:.3}s", threads, elapsed);
    }
    
    // Calculate parallel efficiency
    println!("\nParallel efficiency:");
    let single_thread_time = results[0].1;
    
    for &(threads, time) in &results[1..] {
        let speedup = single_thread_time / time;
        let efficiency = speedup / threads as f64 * 100.0;
        println!("  {} threads: {:.1}x speedup, {:.0}% efficiency", 
                 threads, speedup, efficiency);
    }
}

#[test]
#[ignore]
fn test_memory_usage() {
    println!("\n=== Memory Usage Analysis ===\n");
    
    // This is a simple approximation - in practice you'd use a memory profiler
    let test_sizes = vec![10, 100, 1000];
    
    for &size in &test_sizes {
        // Generate numbers
        let mut numbers = Vec::new();
        for i in 0..size {
            let p = 100 + i;
            let q = 100 + i + 2;
            numbers.push(Number::from(p as u32 * q as u32));
        }
        
        let start_memory = estimate_memory_usage();
        
        let mut collector = ObservationCollector::new();
        let observations = collector.observe_parallel(&numbers).unwrap();
        let patterns = Pattern::discover_from_observations(&observations).unwrap();
        
        let end_memory = estimate_memory_usage();
        let memory_used = end_memory - start_memory;
        
        println!("Dataset size: {}", size);
        println!("  Observations: {}", observations.len());
        println!("  Patterns: {}", patterns.len());
        println!("  Estimated memory: {} MB", memory_used / 1024 / 1024);
        println!("  Memory per observation: {} KB", memory_used / observations.len() / 1024);
    }
}

fn estimate_memory_usage() -> usize {
    // This is a placeholder - in practice you'd use proper memory profiling
    // For now, return a mock value
    1024 * 1024 * 10 // 10 MB
}