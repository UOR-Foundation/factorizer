//! Advanced pattern recognition for specialized number types
//!
//! This module implements recognition for Harmonic, Power, Fibonacci,
//! and Mersenne patterns through empirical observation.

use crate::types::{Number, Pattern, PatternKind};
use crate::utils;

/// Detect harmonic patterns in a number
pub fn detect_harmonic_pattern(n: &Number) -> Option<Pattern> {
    // Analyze frequency components through modular arithmetic
    let primes = utils::generate_primes(100);
    let mut frequency_components = Vec::new();

    // Extract frequency signature
    for p in &primes {
        let residue = n % p;
        let p_val = p.as_integer().to_u64().unwrap_or(2);
        let freq = residue.as_integer().to_f64() / p_val as f64;
        frequency_components.push(freq);
    }

    // Perform harmonic analysis
    let base_frequency = find_fundamental_frequency(&frequency_components);
    let harmonics = extract_harmonic_series(&frequency_components, base_frequency);

    // Check if this exhibits harmonic properties
    if harmonics.len() > 3 && harmonic_strength(&harmonics) > 0.7 {
        Some(Pattern {
            id: format!("harmonic_{:.3}", base_frequency),
            kind: PatternKind::Harmonic {
                base_frequency,
                harmonics: harmonics.clone(),
            },
            frequency: 1.0,
            scale_range: crate::types::pattern::ScaleRange {
                min_bits: n.bit_length().saturating_sub(4),
                max_bits: n.bit_length() + 4,
                unbounded: false,
            },
            parameters: harmonics.clone(),
            description: format!(
                "Harmonic pattern with base frequency {:.3} and {} harmonics",
                base_frequency,
                harmonics.len()
            ),
        })
    } else {
        None
    }
}

/// Detect if a number is a power of a prime
pub fn detect_power_pattern(n: &Number) -> Option<Pattern> {
    // Check small primes first
    let small_primes = utils::generate_primes(1000);

    for p in &small_primes {
        let mut power = 1u32;
        let mut current = p.clone();

        while &current < n && power < 100 {
            if &current == n {
                return Some(Pattern {
                    id: format!("power_{}^{}", p, power),
                    kind: PatternKind::Power {
                        base: p.as_integer().to_u64().unwrap_or(0),
                        exponent: power,
                    },
                    frequency: 1.0,
                    scale_range: crate::types::pattern::ScaleRange {
                        min_bits: n.bit_length(),
                        max_bits: n.bit_length(),
                        unbounded: false,
                    },
                    parameters: vec![p.to_f64().unwrap_or(0.0), power as f64],
                    description: format!("{} = {}^{}", n, p, power),
                });
            }
            current = &current * p;
            power += 1;
        }
    }

    // For larger numbers, check if it's a perfect power
    for exp in 2..=64 {
        if let Some(base) = is_perfect_power(n, exp) {
            if utils::is_probable_prime(&base, 20) {
                return Some(Pattern {
                    id: format!("power_large^{}", exp),
                    kind: PatternKind::Power {
                        base: base.as_integer().to_u64().unwrap_or(0),
                        exponent: exp,
                    },
                    frequency: 1.0,
                    scale_range: crate::types::pattern::ScaleRange {
                        min_bits: n.bit_length(),
                        max_bits: n.bit_length(),
                        unbounded: false,
                    },
                    parameters: vec![0.0, exp as f64],
                    description: format!("{} = base^{}", n, exp),
                });
            }
        }
    }

    None
}

/// Detect Fibonacci-related patterns
pub fn detect_fibonacci_pattern(n: &Number) -> Option<Pattern> {
    // Generate Fibonacci numbers up to n
    let mut fib_prev = Number::from(1u32);
    let mut fib_curr = Number::from(1u32);
    let mut index = 2;

    while &fib_curr <= n {
        // Check direct Fibonacci number
        if &fib_curr == n {
            return Some(Pattern {
                id: format!("fibonacci_{}", index),
                kind: PatternKind::Fibonacci {
                    index,
                    relationship: "direct".to_string(),
                },
                frequency: 1.0,
                scale_range: crate::types::pattern::ScaleRange {
                    min_bits: n.bit_length(),
                    max_bits: n.bit_length(),
                    unbounded: false,
                },
                parameters: vec![index as f64],
                description: format!("{} = F({})", n, index),
            });
        }

        // Check Fibonacci relationships
        // Product of consecutive Fibonacci numbers
        let product = &fib_prev * &fib_curr;
        if &product == n {
            return Some(Pattern {
                id: format!("fibonacci_product_{}_{}", index - 1, index),
                kind: PatternKind::Fibonacci {
                    index: index - 1,
                    relationship: "consecutive_product".to_string(),
                },
                frequency: 1.0,
                scale_range: crate::types::pattern::ScaleRange {
                    min_bits: n.bit_length(),
                    max_bits: n.bit_length(),
                    unbounded: false,
                },
                parameters: vec![(index - 1) as f64, index as f64],
                description: format!("{} = F({}) × F({})", n, index - 1, index),
            });
        }

        // Move to next Fibonacci number
        let next = &fib_prev + &fib_curr;
        fib_prev = fib_curr;
        fib_curr = next;
        index += 1;

        // Prevent infinite loop for very large numbers
        if index > 10000 {
            break;
        }
    }

    None
}

/// Detect Mersenne patterns (2^p - 1)
pub fn detect_mersenne_pattern(n: &Number) -> Option<Pattern> {
    // Check if n+1 is a power of 2
    let n_plus_one = n + 1u32;
    let bit_length = n_plus_one.bit_length();

    // Check if n+1 has exactly one bit set (is a power of 2)
    if n_plus_one.is_power_of_two() {
        let p = bit_length - 1;

        // Check if p is prime (necessary for Mersenne prime)
        let is_p_prime = utils::is_probable_prime(&Number::from(p as u64), 20);

        // Check if n itself is prime (Mersenne prime)
        let is_mersenne_prime = utils::is_probable_prime(n, 20);

        return Some(Pattern {
            id: format!("mersenne_{}", p),
            kind: PatternKind::Mersenne {
                p: p as u32,
                is_prime: is_mersenne_prime,
            },
            frequency: 1.0,
            scale_range: crate::types::pattern::ScaleRange {
                min_bits: n.bit_length(),
                max_bits: n.bit_length(),
                unbounded: false,
            },
            parameters: vec![p as f64, if is_mersenne_prime { 1.0 } else { 0.0 }],
            description: format!(
                "{} = 2^{} - 1 ({})",
                n,
                p,
                if is_mersenne_prime {
                    "Mersenne prime"
                } else if is_p_prime {
                    "Mersenne number"
                } else {
                    "pseudo-Mersenne"
                }
            ),
        });
    }

    None
}

/// Find fundamental frequency from frequency components
fn find_fundamental_frequency(components: &[f64]) -> f64 {
    // Use autocorrelation to find fundamental
    let mut max_correlation = 0.0;
    let mut fundamental = 1.0;

    for period in 1..components.len() / 2 {
        let correlation = calculate_autocorrelation(components, period);
        if correlation > max_correlation {
            max_correlation = correlation;
            fundamental = 1.0 / period as f64;
        }
    }

    fundamental
}

/// Extract harmonic series from frequency components
fn extract_harmonic_series(components: &[f64], base_freq: f64) -> Vec<f64> {
    let mut harmonics = Vec::new();

    for i in 1..=10 {
        let harmonic_freq = base_freq * i as f64;
        let amplitude = measure_frequency_amplitude(components, harmonic_freq);
        if amplitude > 0.1 {
            harmonics.push(amplitude);
        }
    }

    harmonics
}

/// Calculate harmonic strength
fn harmonic_strength(harmonics: &[f64]) -> f64 {
    if harmonics.is_empty() {
        return 0.0;
    }

    let sum: f64 = harmonics.iter().sum();
    let mean = sum / harmonics.len() as f64;

    // Strong harmonics have decreasing amplitudes
    let mut strength = mean;
    for i in 1..harmonics.len() {
        if harmonics[i] > harmonics[i - 1] {
            strength *= 0.9; // Penalty for non-decreasing
        }
    }

    strength.min(1.0)
}

/// Calculate autocorrelation for a given period
fn calculate_autocorrelation(data: &[f64], period: usize) -> f64 {
    let n = data.len();
    if period >= n {
        return 0.0;
    }

    let mut correlation = 0.0;
    let mut count = 0;

    for i in 0..n - period {
        correlation += data[i] * data[i + period];
        count += 1;
    }

    if count > 0 {
        correlation / count as f64
    } else {
        0.0
    }
}

/// Measure amplitude at a specific frequency
fn measure_frequency_amplitude(components: &[f64], target_freq: f64) -> f64 {
    // Simple frequency matching
    let index = (target_freq * components.len() as f64) as usize;
    if index < components.len() {
        components[index].abs()
    } else {
        0.0
    }
}

/// Check if n is a perfect power with given exponent
fn is_perfect_power(n: &Number, exp: u32) -> Option<Number> {
    // Binary search for the base
    let mut low = Number::from(1u32);
    let mut high = n.clone();

    while low <= high {
        let mid = (&low + &high) / &Number::from(2u32);
        let power = mid.pow(exp);

        use std::cmp::Ordering;
        match power.cmp(n) {
            Ordering::Equal => return Some(mid),
            Ordering::Less => low = &mid + &Number::from(1u32),
            Ordering::Greater => {
                if mid == Number::from(1u32) {
                    break;
                }
                high = &mid - &Number::from(1u32);
            },
        }
    }

    None
}

/// Combine all advanced pattern detections
pub fn detect_advanced_patterns(n: &Number) -> Vec<Pattern> {
    let mut patterns = Vec::new();

    // Try each pattern type
    if let Some(pattern) = detect_harmonic_pattern(n) {
        patterns.push(pattern);
    }

    if let Some(pattern) = detect_power_pattern(n) {
        patterns.push(pattern);
    }

    if let Some(pattern) = detect_fibonacci_pattern(n) {
        patterns.push(pattern);
    }

    if let Some(pattern) = detect_mersenne_pattern(n) {
        patterns.push(pattern);
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_pattern() {
        // Test 2^10 = 1024
        let n = Number::from(1024u32);
        let pattern = detect_power_pattern(&n);
        assert!(pattern.is_some());

        if let Some(p) = pattern {
            match p.kind {
                PatternKind::Power { base, exponent } => {
                    assert_eq!(base, 2);
                    assert_eq!(exponent, 10);
                },
                _ => panic!("Wrong pattern type"),
            }
        }
    }

    #[test]
    fn test_mersenne_pattern() {
        // Test 2^7 - 1 = 127 (Mersenne prime)
        let n = Number::from(127u32);
        let pattern = detect_mersenne_pattern(&n);
        assert!(pattern.is_some());

        if let Some(p) = pattern {
            match p.kind {
                PatternKind::Mersenne { p: exp, is_prime } => {
                    assert_eq!(exp, 7);
                    assert!(is_prime);
                },
                _ => panic!("Wrong pattern type"),
            }
        }
    }

    #[test]
    fn test_fibonacci_pattern() {
        // Test F(10) = 55
        let n = Number::from(55u32);
        let pattern = detect_fibonacci_pattern(&n);
        assert!(pattern.is_some());

        if let Some(p) = pattern {
            match p.kind {
                PatternKind::Fibonacci {
                    index,
                    relationship,
                } => {
                    assert_eq!(index, 10);
                    assert_eq!(relationship, "direct");
                },
                _ => panic!("Wrong pattern type"),
            }
        }
    }
}
