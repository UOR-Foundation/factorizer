//! Arbitrary precision mathematical constants
//!
//! This module provides mathematical constants computed to arbitrary precision
//! to avoid limitations of f64 representations.

use crate::types::{Number, Rational};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::collections::HashMap;

/// Cache for computed constants at different precisions
static CONSTANT_CACHE: Lazy<Mutex<HashMap<(ConstantType, u32), Number>>> = 
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ConstantType {
    Pi,
    E,
    Phi,
    Sqrt2,
    Sqrt3,
    Sqrt5,
    Ln2,
    Ln10,
}

/// Get a mathematical constant to specified precision (in bits)
pub fn get_constant(constant_type: ConstantType, precision_bits: u32) -> Number {
    let mut cache = CONSTANT_CACHE.lock().unwrap();
    
    if let Some(value) = cache.get(&(constant_type, precision_bits)) {
        return value.clone();
    }
    
    let value = compute_constant(constant_type, precision_bits);
    cache.insert((constant_type, precision_bits), value.clone());
    value
}

/// Compute a constant to specified precision
fn compute_constant(constant_type: ConstantType, precision_bits: u32) -> Number {
    // For now, we'll use high-precision string representations
    // In production, these would be computed using arbitrary precision algorithms
    match constant_type {
        ConstantType::Pi => pi_to_precision(precision_bits),
        ConstantType::E => e_to_precision(precision_bits),
        ConstantType::Phi => phi_to_precision(precision_bits),
        ConstantType::Sqrt2 => sqrt_to_precision(2, precision_bits),
        ConstantType::Sqrt3 => sqrt_to_precision(3, precision_bits),
        ConstantType::Sqrt5 => sqrt_to_precision(5, precision_bits),
        ConstantType::Ln2 => ln_to_precision(2, precision_bits),
        ConstantType::Ln10 => ln_to_precision(10, precision_bits),
    }
}

/// Compute π using Machin's formula or similar
fn pi_to_precision(bits: u32) -> Number {
    // For demonstration, using a high-precision representation
    // In practice, implement Machin's formula or Chudnovsky algorithm
    let scale = Number::from(10u32).pow(bits / 3); // Approximately bits/3.32 decimal digits
    
    // π ≈ 355/113 (good to 7 decimal places)
    // For better precision, use continued fraction or series expansion
    let num = Number::from(355u32) * &scale;
    let den = Number::from(113u32);
    &num / &den
}

/// Compute e using Taylor series
fn e_to_precision(bits: u32) -> Number {
    let iterations = (bits / 4) as usize + 10;
    let scale = Number::from(1u32) << bits;
    
    let mut sum = scale.clone();
    let mut factorial = Number::from(1u32);
    let mut term = scale.clone();
    
    for i in 1..iterations {
        factorial = &factorial * &Number::from(i as u32);
        term = &scale / &factorial;
        if term.is_zero() {
            break;
        }
        sum = &sum + &term;
    }
    
    sum
}

/// Compute golden ratio φ = (1 + √5) / 2
fn phi_to_precision(bits: u32) -> Number {
    let sqrt5 = sqrt_to_precision(5, bits + 10);
    let one = Number::from(1u32) << bits;
    let sum = &one + &sqrt5;
    &sum / &Number::from(2u32)
}

/// Integer square root using Newton's method
pub fn integer_sqrt(n: &Number) -> Number {
    if n.is_zero() {
        return Number::from(0u32);
    }
    
    // Initial guess: 2^(bits/2)
    let mut x = Number::from(1u32) << (n.bit_length() / 2) as u32;
    
    loop {
        // x_new = (x + n/x) / 2
        let n_div_x = n / &x;
        let x_new = (&x + &n_div_x) / &Number::from(2u32);
        
        if x_new >= x {
            return x;
        }
        x = x_new;
    }
}

/// Integer nth root using Newton's method
pub fn integer_nth_root(n: &Number, root: u32) -> Number {
    if n.is_zero() || root == 0 {
        return Number::from(0u32);
    }
    
    if root == 1 {
        return n.clone();
    }
    
    // Initial guess: 2^(bits/root)
    let mut x = Number::from(1u32) << (n.bit_length() / root as usize) as u32;
    if x.is_zero() {
        x = Number::from(1u32);
    }
    
    loop {
        // x_new = ((root-1)*x + n/x^(root-1)) / root
        let x_pow = x.pow(root - 1);
        let n_div_xpow = n / &x_pow;
        let root_minus_1 = Number::from(root - 1);
        let root_num = Number::from(root);
        
        let x_new = (&(&root_minus_1 * &x) + &n_div_xpow) / &root_num;
        
        // Check for convergence (when x stops changing)
        if x_new == x || (x_new > x && &x_new - &x == Number::from(1u32)) {
            // Check which is closer
            let x_pow_full = x.pow(root);
            let x_new_pow_full = x_new.pow(root);
            
            let diff_x = if &x_pow_full > n {
                &x_pow_full - n
            } else {
                n - &x_pow_full
            };
            
            let diff_x_new = if &x_new_pow_full > n {
                &x_new_pow_full - n
            } else {
                n - &x_new_pow_full
            };
            
            return if diff_x <= diff_x_new { x } else { x_new };
        }
        x = x_new;
    }
}

/// Greatest common divisor using Euclidean algorithm
pub fn gcd(a: &Number, b: &Number) -> Number {
    let mut a = a.abs();
    let mut b = b.abs();
    
    while !b.is_zero() {
        let temp = b.clone();
        b = &a % &b;
        a = temp;
    }
    
    a
}

/// Least common multiple
pub fn lcm(a: &Number, b: &Number) -> Number {
    if a.is_zero() || b.is_zero() {
        return Number::from(0u32);
    }
    
    let gcd_val = gcd(a, b);
    &(a / &gcd_val) * b
}

/// Modular exponentiation: base^exp mod modulus
pub fn mod_pow(base: &Number, exp: &Number, modulus: &Number) -> Number {
    if modulus.is_one() {
        return Number::from(0u32);
    }
    
    let mut result = Number::from(1u32);
    let mut base = base % modulus;
    let mut exp = exp.clone();
    
    while !exp.is_zero() {
        // If exp is odd
        if &exp % &Number::from(2u32) == Number::from(1u32) {
            result = (&result * &base) % modulus;
        }
        
        exp = &exp / &Number::from(2u32);
        base = (&base * &base) % modulus;
    }
    
    result
}

/// Extended Euclidean algorithm
/// Returns (gcd, x, y) where gcd = a*x + b*y
pub fn extended_gcd(a: &Number, b: &Number) -> (Number, Number, Number) {
    if b.is_zero() {
        return (a.clone(), Number::from(1u32), Number::from(0u32));
    }
    
    let (gcd, x1, y1) = extended_gcd(b, &(a % b));
    let x = y1.clone();
    let y = &x1 - &(&(a / b) * &y1);
    
    (gcd, x, y)
}

/// Modular inverse: a^(-1) mod m
/// Returns None if inverse doesn't exist
pub fn mod_inverse(a: &Number, m: &Number) -> Option<Number> {
    let (gcd, x, _) = extended_gcd(a, m);
    
    if !gcd.is_one() {
        return None;
    }
    
    // Make sure result is positive
    let x_mod = &x % m;
    let result = if x_mod.is_negative() {
        &x_mod + m
    } else {
        x_mod
    };
    Some(result)
}

/// Check if a number is probably prime using Miller-Rabin test
pub fn is_probable_prime(n: &Number, k: u32) -> bool {
    if n <= &Number::from(1u32) {
        return false;
    }
    
    if n == &Number::from(2u32) || n == &Number::from(3u32) {
        return true;
    }
    
    if n % &Number::from(2u32) == Number::from(0u32) {
        return false;
    }
    
    // Write n-1 as 2^r * d
    let n_minus_1 = n - &Number::from(1u32);
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    
    while &d % &Number::from(2u32) == Number::from(0u32) {
        d = &d / &Number::from(2u32);
        r += 1;
    }
    
    // Witness loop
    for _ in 0..k {
        // Random witness between 2 and n-2
        // For deterministic testing, use fixed witnesses
        let witnesses = [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        
        for &w in witnesses.iter().take(k as usize) {
            let a = Number::from(w);
            if &a >= n {
                continue;
            }
            
            let mut x = mod_pow(&a, &d, n);
            
            if x == Number::from(1u32) || x == n_minus_1 {
                continue;
            }
            
            let mut composite = true;
            for _ in 0..r-1 {
                x = (&x * &x) % n;
                if x == n_minus_1 {
                    composite = false;
                    break;
                }
            }
            
            if composite {
                return false;
            }
        }
    }
    
    true
}

/// Compute square root to specified precision
fn sqrt_to_precision(n: u32, bits: u32) -> Number {
    // Scale up the input
    let scale = Number::from(1u32) << (2 * bits);
    let scaled_n = &Number::from(n) * &scale;
    
    // Compute sqrt of scaled number
    let sqrt_scaled = integer_sqrt(&scaled_n);
    
    // Scale back down
    &sqrt_scaled << bits
}

/// Compute natural logarithm (placeholder - needs proper implementation)
fn ln_to_precision(n: u32, bits: u32) -> Number {
    // This is a placeholder - implement using series expansion
    // For now, return a rough approximation
    match n {
        2 => {
            // ln(2) ≈ 0.693147...
            let scale = Number::from(10u32).pow(bits / 3);
            let num = Number::from(693147u32) * &scale;
            &num / &Number::from(1000000u32)
        }
        10 => {
            // ln(10) ≈ 2.302585...
            let scale = Number::from(10u32).pow(bits / 3);
            let num = Number::from(2302585u32) * &scale;
            &num / &Number::from(1000000u32)
        }
        _ => Number::from(1u32), // Placeholder
    }
}

/// Fundamental constants as rationals for exact computation
#[derive(Clone, Debug)]
pub struct FundamentalConstantsRational {
    /// Resonance decay (α) as rational
    pub alpha: Rational,
    /// Phase coupling (β) as rational
    pub beta: Rational,
    /// Scale transition (γ) as rational
    pub gamma: Rational,
    /// Interference null (δ) as rational
    pub delta: Rational,
    /// Adelic threshold (ε) as rational
    pub epsilon: Rational,
    /// Golden ratio (φ) as rational
    pub phi: Rational,
    /// Tribonacci (τ) as rational
    pub tau: Rational,
    /// Unity as rational
    pub unity: Rational,
}

impl FundamentalConstantsRational {
    /// Create constants with specified precision
    pub fn new(precision_bits: u32) -> Self {
        let scale = Number::from(1u32) << precision_bits;
        
        Self {
            // These are the empirically discovered values
            // Stored as rationals to maintain exactness
            alpha: Rational::from_ratio(
                Number::from(11750566516490533u64) * &scale,
                Number::from(10000000000000000u64) * &scale
            ),
            beta: Rational::from_ratio(
                Number::from(19968406830149554u64) * &scale,
                Number::from(100000000000000000u64) * &scale
            ),
            gamma: Rational::from_ratio(
                Number::from(1241605776553433u64) * &scale,
                Number::from(100000000000000u64) * &scale
            ),
            delta: Rational::zero(),
            epsilon: Rational::from_ratio(
                Number::from(4329953646807706u64) * &scale,
                Number::from(1000000000000000u64) * &scale
            ),
            phi: Rational::from_ratio(
                &Number::from(1u32) + &sqrt_to_precision(5, precision_bits),
                Number::from(2u32)
            ),
            tau: Rational::from_ratio(
                Number::from(1839286755214161u64) * &scale,
                Number::from(1000000000000000u64) * &scale
            ),
            unity: Rational::one(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_integer_sqrt() {
        let n = Number::from(144u32);
        assert_eq!(integer_sqrt(&n), Number::from(12u32));
        
        let n = Number::from(145u32);
        assert_eq!(integer_sqrt(&n), Number::from(12u32));
        
        // Test large number
        let n = Number::from(1u32) << 200;
        let sqrt = integer_sqrt(&n);
        assert_eq!(sqrt, Number::from(1u32) << 100);
    }
    
    #[test]
    fn test_integer_nth_root() {
        // Cube root of 27
        let n = Number::from(27u32);
        assert_eq!(integer_nth_root(&n, 3), Number::from(3u32));
        
        // 4th root of 16
        let n = Number::from(16u32);
        assert_eq!(integer_nth_root(&n, 4), Number::from(2u32));
        
        // Large number - perfect cube
        let base = Number::from(1u32) << 100;
        let n = &base * &base * &base;  // (2^100)^3 = 2^300
        let root = integer_nth_root(&n, 3);
        assert_eq!(root, base);
    }
    
    #[test]
    fn test_gcd_lcm() {
        let a = Number::from(48u32);
        let b = Number::from(18u32);
        
        assert_eq!(gcd(&a, &b), Number::from(6u32));
        assert_eq!(lcm(&a, &b), Number::from(144u32));
        
        // Test with large numbers
        let a = Number::from(1u32) << 100;
        let b = Number::from(1u32) << 50;
        assert_eq!(gcd(&a, &b), b.clone());
    }
    
    #[test]
    fn test_mod_pow() {
        // 3^5 mod 7 = 243 mod 7 = 5
        let base = Number::from(3u32);
        let exp = Number::from(5u32);
        let modulus = Number::from(7u32);
        assert_eq!(mod_pow(&base, &exp, &modulus), Number::from(5u32));
        
        // Large exponent
        let base = Number::from(2u32);
        let exp = Number::from(1000u32);
        let modulus = Number::from(13u32);
        let result = mod_pow(&base, &exp, &modulus);
        assert_eq!(result, Number::from(3u32)); // 2^1000 mod 13 = 3
    }
    
    #[test]
    fn test_mod_inverse() {
        // 3^(-1) mod 7 = 5 (because 3*5 = 15 = 1 mod 7)
        let a = Number::from(3u32);
        let m = Number::from(7u32);
        assert_eq!(mod_inverse(&a, &m), Some(Number::from(5u32)));
        
        // No inverse exists
        let a = Number::from(4u32);
        let m = Number::from(8u32);
        assert_eq!(mod_inverse(&a, &m), None);
    }
    
    #[test]
    fn test_is_probable_prime() {
        // Small primes
        assert!(is_probable_prime(&Number::from(2u32), 10));
        assert!(is_probable_prime(&Number::from(3u32), 10));
        assert!(is_probable_prime(&Number::from(5u32), 10));
        assert!(is_probable_prime(&Number::from(7u32), 10));
        assert!(is_probable_prime(&Number::from(11u32), 10));
        
        // Composites
        assert!(!is_probable_prime(&Number::from(4u32), 10));
        assert!(!is_probable_prime(&Number::from(9u32), 10));
        assert!(!is_probable_prime(&Number::from(15u32), 10));
        
        // Larger prime
        let prime = Number::from(4294967291u64); // 2^32 - 5
        assert!(is_probable_prime(&prime, 10));
    }
    
    #[test]
    fn test_constants_precision() {
        let phi_64 = get_constant(ConstantType::Phi, 64);
        let phi_128 = get_constant(ConstantType::Phi, 128);
        
        // Higher precision should have more bits
        assert!(phi_128.bit_length() > phi_64.bit_length());
    }
}