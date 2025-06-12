//! Optimized arithmetic operations
//!
//! This module provides highly optimized implementations of
//! arithmetic operations critical to The Pattern's performance.

use crate::types::Number;
use crate::Result;

/// Standard modular reduction
#[inline]
pub fn standard_mod(n: &Number, m: u64) -> u64 {
    (n % &Number::from(m)).as_integer().to_u64().unwrap_or(0)
}

/// SIMD-optimized modular reduction
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn simd_mod(n: &Number, m: u64) -> u64 {
    // For now, fall back to standard implementation
    // TODO: Implement actual SIMD operations
    standard_mod(n, m)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn simd_mod(n: &Number, m: u64) -> u64 {
    standard_mod(n, m)
}

/// Binary GCD algorithm (Stein's algorithm)
/// More efficient than Euclidean algorithm for large numbers
pub fn binary_gcd(a: &Number, b: &Number) -> Number {
    if a == &Number::from(0u32) {
        return b.clone();
    }
    if b == &Number::from(0u32) {
        return a.clone();
    }

    let mut u = a.clone();
    let mut v = b.clone();

    // Find common factor of 2
    let mut shift = 0;
    while u.is_even() && v.is_even() {
        u = &u >> 1u32;
        v = &v >> 1u32;
        shift += 1;
    }

    // Remove remaining factors of 2 from u
    while u.is_even() {
        u = &u >> 1u32;
    }

    // Main loop
    loop {
        // Remove factors of 2 from v
        while v.is_even() {
            v = &v >> 1u32;
        }

        // Ensure u <= v
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }

        v = &v - &u;

        if v == Number::from(0u32) {
            break;
        }
    }

    // Restore common factors of 2
    u << (shift as u32)
}

/// Newton's method for integer square root
/// Converges quadratically, making it very fast for large numbers
pub fn newton_isqrt(n: &Number) -> Result<Number> {
    if n < &Number::from(0u32) {
        return Err(crate::error::PatternError::InvalidInput(
            "Cannot compute square root of negative number".to_string(),
        ));
    }

    if n == &Number::from(0u32) {
        return Ok(Number::from(0u32));
    }

    // Initial guess: 2^(ceil(log2(n)/2))
    let bit_length = n.bit_length();
    let mut x = Number::from(1u32) << (((bit_length + 1) / 2) as u32);

    // Newton iteration: x_{n+1} = (x_n + n/x_n) / 2
    loop {
        let x_new = (&x + n / &x) >> 1u32;

        if x_new >= x {
            break;
        }

        x = x_new;
    }

    Ok(x)
}

/// Montgomery modular multiplication
/// Efficient for repeated modular multiplications with the same modulus
#[derive(Debug)]
pub struct MontgomeryContext {
    modulus: Number,
    r: Number,
    m_prime: Number,
}

impl MontgomeryContext {
    /// Create new Montgomery context for given modulus
    pub fn new(modulus: Number) -> Result<Self> {
        // R = 2^k where k is the bit length of modulus
        let k = modulus.bit_length();
        let r = Number::from(1u32) << (k as u32);

        // Compute R^(-1) mod m and m' where RR^(-1) - mm' = 1
        let (_r_inv, m_prime) = extended_gcd(&r, &modulus)?;

        Ok(MontgomeryContext {
            modulus,
            r,
            m_prime,
        })
    }

    /// Convert to Montgomery form
    pub fn to_montgomery(&self, a: &Number) -> Number {
        (a * &self.r) % &self.modulus
    }

    /// Convert from Montgomery form
    pub fn from_montgomery(&self, a: &Number) -> Number {
        self.montgomery_reduce(&(a * &Number::from(1u32)))
    }

    /// Montgomery reduction
    fn montgomery_reduce(&self, t: &Number) -> Number {
        let u = (t * &self.m_prime) % &self.r;
        let result = (t + &u * &self.modulus) / &self.r;

        if result >= self.modulus {
            &result - &self.modulus
        } else {
            result
        }
    }

    /// Montgomery multiplication
    pub fn montgomery_mul(&self, a: &Number, b: &Number) -> Number {
        self.montgomery_reduce(&(a * b))
    }
}

/// Extended GCD algorithm
fn extended_gcd(a: &Number, b: &Number) -> Result<(Number, Number)> {
    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = Number::from(1u32);
    let mut s = Number::from(0u32);

    while r != Number::from(0u32) {
        let quotient = &old_r / &r;

        let new_r = &old_r - &(&quotient * &r);
        old_r = r;
        r = new_r;

        let new_s = &old_s - &(&quotient * &s);
        old_s = s;
        s = new_s;
    }

    // old_r is the GCD
    if old_r != Number::from(1u32) {
        return Err(crate::error::PatternError::InvalidInput(
            "Numbers are not coprime".to_string(),
        ));
    }

    Ok((old_s, s))
}

/// Batch modular reduction
/// Compute n mod p_i for multiple primes p_i efficiently
pub fn batch_mod(n: &Number, primes: &[u64]) -> Vec<u64> {
    // TODO: Implement remainder tree algorithm for better efficiency
    primes.iter().map(|&p| standard_mod(n, p)).collect()
}

/// Precomputed reciprocals for fast division
pub struct ReciprocalTable {
    reciprocals: Vec<(u64, u64, u32)>, // (divisor, reciprocal, shift)
}

impl ReciprocalTable {
    /// Create table for divisors up to max_divisor
    pub fn new(max_divisor: u64) -> Self {
        let mut reciprocals = Vec::with_capacity(max_divisor as usize);

        for d in 1..=max_divisor {
            let shift = 64 - d.leading_zeros();
            let reciprocal = ((1u128 << (shift + 64)) / d as u128) as u64;
            reciprocals.push((d, reciprocal, shift));
        }

        ReciprocalTable { reciprocals }
    }

    /// Fast division using precomputed reciprocal
    pub fn fast_div(&self, n: u64, divisor_index: usize) -> Option<u64> {
        self.reciprocals.get(divisor_index).map(|&(_, reciprocal, shift)| {
            ((n as u128 * reciprocal as u128) >> (64 + shift)) as u64
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_gcd() {
        let a = Number::from(48u32);
        let b = Number::from(18u32);
        let gcd = binary_gcd(&a, &b);
        assert_eq!(gcd, Number::from(6u32));
    }

    #[test]
    fn test_newton_isqrt() {
        let n = Number::from(144u32);
        let sqrt = newton_isqrt(&n).unwrap();
        assert_eq!(sqrt, Number::from(12u32));

        let n = Number::from(145u32);
        let sqrt = newton_isqrt(&n).unwrap();
        assert_eq!(sqrt, Number::from(12u32)); // Floor of sqrt(145)
    }
}
