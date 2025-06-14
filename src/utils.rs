//! Utility functions that emerge from pattern observation

use crate::types::Number;
use crate::Result;
use rug::rand::RandState;

/// Calculate integer square root using Newton's method
/// This emerges from observing that sqrt appears in many patterns
pub fn integer_sqrt(n: &Number) -> Result<Number> {
    if n.is_negative() {
        return Err(crate::error::PatternError::ArithmeticError(
            "Cannot take square root of negative number".to_string(),
        ));
    }

    if n.is_zero() || n.is_one() {
        return Ok(n.clone());
    }

    // Newton's method: x_{n+1} = (x_n + n/x_n) / 2
    let mut x = n.clone();
    let two = Number::from(2u32);

    loop {
        let x_next = (&x + n / &x) / &two;
        if x_next >= x {
            return Ok(x);
        }
        x = x_next;
    }
}

/// Check if a number is probably prime using Miller-Rabin
/// This emerges from pattern observation that primes have distinct signatures
pub fn is_probable_prime(n: &Number, k: u32) -> bool {
    if n <= &Number::from(1u32) {
        return false;
    }
    if n == &Number::from(2u32) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Miller-Rabin test
    n.is_probably_prime(k)
}

/// Generate the first n prime numbers
/// Emerges from observation that small primes appear in many patterns
pub fn generate_primes(count: usize) -> Vec<Number> {
    let mut primes = Vec::with_capacity(count);
    if count == 0 {
        return primes;
    }

    primes.push(Number::from(2u32));
    let mut candidate = Number::from(3u32);

    while primes.len() < count {
        let mut is_prime = true;
        for p in &primes {
            if &candidate % p == Number::from(0u32) {
                is_prime = false;
                break;
            }
            if p * p > candidate {
                break;
            }
        }
        if is_prime {
            primes.push(candidate.clone());
        }
        candidate += 2u32;
    }

    primes
}

/// Calculate the nth Fibonacci number
/// Emerges from observation of Fibonacci patterns in factorizations
pub fn fibonacci(n: usize) -> Number {
    if n == 0 {
        return Number::from(0u32);
    }
    if n == 1 {
        return Number::from(1u32);
    }

    let mut a = Number::from(0u32);
    let mut b = Number::from(1u32);

    for _ in 2..=n {
        let c = &a + &b;
        a = b;
        b = c;
    }

    b
}

/// Calculate GCD of two numbers
/// Emerges from pattern observation of common factors
pub fn gcd(a: &Number, b: &Number) -> Number {
    let mut a = a.clone();
    let mut b = b.clone();

    while !b.is_zero() {
        let temp = b.clone();
        b = a % &b;
        a = temp;
    }

    a
}

/// Format a large number for display
/// Emerges from need to visualize patterns in large numbers
pub fn format_number(n: &Number) -> String {
    let s = n.to_string();
    if s.len() <= 20 {
        s
    } else {
        format!("{}...{} ({} digits)", &s[..10], &s[s.len() - 10..], s.len())
    }
}

/// Generate a random prime of specified bit length
/// Emerges from pattern testing needs
pub fn generate_random_prime(bits: u32) -> Result<Number> {
    let mut rng = RandState::new();
    let mut candidate = Number::random_bits(bits, &mut rng);

    // Ensure it's odd
    candidate.set_bit(0, true);

    // Find next probable prime
    while !is_probable_prime(&candidate, 25) {
        candidate += 2u32;
    }

    Ok(candidate)
}

/// Perform trial division to find small factors
/// Emerges from observation that small factors have distinct patterns
pub fn trial_division(n: &Number, limit: Option<&Number>) -> Result<Vec<Number>> {
    let mut factors = Vec::new();
    let mut n = n.clone();

    // Check for factor of 2
    while n.is_even() {
        factors.push(Number::from(2u32));
        n /= 2u32;
    }

    // Check odd factors up to sqrt(n) or limit
    let sqrt_n = integer_sqrt(&n)?;
    let limit = limit.cloned().unwrap_or(sqrt_n);
    let mut factor = Number::from(3u32);

    while factor <= limit && &factor * &factor <= n {
        while &n % &factor == Number::from(0u32) {
            factors.push(factor.clone());
            n = n / &factor;
        }
        factor += 2u32;
    }

    // If n is still greater than 1, it's a prime factor
    if n > Number::from(1u32) {
        factors.push(n);
    }

    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_sqrt() {
        let n = Number::from(144u32);
        let sqrt = integer_sqrt(&n).unwrap();
        assert_eq!(sqrt, Number::from(12u32));

        let n = Number::from(143u32);
        let sqrt = integer_sqrt(&n).unwrap();
        assert_eq!(sqrt, Number::from(11u32)); // Floor of sqrt(143)
    }

    #[test]
    fn test_generate_primes() {
        let primes = generate_primes(10);
        let expected: Vec<u32> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let expected: Vec<Number> = expected.into_iter().map(Number::from).collect();
        assert_eq!(primes, expected);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), Number::from(0u32));
        assert_eq!(fibonacci(1), Number::from(1u32));
        assert_eq!(fibonacci(10), Number::from(55u32));
    }

    #[test]
    fn test_is_probable_prime() {
        assert!(is_probable_prime(&Number::from(2u32), 10));
        assert!(is_probable_prime(&Number::from(17u32), 10));
        assert!(!is_probable_prime(&Number::from(15u32), 10));
        assert!(!is_probable_prime(&Number::from(1u32), 10));
    }

    #[test]
    fn test_trial_division() {
        let n = Number::from(60u32);
        let factors = trial_division(&n, None).unwrap();
        assert_eq!(
            factors,
            vec![
                Number::from(2u32),
                Number::from(2u32),
                Number::from(3u32),
                Number::from(5u32)
            ]
        );

        let n = Number::from(17u32);
        let factors = trial_division(&n, None).unwrap();
        assert_eq!(factors, vec![Number::from(17u32)]);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(
            gcd(&Number::from(48u32), &Number::from(18u32)),
            Number::from(6u32)
        );
        assert_eq!(
            gcd(&Number::from(17u32), &Number::from(19u32)),
            Number::from(1u32)
        );
    }
}
