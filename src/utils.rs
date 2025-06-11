//! Utility functions that emerge from pattern observation

use crate::types::Number;
use crate::Result;

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
    use rug::rand::RandState;

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
    let mut rng = RandState::new();
    n.is_probably_prime(k, &mut rng) != rug::integer::IsPrime::No
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
        format!(
            "{}...{} ({} digits)",
            &s[..10],
            &s[s.len() - 10..],
            s.len()
        )
    }
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
}