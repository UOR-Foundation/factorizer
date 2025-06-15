//! Special case detection for twin primes, perfect squares, and other patterns

use num_bigint::BigInt;
use num_traits::{Zero, One};
use crate::Factors;

/// Special case types that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialCase {
    /// Perfect square (n = p²)
    PerfectSquare,
    /// Twin primes (p and q differ by 2)
    TwinPrimes,
    /// Sophie Germain primes (q = 2p + 1)
    SophieGermain,
    /// Safe primes (p = (q-1)/2)
    SafePrime,
    /// Cousin primes (differ by 4)
    CousinPrimes,
    /// Sexy primes (differ by 6)
    SexyPrimes,
}

/// Result of special case detection
#[derive(Debug, Clone)]
pub struct SpecialCaseResult {
    /// Type of special case detected
    pub case_type: SpecialCase,
    /// The factors
    pub factors: Factors,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Detect special cases in a semiprime
pub fn detect_special_cases(n: &BigInt) -> Vec<SpecialCaseResult> {
    let mut results = Vec::new();
    
    // Check for perfect square
    if let Some(result) = check_perfect_square(n) {
        results.push(result);
    }
    
    // Check for twin primes
    if let Some(result) = check_twin_primes(n) {
        results.push(result);
    }
    
    // Check for Sophie Germain primes
    if let Some(result) = check_sophie_germain(n) {
        results.push(result);
    }
    
    // Check for cousin primes
    if let Some(result) = check_cousin_primes(n) {
        results.push(result);
    }
    
    // Check for sexy primes
    if let Some(result) = check_sexy_primes(n) {
        results.push(result);
    }
    
    results
}

/// Check if n is a perfect square
fn check_perfect_square(n: &BigInt) -> Option<SpecialCaseResult> {
    let sqrt = n.sqrt();
    
    if &sqrt * &sqrt == *n {
        // It's a perfect square
        return Some(SpecialCaseResult {
            case_type: SpecialCase::PerfectSquare,
            factors: Factors::new(sqrt.clone(), sqrt),
            confidence: 1.0,
        });
    }
    
    // Check if it's close to a perfect square (useful for noisy data)
    let lower = &sqrt * &sqrt;
    let upper = (&sqrt + 1) * (&sqrt + 1);
    
    if n > &lower && n < &upper {
        let diff_lower = n - lower;
        let diff_upper = upper - n;
        
        if diff_lower < diff_upper && diff_lower < sqrt {
            // Closer to lower square
            let confidence = 1.0 - (diff_lower.bits() as f64 / n.bits() as f64);
            if confidence > 0.9 {
                return Some(SpecialCaseResult {
                    case_type: SpecialCase::PerfectSquare,
                    factors: Factors::new(sqrt.clone(), sqrt),
                    confidence,
                });
            }
        }
    }
    
    None
}

/// Check for twin primes (differ by 2)
fn check_twin_primes(n: &BigInt) -> Option<SpecialCaseResult> {
    // For twin primes p and p+2, we have n = p(p+2) = p² + 2p
    // This means n = p² + 2p, so p² + 2p - n = 0
    // Using quadratic formula: p = (-2 ± sqrt(4 + 4n)) / 2 = -1 ± sqrt(1 + n)
    
    let discriminant = BigInt::one() + n;
    let sqrt_disc = discriminant.sqrt();
    
    // Check if discriminant is a perfect square
    if &sqrt_disc * &sqrt_disc == discriminant {
        let p = &sqrt_disc - 1;
        if p > BigInt::zero() {
            let q = &p + 2;
            
            // Verify
            if &p * &q == *n {
                return Some(SpecialCaseResult {
                    case_type: SpecialCase::TwinPrimes,
                    factors: Factors::new(p, q),
                    confidence: 1.0,
                });
            }
        }
    }
    
    None
}

/// Check for Sophie Germain primes (q = 2p + 1)
fn check_sophie_germain(n: &BigInt) -> Option<SpecialCaseResult> {
    // For Sophie Germain primes, n = p(2p + 1) = 2p² + p
    // So 2p² + p - n = 0
    // Using quadratic formula: p = (-1 ± sqrt(1 + 8n)) / 4
    
    let discriminant = BigInt::one() + (n * BigInt::from(8)); // 1 + 8n
    let sqrt_disc = discriminant.sqrt();
    
    if &sqrt_disc * &sqrt_disc == discriminant {
        // Check if (-1 + sqrt_disc) is divisible by 4
        if sqrt_disc > BigInt::one() {
            let numerator = &sqrt_disc - 1;
            if &numerator & BigInt::from(3) == BigInt::zero() {
                let p = numerator >> 2;
                if p > BigInt::zero() {
                    let q = (&p << 1) + 1; // 2p + 1
                    
                    // Verify
                    if &p * &q == *n {
                        return Some(SpecialCaseResult {
                            case_type: SpecialCase::SophieGermain,
                            factors: Factors::new(p, q),
                            confidence: 1.0,
                        });
                    }
                }
            }
        }
    }
    
    None
}

/// Check for cousin primes (differ by 4)
fn check_cousin_primes(n: &BigInt) -> Option<SpecialCaseResult> {
    // For cousin primes p and p+4, we have n = p(p+4) = p² + 4p
    // So p² + 4p - n = 0
    // Using quadratic formula: p = (-4 ± sqrt(16 + 4n)) / 2 = -2 ± sqrt(4 + n)
    
    let discriminant = BigInt::from(4) + n;
    let sqrt_disc = discriminant.sqrt();
    
    if &sqrt_disc * &sqrt_disc == discriminant {
        if sqrt_disc > BigInt::from(2) {
            let p = &sqrt_disc - 2;
            if p > BigInt::zero() {
                let q = &p + 4;
                
                // Verify
                if &p * &q == *n {
                    return Some(SpecialCaseResult {
                        case_type: SpecialCase::CousinPrimes,
                        factors: Factors::new(p, q),
                        confidence: 1.0,
                    });
                }
            }
        }
    }
    
    None
}

/// Check for sexy primes (differ by 6)
fn check_sexy_primes(n: &BigInt) -> Option<SpecialCaseResult> {
    // For sexy primes p and p+6, we have n = p(p+6) = p² + 6p
    // So p² + 6p - n = 0
    // Using quadratic formula: p = (-6 ± sqrt(36 + 4n)) / 2 = -3 ± sqrt(9 + n)
    
    let discriminant = BigInt::from(9) + n;
    let sqrt_disc = discriminant.sqrt();
    
    if &sqrt_disc * &sqrt_disc == discriminant {
        if sqrt_disc > BigInt::from(3) {
            let p = &sqrt_disc - 3;
            if p > BigInt::zero() {
                let q = &p + 6;
                
                // Verify
                if &p * &q == *n {
                    return Some(SpecialCaseResult {
                        case_type: SpecialCase::SexyPrimes,
                        factors: Factors::new(p, q),
                        confidence: 1.0,
                    });
                }
            }
        }
    }
    
    None
}

/// Apply special case optimizations to factorization
pub fn try_special_cases(n: &BigInt) -> Option<Factors> {
    let special_cases = detect_special_cases(n);
    
    // Return the first high-confidence result
    special_cases.into_iter()
        .filter(|result| result.confidence >= 0.99)
        .map(|result| result.factors)
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perfect_square() {
        let n = BigInt::from(9409); // 97²
        let result = check_perfect_square(&n);
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.case_type, SpecialCase::PerfectSquare);
        assert_eq!(result.factors.p, BigInt::from(97));
        assert_eq!(result.factors.q, BigInt::from(97));
        assert_eq!(result.confidence, 1.0);
    }
    
    #[test]
    fn test_twin_primes() {
        let n = BigInt::from(35); // 5 × 7
        let result = check_twin_primes(&n);
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.case_type, SpecialCase::TwinPrimes);
        assert_eq!(result.factors.p, BigInt::from(5));
        assert_eq!(result.factors.q, BigInt::from(7));
    }
    
    #[test]
    fn test_sophie_germain() {
        let n = BigInt::from(143); // 11 × 13, but 11 × 23 = 253 is Sophie Germain
        // This test shows n=143 is not Sophie Germain
        let result = check_sophie_germain(&n);
        assert!(result.is_none());
        
        // Test actual Sophie Germain: 11 × 23 = 253
        let n = BigInt::from(253);
        let result = check_sophie_germain(&n);
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.case_type, SpecialCase::SophieGermain);
        assert_eq!(result.factors.p, BigInt::from(11));
        assert_eq!(result.factors.q, BigInt::from(23));
    }
    
    #[test]
    fn test_cousin_primes() {
        let n = BigInt::from(77); // 7 × 11
        let result = check_cousin_primes(&n);
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.case_type, SpecialCase::CousinPrimes);
        assert_eq!(result.factors.p, BigInt::from(7));
        assert_eq!(result.factors.q, BigInt::from(11));
    }
    
    #[test]
    fn test_sexy_primes() {
        let n = BigInt::from(91); // 7 × 13
        let result = check_sexy_primes(&n);
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert_eq!(result.case_type, SpecialCase::SexyPrimes);
        assert_eq!(result.factors.p, BigInt::from(7));
        assert_eq!(result.factors.q, BigInt::from(13));
    }
}