//! Rational arithmetic for exact scaling operations
//!
//! This module provides exact rational arithmetic to avoid precision loss
//! when scaling patterns and computing with large numbers.

use crate::types::Number;
use std::cmp::Ordering;
use std::fmt;
use serde::{Serialize, Deserialize};

/// Exact rational number representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rational {
    /// Numerator (can be any size)
    pub num: Number,
    /// Denominator (always positive, never zero)
    pub den: Number,
}

impl Rational {
    /// Create a new rational number
    pub fn new(num: Number, den: Number) -> Self {
        assert!(!den.is_zero(), "Denominator cannot be zero");
        
        let mut r = Rational { num, den };
        r.reduce();
        r
    }
    
    /// Create rational from integer
    pub fn from_integer(n: Number) -> Self {
        Rational {
            num: n,
            den: Number::from(1u32),
        }
    }
    
    /// Create rational from two integers
    pub fn from_ratio(num: impl Into<Number>, den: impl Into<Number>) -> Self {
        Self::new(num.into(), den.into())
    }
    
    /// Get one
    pub fn one() -> Self {
        Self::from_integer(Number::from(1u32))
    }
    
    /// Get zero  
    pub fn zero() -> Self {
        Self::from_integer(Number::from(0u32))
    }
    
    /// Reduce to lowest terms
    fn reduce(&mut self) {
        let gcd = self.gcd(&self.num.abs(), &self.den.abs());
        if !gcd.is_one() {
            self.num = &self.num / &gcd;
            self.den = &self.den / &gcd;
        }
        
        // Ensure denominator is positive
        if self.den.is_negative() {
            self.num = &Number::from(0u32) - &self.num;
            self.den = &Number::from(0u32) - &self.den;
        }
    }
    
    /// Greatest common divisor using Euclidean algorithm
    fn gcd(&self, a: &Number, b: &Number) -> Number {
        if b.is_zero() {
            a.clone()
        } else {
            self.gcd(b, &(a % b))
        }
    }
    
    /// Convert to Number by integer division (truncating)
    pub fn to_integer(&self) -> Number {
        &self.num / &self.den
    }
    
    /// Convert to Number by rounding to nearest integer
    pub fn round(&self) -> Number {
        let quotient = &self.num / &self.den;
        let remainder = &self.num % &self.den;
        let doubled_remainder = &remainder * &Number::from(2u32);
        
        if doubled_remainder >= self.den {
            &quotient + &Number::from(1u32)
        } else {
            quotient
        }
    }
    
    /// Check if this is an integer
    pub fn is_integer(&self) -> bool {
        &self.num % &self.den == Number::from(0u32)
    }
    
    /// Get the reciprocal (1/x)
    pub fn reciprocal(&self) -> Self {
        assert!(!self.num.is_zero(), "Cannot take reciprocal of zero");
        
        if self.num.is_negative() {
            Rational::new(
                &Number::from(0u32) - &self.den,
                &Number::from(0u32) - &self.num
            )
        } else {
            Rational::new(self.den.clone(), self.num.clone())
        }
    }
    
    /// Multiply by an integer
    pub fn mul_integer(&self, n: &Number) -> Self {
        Rational::new(&self.num * n, self.den.clone())
    }
    
    /// Divide by an integer
    pub fn div_integer(&self, n: &Number) -> Self {
        assert!(!n.is_zero(), "Cannot divide by zero");
        Rational::new(self.num.clone(), &self.den * n)
    }
    
    /// Power to integer exponent
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            Self::one()
        } else {
            Rational::new(
                self.num.pow(exp),
                self.den.pow(exp)
            )
        }
    }
    
    /// Get numerator
    pub fn numerator(&self) -> &Number {
        &self.num
    }
    
    /// Get denominator
    pub fn denominator(&self) -> &Number {
        &self.den
    }
    
    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    
    /// Check if one
    pub fn is_one(&self) -> bool {
        self.num == self.den
    }
    
    /// Check if negative
    pub fn is_negative(&self) -> bool {
        self.num.is_negative()
    }
}

/// Addition
impl std::ops::Add for &Rational {
    type Output = Rational;
    
    fn add(self, other: &Rational) -> Rational {
        // a/b + c/d = (ad + bc) / bd
        let num = &(&self.num * &other.den) + &(&other.num * &self.den);
        let den = &self.den * &other.den;
        Rational::new(num, den)
    }
}

/// Subtraction
impl std::ops::Sub for &Rational {
    type Output = Rational;
    
    fn sub(self, other: &Rational) -> Rational {
        // a/b - c/d = (ad - bc) / bd
        let num = &(&self.num * &other.den) - &(&other.num * &self.den);
        let den = &self.den * &other.den;
        Rational::new(num, den)
    }
}

/// Multiplication
impl std::ops::Mul for &Rational {
    type Output = Rational;
    
    fn mul(self, other: &Rational) -> Rational {
        // a/b * c/d = ac / bd
        let num = &self.num * &other.num;
        let den = &self.den * &other.den;
        Rational::new(num, den)
    }
}

/// Division
impl std::ops::Div for &Rational {
    type Output = Rational;
    
    fn div(self, other: &Rational) -> Rational {
        // a/b / c/d = a/b * d/c = ad / bc
        assert!(!other.num.is_zero(), "Cannot divide by zero");
        self * &other.reciprocal()
    }
}

/// Comparison
impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        // a/b = c/d iff ad = bc
        &(&self.num * &other.den) == &(&other.num * &self.den)
    }
}

impl Eq for Rational {}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b < c/d iff ad < bc (assuming positive denominators)
        (&self.num * &other.den).cmp(&(&other.num * &self.den))
    }
}

/// Remainder/modulo operation
impl std::ops::Rem for &Rational {
    type Output = Rational;
    
    fn rem(self, other: &Rational) -> Rational {
        // a/b % c/d = (a/b - floor(a/b / c/d) * c/d)
        let quotient = self / other;
        let floor = quotient.to_integer();
        let floor_rat = Rational::from_integer(floor);
        self - &(&floor_rat * other)
    }
}

/// Display
impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.is_one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Square root approximation using Newton's method
impl Rational {
    /// Approximate square root to given iterations
    pub fn sqrt_approx(&self, iterations: usize) -> Self {
        assert!(!self.num.is_negative(), "Cannot take square root of negative number");
        
        // Initial guess: sqrt(a/b) ≈ sqrt(a)/sqrt(b)
        let mut x = Rational::from_integer(self.num.sqrt());
        let sqrt_den = Rational::from_integer(self.den.sqrt());
        x = &x / &sqrt_den;
        
        // Newton's method: x_{n+1} = (x_n + a/x_n) / 2
        for _ in 0..iterations {
            let x_inv = x.reciprocal();
            let next = &(&x + &(self * &x_inv)) / &Rational::from_ratio(2u32, 1u32);
            x = next;
        }
        
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::from_ratio(3u32, 4u32);
        let b = Rational::from_ratio(2u32, 3u32);
        
        // 3/4 + 2/3 = 9/12 + 8/12 = 17/12
        let sum = &a + &b;
        assert_eq!(sum, Rational::from_ratio(17u32, 12u32));
        
        // 3/4 * 2/3 = 6/12 = 1/2
        let product = &a * &b;
        assert_eq!(product, Rational::from_ratio(1u32, 2u32));
        
        // 3/4 / 2/3 = 3/4 * 3/2 = 9/8
        let quotient = &a / &b;
        assert_eq!(quotient, Rational::from_ratio(9u32, 8u32));
    }
    
    #[test]
    fn test_large_rationals() {
        // Test with 200-bit numbers
        let large = Number::from(1u32) << 200;
        let a = Rational::from_ratio(large.clone(), 3u32);
        let b = Rational::from_ratio(2u32, large.clone());
        
        let product = &a * &b;
        assert_eq!(product, Rational::from_ratio(2u32, 3u32));
    }
}