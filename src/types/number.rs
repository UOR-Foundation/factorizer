//! Arbitrary precision number type
//!
//! This type emerges from the observation that patterns exist at all scales.

use rug::Integer;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Rem, Shl, Shr, Sub};
use std::str::FromStr;

/// Arbitrary precision integer for pattern observation
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Number(Integer);

impl Number {
    /// Create a new number from any integer type
    pub fn from<T: Into<Integer>>(value: T) -> Self {
        Number(value.into())
    }

    /// Create number from string
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, rug::integer::ParseIntegerError> {
        Integer::from_str_radix(s, radix as i32).map(Number)
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Check if one
    pub fn is_one(&self) -> bool {
        self.0 == 1
    }

    /// Check if negative
    pub fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    /// Check if even
    pub fn is_even(&self) -> bool {
        self.0.is_even()
    }

    /// Check if odd
    pub fn is_odd(&self) -> bool {
        self.0.is_odd()
    }

    /// Get bit length
    pub fn bit_length(&self) -> usize {
        self.0.significant_bits() as usize
    }

    /// Get digit length in base 10
    pub fn digit_length(&self) -> usize {
        self.0.to_string().len()
    }

    /// Convert to f64 (may lose precision)
    pub fn to_f64(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }

    /// Get inner Integer reference
    pub fn as_integer(&self) -> &Integer {
        &self.0
    }

    /// Check probable primality
    pub fn is_probably_prime(&self, reps: u32) -> bool {
        self.0.is_probably_prime(reps) != rug::integer::IsPrime::No
    }

    /// Power operation
    pub fn pow(&self, exp: u32) -> Self {
        use rug::ops::Pow;
        Number((&self.0).pow(exp).into())
    }

    /// Integer square root
    pub fn sqrt(&self) -> Self {
        Number(self.0.sqrt_ref().into())
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        Number(self.0.abs_ref().into())
    }

    /// Generate random number with specified bit count
    pub fn random_bits(bits: u32, rng: &mut rug::rand::RandState<'_>) -> Self {
        let n = Integer::from(Integer::random_bits(bits, rng));
        Number(n)
    }

    /// Set a specific bit
    pub fn set_bit(&mut self, bit_index: u32, value: bool) {
        self.0.set_bit(bit_index, value);
    }

    /// Check if the number is a power of two
    pub fn is_power_of_two(&self) -> bool {
        if self.0 <= 0 {
            return false;
        }
        // A power of two has only one bit set
        // Count the number of set bits
        self.0.count_ones() == Some(1)
    }
}

// Arithmetic operations
impl Add for &Number {
    type Output = Number;
    fn add(self, other: &Number) -> Number {
        Number(Integer::from(&self.0 + &other.0))
    }
}

impl Add<u32> for &Number {
    type Output = Number;
    fn add(self, other: u32) -> Number {
        Number(Integer::from(&self.0 + other))
    }
}

impl Sub for &Number {
    type Output = Number;
    fn sub(self, other: &Number) -> Number {
        Number(Integer::from(&self.0 - &other.0))
    }
}

impl Mul for &Number {
    type Output = Number;
    fn mul(self, other: &Number) -> Number {
        Number(Integer::from(&self.0 * &other.0))
    }
}

impl Div for &Number {
    type Output = Number;
    fn div(self, other: &Number) -> Number {
        Number(Integer::from(&self.0 / &other.0))
    }
}

impl Rem for &Number {
    type Output = Number;
    fn rem(self, other: &Number) -> Number {
        Number(Integer::from(&self.0 % &other.0))
    }
}

impl AddAssign<u32> for Number {
    fn add_assign(&mut self, other: u32) {
        self.0 += other;
    }
}

impl DivAssign<u32> for Number {
    fn div_assign(&mut self, other: u32) {
        self.0 /= other;
    }
}

impl Add<Number> for &Number {
    type Output = Number;
    fn add(self, other: Number) -> Number {
        Number(Integer::from(&self.0 + &other.0))
    }
}

impl Add<u32> for Number {
    type Output = Number;
    fn add(self, other: u32) -> Number {
        Number(Integer::from(self.0 + other))
    }
}

impl Div<&Number> for Number {
    type Output = Number;
    fn div(self, other: &Number) -> Number {
        Number(Integer::from(self.0 / &other.0))
    }
}

impl Rem<&Number> for Number {
    type Output = Number;
    fn rem(self, other: &Number) -> Number {
        Number(Integer::from(self.0 % &other.0))
    }
}

impl Mul<Number> for Number {
    type Output = Number;
    fn mul(self, other: Number) -> Number {
        Number(Integer::from(self.0 * other.0))
    }
}

impl Mul<&Number> for Number {
    type Output = Number;
    fn mul(self, other: &Number) -> Number {
        Number(Integer::from(self.0 * &other.0))
    }
}

impl Mul<Number> for &Number {
    type Output = Number;
    fn mul(self, other: Number) -> Number {
        Number(Integer::from(&self.0 * other.0))
    }
}

// Shift operations
impl Shl<u32> for &Number {
    type Output = Number;
    fn shl(self, shift: u32) -> Number {
        Number(Integer::from(&self.0 << shift))
    }
}

impl Shl<u32> for Number {
    type Output = Number;
    fn shl(self, shift: u32) -> Number {
        Number(Integer::from(self.0 << shift))
    }
}

impl Shr<u32> for &Number {
    type Output = Number;
    fn shr(self, shift: u32) -> Number {
        Number(Integer::from(&self.0 >> shift))
    }
}

impl Shr<u32> for Number {
    type Output = Number;
    fn shr(self, shift: u32) -> Number {
        Number(Integer::from(self.0 >> shift))
    }
}

// Conversions
impl From<u32> for Number {
    fn from(value: u32) -> Self {
        Number(Integer::from(value))
    }
}

impl From<u64> for Number {
    fn from(value: u64) -> Self {
        Number(Integer::from(value))
    }
}

impl From<&str> for Number {
    fn from(s: &str) -> Self {
        Number(Integer::from_str(s).unwrap())
    }
}

impl FromStr for Number {
    type Err = rug::integer::ParseIntegerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Integer::from_str(s).map(Number)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Serialization
impl Serialize for Number {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Number::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_operations() {
        let a = Number::from(143u32);
        let b = Number::from(11u32);

        assert_eq!(&a / &b, Number::from(13u32));
        assert_eq!(&a % &b, Number::from(0u32));

        let c = &a + &b;
        assert_eq!(c, Number::from(154u32));
    }

    #[test]
    fn test_large_numbers() {
        let n = Number::from_str("123456789012345678901234567890").unwrap();
        assert_eq!(n.bit_length(), 97);
        assert_eq!(n.digit_length(), 30);
    }
}
