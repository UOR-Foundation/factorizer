//! Channel decomposition and operations for The Pattern
//! 
//! Handles the decomposition of numbers into 8-bit channels and
//! provides operations for working with channel data.

use num_bigint::BigInt;

/// Decompose a BigInt into 8-bit channels
/// 
/// The number is decomposed into a vector of bytes representing channels.
/// Uses little-endian ordering so channels[0] = N mod 256.
pub fn decompose(n: &BigInt) -> Vec<u8> {
    let bytes = n.to_bytes_le().1; // Little-endian: LSB first
    
    if bytes.is_empty() {
        return vec![0u8];
    }
    
    bytes
}

/// Reconstruct a BigInt from a slice of channels
/// 
/// This is the inverse of decompose - takes channel bytes and
/// reconstructs the original number using little-endian ordering.
pub fn reconstruct(channels: &[u8]) -> BigInt {
    BigInt::from_bytes_le(num_bigint::Sign::Plus, channels)
}

/// Extract a specific range of channels and reconstruct as BigInt
/// 
/// Used during factor extraction to get the value represented by
/// a subset of aligned channels.
pub fn extract_channel_range(channels: &[u8], start: usize, end: usize) -> Option<BigInt> {
    if start > end || end >= channels.len() {
        return None;
    }
    
    let slice = &channels[start..=end];
    Some(BigInt::from_bytes_be(num_bigint::Sign::Plus, slice))
}

/// Calculate the number of channels needed for a given bit size
pub fn channels_for_bits(bit_size: usize) -> usize {
    (bit_size + 7) / 8
}

/// Get the bit size of a number
pub fn bit_size(n: &BigInt) -> usize {
    n.bits() as usize
}

/// Channel iterator for sliding window operations
pub struct ChannelWindow<'a> {
    channels: &'a [u8],
    window_size: usize,
    position: usize,
}

impl<'a> ChannelWindow<'a> {
    /// Create a new channel window iterator
    pub fn new(channels: &'a [u8], window_size: usize) -> Self {
        Self {
            channels,
            window_size,
            position: 0,
        }
    }
}

impl<'a> Iterator for ChannelWindow<'a> {
    type Item = (usize, &'a [u8]);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.position + self.window_size > self.channels.len() {
            return None;
        }
        
        let window = &self.channels[self.position..self.position + self.window_size];
        let pos = self.position;
        self.position += 1;
        
        Some((pos, window))
    }
}

/// Compute channel-to-channel coupling strength based on Hamming distance
pub fn coupling_strength(channel1: u8, channel2: u8) -> u8 {
    // XOR gives bits that differ
    let diff = channel1 ^ channel2;
    // Count set bits (Hamming distance)
    let distance = diff.count_ones() as u8;
    // Coupling is inverse of distance (8 - distance)
    8 - distance
}

/// Check if a sequence of channels forms an arithmetic progression
/// in their harmonic signatures
pub fn is_harmonic_progression(channels: &[u8], expected_step: u8) -> bool {
    if channels.len() < 2 {
        return true;
    }
    
    for i in 1..channels.len() {
        let diff = channels[i].wrapping_sub(channels[i-1]);
        if diff != expected_step {
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decompose_small() {
        let n = BigInt::from(143u32); // Binary: 10001111
        let channels = decompose(&n);
        
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0], 143);
    }
    
    #[test]
    fn test_decompose_with_padding() {
        let n = BigInt::from(0x1234u32); // 2 bytes
        let channels = decompose(&n);
        
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0], 0x12);
        assert_eq!(channels[1], 0x34);
    }
    
    #[test]
    fn test_decompose_large() {
        let n = BigInt::from(0x123456789ABCDEFu64);
        let channels = decompose(&n);
        
        assert_eq!(channels.len(), 8);
        assert_eq!(channels[0], 0x01);
        assert_eq!(channels[7], 0xEF);
    }
    
    #[test]
    fn test_reconstruct() {
        let original = BigInt::from(0x123456789ABCDEFu64);
        let channels = decompose(&original);
        let reconstructed = reconstruct(&channels);
        
        assert_eq!(original, reconstructed);
    }
    
    #[test]
    fn test_extract_channel_range() {
        let channels = vec![0x12, 0x34, 0x56, 0x78];
        
        let extracted = extract_channel_range(&channels, 1, 2).unwrap();
        assert_eq!(extracted, BigInt::from(0x3456u32));
    }
    
    #[test]
    fn test_channel_window() {
        let channels = vec![1, 2, 3, 4, 5];
        let windows: Vec<_> = ChannelWindow::new(&channels, 3).collect();
        
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], (0, &[1, 2, 3][..]));
        assert_eq!(windows[1], (1, &[2, 3, 4][..]));
        assert_eq!(windows[2], (2, &[3, 4, 5][..]));
    }
    
    #[test]
    fn test_coupling_strength() {
        assert_eq!(coupling_strength(0b11111111, 0b11111111), 8); // Identical
        assert_eq!(coupling_strength(0b11111111, 0b00000000), 0); // Opposite
        assert_eq!(coupling_strength(0b10101010, 0b10101011), 7); // 1 bit diff
    }
    
    #[test]
    fn test_harmonic_progression() {
        assert!(is_harmonic_progression(&[10, 15, 20, 25], 5));
        assert!(!is_harmonic_progression(&[10, 15, 21, 25], 5));
        assert!(is_harmonic_progression(&[0xFF, 0x00, 0x01], 1)); // With wrapping
    }
}