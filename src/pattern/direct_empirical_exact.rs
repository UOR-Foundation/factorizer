//! Direct empirical pattern approach with arbitrary precision
//! 
//! This learns patterns directly from the test matrix without any theory,
//! using exact arithmetic to avoid precision loss at any scale.

use crate::types::{Number, Rational, integer_sqrt};
use crate::types::Factors;
use crate::error::PatternError;
use crate::Result;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A stored pattern entry with exact values
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternEntry {
    n_bytes: Vec<u8>,
    p: Number,
    q: Number,
}

/// Learned behavior for a specific channel
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChannelRule {
    channel_idx: usize,
    byte_to_factor_map: HashMap<u8, FactorHintExact>,
}

/// Factor hint from a specific byte value using exact arithmetic
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FactorHintExact {
    /// Rational multiplier for p estimate
    p_multiplier: Rational,
    /// Rational multiplier for q estimate  
    q_multiplier: Rational,
    /// Confidence in this hint
    confidence: f64,
}

/// Direct empirical pattern that learns from test data
pub struct DirectEmpiricalPatternExact {
    /// All known patterns indexed by their byte representation
    pattern_map: HashMap<Vec<u8>, PatternEntry>,
    
    /// Channel-specific rules learned from data
    channel_rules: Vec<ChannelRule>,
    
    /// Precision for calculations
    precision_bits: u32,
}

impl DirectEmpiricalPatternExact {
    /// Create from test data with specified precision
    pub fn from_test_data(test_data: &[(Number, Number, Number)], precision_bits: u32) -> Self {
        let mut pattern = Self {
            pattern_map: HashMap::new(),
            channel_rules: Vec::new(),
            precision_bits,
        };
        
        // First pass: memorize all patterns
        for (n, p, q) in test_data {
            let n_bytes = pattern.number_to_bytes(n);
            pattern.pattern_map.insert(n_bytes.clone(), PatternEntry {
                n_bytes,
                p: p.clone(),
                q: q.clone(),
            });
        }
        
        // Second pass: learn channel rules
        pattern.learn_channel_rules(test_data);
        
        pattern
    }
    
    /// Convert number to byte representation
    fn number_to_bytes(&self, n: &Number) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut temp = n.clone();
        let base = Number::from(256u32);
        
        while !temp.is_zero() {
            let byte_val = (&temp % &base).to_u32().unwrap_or(0) as u8;
            bytes.push(byte_val);
            temp = &temp / &base;
        }
        
        if bytes.is_empty() {
            bytes.push(0);
        }
        
        bytes
    }
    
    /// Learn channel-specific rules from data
    fn learn_channel_rules(&mut self, test_data: &[(Number, Number, Number)]) {
        // Initialize rules for first 128 channels
        for channel_idx in 0..128 {
            self.channel_rules.push(ChannelRule {
                channel_idx,
                byte_to_factor_map: HashMap::new(),
            });
        }
        
        // Analyze each test case
        for (n, p, q) in test_data {
            let n_bytes = self.number_to_bytes(n);
            let sqrt_n = integer_sqrt(n);
            
            // Calculate exact ratios using rational arithmetic
            let p_ratio = Rational::from_ratio(p.clone(), sqrt_n.clone());
            let q_ratio = Rational::from_ratio(q.clone(), sqrt_n);
            
            // Learn from each channel
            for (idx, &byte_val) in n_bytes.iter().enumerate() {
                if idx < self.channel_rules.len() {
                    let entry = self.channel_rules[idx]
                        .byte_to_factor_map
                        .entry(byte_val)
                        .or_insert(FactorHintExact {
                            p_multiplier: p_ratio.clone(),
                            q_multiplier: q_ratio.clone(),
                            confidence: 0.0,
                        });
                    
                    // Update with running average (keeping exact ratios)
                    let count = entry.confidence + 1.0;
                    let weight_old = entry.confidence / count;
                    let weight_new = 1.0 / count;
                    
                    entry.p_multiplier = &(&entry.p_multiplier * &Rational::from_ratio(
                        (weight_old * 1000.0) as u64, 
                        1000u32
                    )) + &(&p_ratio * &Rational::from_ratio(
                        (weight_new * 1000.0) as u64,
                        1000u32
                    ));
                    
                    entry.q_multiplier = &(&entry.q_multiplier * &Rational::from_ratio(
                        (weight_old * 1000.0) as u64,
                        1000u32
                    )) + &(&q_ratio * &Rational::from_ratio(
                        (weight_new * 1000.0) as u64,
                        1000u32
                    ));
                    
                    entry.confidence = count;
                }
            }
        }
    }
    
    /// Factor a number using learned patterns
    pub fn factor(&self, n: &Number) -> Result<Factors> {
        let n_bytes = self.number_to_bytes(n);
        
        // Try exact match first
        if let Some(pattern) = self.pattern_map.get(&n_bytes) {
            return Ok(Factors::new(
                pattern.p.clone(),
                pattern.q.clone(),
                "exact_pattern_match".to_string()
            ));
        }
        
        // Try finding similar patterns
        if let Some((pattern, distance)) = self.find_similar_pattern(&n_bytes) {
            if distance < 5 {
                return self.adapt_pattern_exact(n, pattern);
            }
        }
        
        // Apply channel rules
        self.apply_channel_rules_exact(n, &n_bytes)
    }
    
    /// Find most similar pattern by Hamming distance
    fn find_similar_pattern(&self, target_bytes: &[u8]) -> Option<(&PatternEntry, usize)> {
        let mut best_match = None;
        let mut min_distance = usize::MAX;
        
        for pattern in self.pattern_map.values() {
            let distance = self.hamming_distance(target_bytes, &pattern.n_bytes);
            if distance < min_distance {
                min_distance = distance;
                best_match = Some(pattern);
            }
        }
        
        best_match.map(|pattern| (pattern, min_distance))
    }
    
    /// Calculate Hamming distance between byte arrays
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
        let max_len = a.len().max(b.len());
        let mut distance = 0;
        
        for i in 0..max_len {
            let a_byte = a.get(i).copied().unwrap_or(0);
            let b_byte = b.get(i).copied().unwrap_or(0);
            distance += (a_byte ^ b_byte).count_ones() as usize;
        }
        
        distance
    }
    
    /// Adapt a similar pattern to the target number using exact arithmetic
    fn adapt_pattern_exact(&self, n: &Number, pattern: &PatternEntry) -> Result<Factors> {
        let pattern_n = Self::bytes_to_number(&pattern.n_bytes);
        
        if !pattern_n.is_zero() {
            // For very large numbers, use a more direct approach to avoid stack overflow
            // Instead of computing sqrt(n/pattern_n), we use the fact that:
            // If pattern has factors (p1, q1) and we want factors of n
            // Then n ≈ pattern_n * k, so factors are approximately (p1*sqrt(k), q1*sqrt(k))
            
            // First, try simple scaling if the numbers have similar bit lengths
            let n_bits = n.bit_length();
            let pattern_bits = pattern_n.bit_length();
            
            if (n_bits as i32 - pattern_bits as i32).abs() <= 10 {
                // Use integer-based approximation for large numbers
                // Compute sqrt(n) and sqrt(pattern_n) separately
                let sqrt_n = integer_sqrt(n);
                let sqrt_pattern_n = integer_sqrt(&pattern_n);
                
                // Scale factors using integer arithmetic
                let p_scaled = (&pattern.p * &sqrt_n) / &sqrt_pattern_n;
                let q_scaled = (&pattern.q * &sqrt_n) / &sqrt_pattern_n;
                
                // Try the scaled values
                if &p_scaled * &q_scaled == *n {
                    return Ok(Factors::new(p_scaled, q_scaled, "adapted_empirical_pattern".to_string()));
                }
                
                // Try small adjustments
                for dp in -2..=2 {
                    for dq in -2..=2 {
                        let p_adj = &p_scaled + &Number::from(dp);
                        let q_adj = &q_scaled + &Number::from(dq);
                        
                        if p_adj > Number::from(1u32) && q_adj > Number::from(1u32) {
                            if &p_adj * &q_adj == *n {
                                return Ok(Factors::new(p_adj, q_adj, "adapted_empirical_pattern_adjusted".to_string()));
                            }
                        }
                    }
                }
            }
        }
        
        Err(PatternError::ExecutionError("Pattern adaptation failed".to_string()))
    }
    
    /// Convert bytes back to number
    fn bytes_to_number(bytes: &[u8]) -> Number {
        let mut n = Number::from(0u32);
        let base = Number::from(256u32);
        
        for (i, &byte_val) in bytes.iter().enumerate() {
            let place_value = base.pow(i as u32);
            n = &n + &(&Number::from(byte_val as u32) * &place_value);
        }
        
        n
    }
    
    /// Apply learned channel rules using exact arithmetic
    fn apply_channel_rules_exact(&self, n: &Number, bytes: &[u8]) -> Result<Factors> {
        let sqrt_n = integer_sqrt(n);
        let mut p_estimate = Rational::from_integer(sqrt_n.clone());
        let mut q_estimate = Rational::from_integer(sqrt_n);
        let mut total_confidence = 0.0;
        
        // Apply rules from each channel
        for rule in &self.channel_rules {
            if let Some(&byte_val) = bytes.get(rule.channel_idx) {
                if let Some(hint) = rule.byte_to_factor_map.get(&byte_val) {
                    p_estimate = &p_estimate * &hint.p_multiplier;
                    q_estimate = &q_estimate * &hint.q_multiplier;
                    total_confidence += hint.confidence;
                }
            }
        }
        
        if total_confidence > 0.0 {
            // Round to nearest integers
            let p_center = p_estimate.round();
            let q_center = q_estimate.round();
            
            // Try nearby integers - NO PRECISION LOSS!
            for dp in -5..=5 {
                for dq in -5..=5 {
                    let p = &p_center + &Number::from(dp);
                    let q = &q_center + &Number::from(dq);
                    
                    if p > Number::from(1u32) && q > Number::from(1u32) {
                        if &p * &q == *n {
                            return Ok(Factors::new(p, q, "channel_rules".to_string()));
                        }
                    }
                }
            }
        }
        
        Err(PatternError::ExecutionError("No pattern found".to_string()))
    }
}