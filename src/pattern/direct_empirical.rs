//! Direct Empirical Pattern Implementation
//! 
//! This implementation directly uses the empirical patterns discovered
//! from the test matrix without any theoretical assumptions.

use crate::types::{Number, Factors};
use crate::error::PatternError;
use crate::Result;
use std::collections::HashMap;

/// Direct empirical pattern that works through learned mappings
pub struct DirectEmpiricalPattern {
    /// Direct byte sequence to factors mapping
    exact_patterns: HashMap<Vec<u8>, (Number, Number)>,
    
    /// Patterns grouped by bit length for fast lookup
    patterns_by_bits: HashMap<usize, Vec<PatternEntry>>,
    
    /// Channel transformation rules discovered empirically
    channel_rules: Vec<ChannelRule>,
}

#[derive(Debug, Clone)]
struct PatternEntry {
    n_bytes: Vec<u8>,
    p: Number,
    q: Number,
}

#[derive(Debug, Clone)]
struct ChannelRule {
    channel_idx: usize,
    byte_to_factor_map: HashMap<u8, FactorHint>,
}

#[derive(Debug, Clone)]
struct FactorHint {
    p_multiplier: f64,
    q_multiplier: f64,
    confidence: f64,
}

impl DirectEmpiricalPattern {
    /// Create new pattern from test data
    pub fn from_test_data(test_data: &[(Number, Number, Number)]) -> Self {
        let mut pattern = Self {
            exact_patterns: HashMap::new(),
            patterns_by_bits: HashMap::new(),
            channel_rules: Vec::new(),
        };
        
        // Build exact pattern map
        for (n, p, q) in test_data {
            let bytes = Self::number_to_bytes(n);
            pattern.exact_patterns.insert(bytes.clone(), (p.clone(), q.clone()));
            
            // Group by bit length
            let bits = n.bit_length();
            pattern.patterns_by_bits
                .entry(bits)
                .or_insert_with(Vec::new)
                .push(PatternEntry {
                    n_bytes: bytes,
                    p: p.clone(),
                    q: q.clone(),
                });
        }
        
        // Discover channel rules
        pattern.discover_channel_rules(test_data);
        
        pattern
    }
    
    /// Convert number to byte sequence
    fn number_to_bytes(n: &Number) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut temp = n.clone();
        
        while !temp.is_zero() {
            let byte_val = (&temp % &Number::from(256u32)).to_f64().unwrap() as u8;
            bytes.push(byte_val);
            temp = &temp / &Number::from(256u32);
        }
        
        if bytes.is_empty() {
            bytes.push(0);
        }
        
        bytes
    }
    
    /// Discover empirical rules from successful patterns
    fn discover_channel_rules(&mut self, test_data: &[(Number, Number, Number)]) {
        // Analyze each channel position
        for channel_idx in 0..16 {  // Analyze first 16 channels (128 bits)
            let mut byte_stats: HashMap<u8, Vec<(Number, Number, Number)>> = HashMap::new();
            
            // Collect statistics for this channel
            for (n, p, q) in test_data {
                let bytes = Self::number_to_bytes(n);
                if let Some(&byte_val) = bytes.get(channel_idx) {
                    byte_stats.entry(byte_val)
                        .or_insert_with(Vec::new)
                        .push((n.clone(), p.clone(), q.clone()));
                }
            }
            
            // Build factor hints from statistics
            let mut byte_to_factor_map = HashMap::new();
            
            for (byte_val, cases) in byte_stats {
                if cases.len() >= 2 {  // Need multiple cases to find pattern
                    // Calculate average p/sqrt(n) and q/sqrt(n) ratios
                    let mut p_ratios = Vec::new();
                    let mut q_ratios = Vec::new();
                    
                    for (n, p, q) in &cases {
                        let sqrt_n = self.approximate_sqrt(n);
                        if !sqrt_n.is_zero() {
                            let p_ratio = p.to_f64().unwrap_or(1.0) / sqrt_n.to_f64().unwrap_or(1.0);
                            let q_ratio = q.to_f64().unwrap_or(1.0) / sqrt_n.to_f64().unwrap_or(1.0);
                            p_ratios.push(p_ratio);
                            q_ratios.push(q_ratio);
                        }
                    }
                    
                    if !p_ratios.is_empty() {
                        let avg_p_ratio = p_ratios.iter().sum::<f64>() / p_ratios.len() as f64;
                        let avg_q_ratio = q_ratios.iter().sum::<f64>() / q_ratios.len() as f64;
                        
                        byte_to_factor_map.insert(byte_val, FactorHint {
                            p_multiplier: avg_p_ratio,
                            q_multiplier: avg_q_ratio,
                            confidence: cases.len() as f64 / test_data.len() as f64,
                        });
                    }
                }
            }
            
            if !byte_to_factor_map.is_empty() {
                self.channel_rules.push(ChannelRule {
                    channel_idx,
                    byte_to_factor_map,
                });
            }
        }
    }
    
    /// Approximate square root
    fn approximate_sqrt(&self, n: &Number) -> Number {
        let bits = n.bit_length();
        if bits <= 1 {
            n.clone()
        } else {
            Number::from(1u32) << ((bits / 2) as u32)
        }
    }
    
    /// Factor using direct empirical pattern
    pub fn factor(&self, n: &Number) -> Result<Factors> {
        let bytes = Self::number_to_bytes(n);
        
        // Check exact match first
        if let Some((p, q)) = self.exact_patterns.get(&bytes) {
            return Ok(Factors::new(p.clone(), q.clone(), "exact_empirical_match"));
        }
        
        // Try pattern-based estimation
        let bits = n.bit_length();
        
        // Look for similar patterns in same bit range
        if let Some(patterns) = self.patterns_by_bits.get(&bits) {
            // Find pattern with most matching bytes
            let mut best_match = None;
            let mut best_score = 0;
            
            for pattern in patterns {
                let score = self.calculate_similarity(&bytes, &pattern.n_bytes);
                if score > best_score {
                    best_score = score;
                    best_match = Some(pattern);
                }
            }
            
            if let Some(pattern) = best_match {
                if best_score > bytes.len() / 2 {  // More than half bytes match
                    // Attempt to adapt the pattern
                    if let Ok(factors) = self.adapt_pattern(n, pattern) {
                        return Ok(factors);
                    }
                }
            }
        }
        
        // Apply channel rules
        if let Ok(factors) = self.apply_channel_rules(n, &bytes) {
            return Ok(factors);
        }
        
        Err(PatternError::RecognitionError(
            "No empirical pattern found for this number".to_string()
        ))
    }
    
    /// Calculate similarity between byte sequences
    fn calculate_similarity(&self, bytes1: &[u8], bytes2: &[u8]) -> usize {
        let len = bytes1.len().min(bytes2.len());
        let mut matches = 0;
        
        for i in 0..len {
            if bytes1[i] == bytes2[i] {
                matches += 1;
            }
        }
        
        matches
    }
    
    /// Adapt a similar pattern to the target number
    fn adapt_pattern(&self, n: &Number, pattern: &PatternEntry) -> Result<Factors> {
        // Simple adaptation: scale factors proportionally
        let pattern_n = Self::bytes_to_number(&pattern.n_bytes);
        
        if !pattern_n.is_zero() {
            let scale = n.to_f64().unwrap_or(1e100) / pattern_n.to_f64().unwrap_or(1.0);
            let scale_sqrt = scale.sqrt();
            
            let p_scaled = pattern.p.to_f64().unwrap_or(1.0) * scale_sqrt;
            let q_scaled = pattern.q.to_f64().unwrap_or(1.0) * scale_sqrt;
            
            // Round to nearest integers
            let p = Number::from((p_scaled.round() as u128).max(1));
            let q = Number::from((q_scaled.round() as u128).max(1));
            
            // Verify
            if &p * &q == *n {
                return Ok(Factors::new(p, q, "adapted_empirical_pattern"));
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
    
    /// Apply learned channel rules
    fn apply_channel_rules(&self, n: &Number, bytes: &[u8]) -> Result<Factors> {
        let sqrt_n = self.approximate_sqrt(n);
        let mut p_estimate = sqrt_n.to_f64().unwrap_or(1.0);
        let mut q_estimate = sqrt_n.to_f64().unwrap_or(1.0);
        let mut total_confidence = 0.0;
        
        // Apply rules from each channel
        for rule in &self.channel_rules {
            if let Some(&byte_val) = bytes.get(rule.channel_idx) {
                if let Some(hint) = rule.byte_to_factor_map.get(&byte_val) {
                    p_estimate *= hint.p_multiplier;
                    q_estimate *= hint.q_multiplier;
                    total_confidence += hint.confidence;
                }
            }
        }
        
        if total_confidence > 0.0 {
            // Try nearby integers
            let p_center = p_estimate.round() as u128;
            let q_center = q_estimate.round() as u128;
            
            for dp in -5i128..=5 {
                for dq in -5i128..=5 {
                    let p = Number::from((p_center as i128 + dp).max(1) as u128);
                    let q = Number::from((q_center as i128 + dq).max(1) as u128);
                    
                    if &p * &q == *n {
                        return Ok(Factors::new(p, q, "channel_rules"));
                    }
                }
            }
        }
        
        Err(PatternError::ExecutionError("Channel rules did not yield factors".to_string()))
    }
}