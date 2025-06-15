//! Hierarchical channel grouping for multi-scale pattern recognition
//!
//! Implements hierarchical organization of channels into groups of 2, 4, 8, 16, etc.
//! to capture patterns at different scales.

use crate::{ResonanceTuple, TunerParams, CouplingMatrix};
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive};
use std::collections::HashMap;

/// Hierarchical grouping level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupingLevel {
    /// Individual channels (no grouping)
    Single,
    /// Pairs of channels (2-channel groups)
    Pair,
    /// Quads of channels (4-channel groups)
    Quad,
    /// Octets of channels (8-channel groups)
    Octet,
    /// 16-channel groups
    Hex,
    /// 32-channel groups
    ThirtyTwo,
    /// 64-channel groups
    SixtyFour,
}

impl GroupingLevel {
    /// Get the size of this grouping level
    pub fn size(&self) -> usize {
        match self {
            GroupingLevel::Single => 1,
            GroupingLevel::Pair => 2,
            GroupingLevel::Quad => 4,
            GroupingLevel::Octet => 8,
            GroupingLevel::Hex => 16,
            GroupingLevel::ThirtyTwo => 32,
            GroupingLevel::SixtyFour => 64,
        }
    }
    
    /// Get the next larger grouping level
    pub fn next_level(&self) -> Option<GroupingLevel> {
        match self {
            GroupingLevel::Single => Some(GroupingLevel::Pair),
            GroupingLevel::Pair => Some(GroupingLevel::Quad),
            GroupingLevel::Quad => Some(GroupingLevel::Octet),
            GroupingLevel::Octet => Some(GroupingLevel::Hex),
            GroupingLevel::Hex => Some(GroupingLevel::ThirtyTwo),
            GroupingLevel::ThirtyTwo => Some(GroupingLevel::SixtyFour),
            GroupingLevel::SixtyFour => None,
        }
    }
    
    /// Determine appropriate grouping levels for a given number of channels
    pub fn levels_for_channels(num_channels: usize) -> Vec<GroupingLevel> {
        let mut levels = vec![GroupingLevel::Single];
        
        if num_channels >= 2 {
            levels.push(GroupingLevel::Pair);
        }
        if num_channels >= 4 {
            levels.push(GroupingLevel::Quad);
        }
        if num_channels >= 8 {
            levels.push(GroupingLevel::Octet);
        }
        if num_channels >= 16 {
            levels.push(GroupingLevel::Hex);
        }
        if num_channels >= 32 {
            levels.push(GroupingLevel::ThirtyTwo);
        }
        if num_channels >= 64 {
            levels.push(GroupingLevel::SixtyFour);
        }
        
        levels
    }
}

/// A group of channels at a specific hierarchical level
#[derive(Debug, Clone)]
pub struct ChannelGroup {
    /// Grouping level
    pub level: GroupingLevel,
    /// Starting channel index
    pub start_idx: usize,
    /// Ending channel index (inclusive)
    pub end_idx: usize,
    /// Channel values in this group
    pub channel_values: Vec<u8>,
    /// Combined resonance for the group
    pub group_resonance: ResonanceTuple,
    /// Sub-groups (for hierarchical structure)
    pub sub_groups: Vec<ChannelGroup>,
}

impl ChannelGroup {
    /// Create a new channel group
    pub fn new(
        level: GroupingLevel,
        start_idx: usize,
        channels: &[(usize, u8, ResonanceTuple)],
    ) -> Option<Self> {
        let size = level.size();
        if channels.len() < size {
            return None;
        }
        
        let end_idx = start_idx + size - 1;
        
        // Extract channel values
        let channel_values: Vec<u8> = channels[..size]
            .iter()
            .map(|(_, val, _)| *val)
            .collect();
        
        // Combine resonances based on level
        let group_resonance = match level {
            GroupingLevel::Single => channels[0].2.clone(),
            _ => combine_group_resonances(&channels[..size], level),
        };
        
        // Create sub-groups for hierarchical structure
        let sub_groups = if level != GroupingLevel::Single {
            // Find appropriate sub-level (should be smaller than current level)
            let sub_level = match level {
                GroupingLevel::Pair => GroupingLevel::Single,
                GroupingLevel::Quad => GroupingLevel::Pair,
                GroupingLevel::Octet => GroupingLevel::Quad,
                GroupingLevel::Hex => GroupingLevel::Octet,
                GroupingLevel::ThirtyTwo => GroupingLevel::Hex,
                GroupingLevel::SixtyFour => GroupingLevel::ThirtyTwo,
                _ => GroupingLevel::Single,
            };
            
            if sub_level.size() < level.size() {
                create_sub_groups(&channels[..size], sub_level)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        
        Some(ChannelGroup {
            level,
            start_idx,
            end_idx,
            channel_values,
            group_resonance,
            sub_groups,
        })
    }
    
    /// Get the size of this group
    pub fn size(&self) -> usize {
        self.end_idx - self.start_idx + 1
    }
    
    /// Check if this group shows factor-like patterns
    pub fn has_factor_pattern(&self, n: &BigInt) -> bool {
        use num_integer::Integer;
        
        // Check if group resonance indicates a factor
        let gcd = self.group_resonance.primary_resonance.gcd(n);
        if gcd > BigInt::one() && &gcd < n && n % &gcd == BigInt::zero() {
            return true;
        }
        
        // For larger groups, check if the combined channel value is meaningful
        if self.level as usize >= GroupingLevel::Pair as usize {
            let combined = self.channel_values.iter()
                .fold(BigInt::zero(), |acc, &val| acc * 256 + BigInt::from(val));
            
            if combined > BigInt::one() && &combined <= &n.sqrt() {
                if n % &combined == BigInt::zero() {
                    return true;
                }
            }
        }
        
        false
    }
}

/// Combine resonances for a group based on the grouping level
fn combine_group_resonances(
    channels: &[(usize, u8, ResonanceTuple)],
    level: GroupingLevel,
) -> ResonanceTuple {
    if channels.is_empty() {
        return ResonanceTuple::new(BigInt::one(), 0, BigInt::zero());
    }
    
    match level {
        GroupingLevel::Single => channels[0].2.clone(),
        GroupingLevel::Pair => {
            if channels.len() < 2 {
                return channels[0].2.clone();
            }
            // Use coupling matrix for pairs
            let coupling = CouplingMatrix::standard();
            let (coupled1, coupled2) = crate::apply_channel_coupling(
                &channels[0].2,
                &channels[1].2,
                &coupling,
            );
            ResonanceTuple::new(
                &coupled1.primary_resonance * &coupled2.primary_resonance,
                coupled1.harmonic_signature ^ coupled2.harmonic_signature,
                &coupled1.phase_offset + &coupled2.phase_offset,
            )
        }
        _ => {
            // For larger groups, combine pairwise
            let mut result = channels[0].2.clone();
            
            for i in 1..channels.len().min(level.size()) {
                // Combine with next channel
                let combined = ResonanceTuple::new(
                    &result.primary_resonance * &channels[i].2.primary_resonance,
                    result.harmonic_signature ^ channels[i].2.harmonic_signature,
                    &result.phase_offset + &channels[i].2.phase_offset,
                );
                result = combined;
            }
            
            result
        }
    }
}

/// Create sub-groups at a given level
fn create_sub_groups(
    channels: &[(usize, u8, ResonanceTuple)],
    level: GroupingLevel,
) -> Vec<ChannelGroup> {
    let mut sub_groups = Vec::new();
    let size = level.size();
    
    for i in (0..channels.len()).step_by(size) {
        if i + size <= channels.len() {
            if let Some(group) = ChannelGroup::new(level, i, &channels[i..]) {
                sub_groups.push(group);
            }
        }
    }
    
    sub_groups
}

/// Hierarchical channel analysis result
#[derive(Debug, Clone)]
pub struct HierarchicalAnalysis {
    /// Root level groups
    pub root_groups: Vec<ChannelGroup>,
    /// All groups by level
    pub groups_by_level: HashMap<GroupingLevel, Vec<ChannelGroup>>,
    /// Detected patterns at each level
    pub patterns_by_level: HashMap<GroupingLevel, Vec<GroupPattern>>,
}

/// Pattern detected at a specific grouping level
#[derive(Debug, Clone)]
pub struct GroupPattern {
    /// Grouping level where pattern was found
    pub level: GroupingLevel,
    /// Groups involved in the pattern
    pub groups: Vec<usize>,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f64,
    /// Potential factor indicated by pattern
    pub factor_candidate: Option<BigInt>,
}

/// Perform hierarchical analysis of channels
pub fn analyze_channel_hierarchy(
    channels: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    params: &TunerParams,
) -> HierarchicalAnalysis {
    let mut groups_by_level = HashMap::new();
    let mut patterns_by_level = HashMap::new();
    
    // Determine appropriate grouping levels
    let levels = GroupingLevel::levels_for_channels(channels.len());
    
    // Analyze at each level
    for level in &levels {
        let groups = create_groups_at_level(channels, *level);
        let patterns = detect_patterns_at_level(&groups, n, params);
        
        groups_by_level.insert(*level, groups.clone());
        patterns_by_level.insert(*level, patterns);
    }
    
    // Get root groups (largest level that covers all channels)
    let root_level = levels.iter()
        .rev()
        .find(|&&l| l.size() <= channels.len())
        .copied()
        .unwrap_or(GroupingLevel::Single);
    
    let root_groups = groups_by_level.get(&root_level)
        .cloned()
        .unwrap_or_default();
    
    HierarchicalAnalysis {
        root_groups,
        groups_by_level,
        patterns_by_level,
    }
}

/// Create channel groups at a specific level
fn create_groups_at_level(
    channels: &[(usize, u8, ResonanceTuple)],
    level: GroupingLevel,
) -> Vec<ChannelGroup> {
    let mut groups = Vec::new();
    let size = level.size();
    
    for i in (0..channels.len()).step_by(size) {
        if i + size <= channels.len() {
            if let Some(group) = ChannelGroup::new(level, i, &channels[i..]) {
                groups.push(group);
            }
        }
    }
    
    groups
}

/// Detect patterns at a specific grouping level
fn detect_patterns_at_level(
    groups: &[ChannelGroup],
    n: &BigInt,
    params: &TunerParams,
) -> Vec<GroupPattern> {
    let mut patterns = Vec::new();
    
    // Look for factor patterns in individual groups
    for (i, group) in groups.iter().enumerate() {
        if group.has_factor_pattern(n) {
            let factor_candidate = extract_factor_from_group(group, n);
            patterns.push(GroupPattern {
                level: group.level,
                groups: vec![i],
                strength: calculate_pattern_strength(group, n),
                factor_candidate,
            });
        }
    }
    
    // Look for patterns across adjacent groups
    if groups.len() >= 2 {
        for i in 0..groups.len() - 1 {
            let pattern_strength = calculate_cross_group_pattern(
                &groups[i],
                &groups[i + 1],
                n,
                params,
            );
            
            if pattern_strength > 0.5 {
                let factor_candidate = extract_factor_from_group_pair(
                    &groups[i],
                    &groups[i + 1],
                    n,
                );
                
                patterns.push(GroupPattern {
                    level: groups[0].level,
                    groups: vec![i, i + 1],
                    strength: pattern_strength,
                    factor_candidate,
                });
            }
        }
    }
    
    patterns
}

/// Calculate pattern strength for a group
fn calculate_pattern_strength(group: &ChannelGroup, n: &BigInt) -> f64 {
    use num_integer::Integer;
    
    // Base strength on GCD relationship
    let gcd = group.group_resonance.primary_resonance.gcd(n);
    
    if gcd > BigInt::one() && &gcd < n {
        // Strength based on size of GCD relative to sqrt(n)
        let sqrt_n = n.sqrt();
        if gcd <= sqrt_n {
            0.8 + 0.2 * (gcd.to_f64().unwrap_or(1.0) / sqrt_n.to_f64().unwrap_or(1.0))
        } else {
            0.7
        }
    } else {
        // Check harmonic signature
        let pattern = (group.group_resonance.harmonic_signature & 0xFFFF) as u16;
        if pattern > 0 && BigInt::from(pattern).gcd(n) > BigInt::one() {
            0.6
        } else {
            0.0
        }
    }
}

/// Calculate pattern strength across two groups
fn calculate_cross_group_pattern(
    group1: &ChannelGroup,
    group2: &ChannelGroup,
    n: &BigInt,
    _params: &TunerParams,
) -> f64 {
    // Check phase relationship
    let phase_diff = &group2.group_resonance.phase_offset - &group1.group_resonance.phase_offset;
    let phase_mod = phase_diff % n;
    
    // Strong pattern if phase difference is small
    if phase_mod < BigInt::from(256u32) {
        0.7
    } else {
        // Check resonance alignment
        use num_integer::Integer;
        let gcd_resonances = group1.group_resonance.primary_resonance
            .gcd(&group2.group_resonance.primary_resonance);
        
        if gcd_resonances > BigInt::one() && gcd_resonances.gcd(n) > BigInt::one() {
            0.6
        } else {
            0.0
        }
    }
}

/// Extract potential factor from a channel group
fn extract_factor_from_group(group: &ChannelGroup, n: &BigInt) -> Option<BigInt> {
    use num_integer::Integer;
    
    // Try GCD of group resonance
    let gcd = group.group_resonance.primary_resonance.gcd(n);
    if gcd > BigInt::one() && &gcd < n && n % &gcd == BigInt::zero() {
        return Some(gcd);
    }
    
    // Try combined channel value
    let combined = group.channel_values.iter()
        .fold(BigInt::zero(), |acc, &val| acc * 256 + BigInt::from(val));
    
    if combined > BigInt::one() && &combined <= &n.sqrt() && n % &combined == BigInt::zero() {
        return Some(combined);
    }
    
    None
}

/// Extract potential factor from a pair of groups
fn extract_factor_from_group_pair(
    group1: &ChannelGroup,
    group2: &ChannelGroup,
    n: &BigInt,
) -> Option<BigInt> {
    // Combine all channel values from both groups
    let mut all_values = group1.channel_values.clone();
    all_values.extend(&group2.channel_values);
    
    let combined = all_values.iter()
        .fold(BigInt::zero(), |acc, &val| acc * 256 + BigInt::from(val));
    
    if combined > BigInt::one() && &combined <= &n.sqrt() && n % &combined == BigInt::zero() {
        Some(combined)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grouping_level_size() {
        assert_eq!(GroupingLevel::Single.size(), 1);
        assert_eq!(GroupingLevel::Pair.size(), 2);
        assert_eq!(GroupingLevel::Quad.size(), 4);
        assert_eq!(GroupingLevel::Octet.size(), 8);
    }
    
    #[test]
    fn test_levels_for_channels() {
        let levels = GroupingLevel::levels_for_channels(10);
        assert!(levels.contains(&GroupingLevel::Single));
        assert!(levels.contains(&GroupingLevel::Pair));
        assert!(levels.contains(&GroupingLevel::Quad));
        assert!(levels.contains(&GroupingLevel::Octet));
        assert!(!levels.contains(&GroupingLevel::Hex));
    }
    
    #[test]
    fn test_channel_group_creation() {
        let channels = vec![
            (0, 10u8, ResonanceTuple::new(BigInt::from(100), 0x1234, BigInt::from(10))),
            (1, 20u8, ResonanceTuple::new(BigInt::from(200), 0x5678, BigInt::from(20))),
        ];
        
        let group = ChannelGroup::new(GroupingLevel::Pair, 0, &channels).unwrap();
        assert_eq!(group.size(), 2);
        assert_eq!(group.channel_values, vec![10, 20]);
    }
}