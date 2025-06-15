//! Phase propagation model for multi-channel coordination
//!
//! Implements phase relationships between channels to track how resonance
//! phases propagate through the number's byte representation.

use crate::{ResonanceTuple, TunerParams};
use num_bigint::BigInt;
use num_traits::{Zero, One, ToPrimitive};
use std::f64::consts::PI;

/// Phase propagation state for a sequence of channels
#[derive(Debug, Clone)]
pub struct PhaseState {
    /// Current accumulated phase
    pub accumulated_phase: BigInt,
    /// Phase velocity (rate of change)
    pub phase_velocity: BigInt,
    /// Phase acceleration (for non-linear propagation)
    pub phase_acceleration: BigInt,
    /// Channel position where this state applies
    pub channel_position: usize,
}

impl PhaseState {
    /// Create initial phase state
    pub fn new(initial_phase: BigInt) -> Self {
        Self {
            accumulated_phase: initial_phase,
            phase_velocity: BigInt::zero(),
            phase_acceleration: BigInt::zero(),
            channel_position: 0,
        }
    }
    
    /// Create phase state from resonance
    pub fn from_resonance(resonance: &ResonanceTuple, channel_pos: usize) -> Self {
        Self {
            accumulated_phase: resonance.phase_offset.clone(),
            phase_velocity: BigInt::from(channel_pos * 8), // Base velocity from position
            phase_acceleration: BigInt::zero(),
            channel_position: channel_pos,
        }
    }
    
    /// Propagate phase to next channel
    pub fn propagate(&self) -> Self {
        let new_phase = &self.accumulated_phase + &self.phase_velocity + &self.phase_acceleration;
        let new_velocity = &self.phase_velocity + &self.phase_acceleration;
        
        Self {
            accumulated_phase: new_phase,
            phase_velocity: new_velocity,
            phase_acceleration: self.phase_acceleration.clone(),
            channel_position: self.channel_position + 1,
        }
    }
    
    /// Apply damping to prevent unbounded growth
    pub fn apply_damping(&mut self, damping_factor: f64) {
        if damping_factor < 1.0 && damping_factor > 0.0 {
            // Convert to f64, apply damping, convert back
            if let Some(vel_f64) = self.phase_velocity.to_f64() {
                self.phase_velocity = BigInt::from((vel_f64 * damping_factor) as i64);
            }
            if let Some(acc_f64) = self.phase_acceleration.to_f64() {
                self.phase_acceleration = BigInt::from((acc_f64 * damping_factor) as i64);
            }
        }
    }
}

/// Phase relationship between two channels
#[derive(Debug, Clone)]
pub struct PhaseRelation {
    /// Phase difference between channels
    pub phase_diff: BigInt,
    /// Phase coherence (0.0 = random, 1.0 = perfectly coherent)
    pub coherence: f64,
    /// Phase lock indicator
    pub is_locked: bool,
}

/// Calculate phase propagation through a sequence of channels
pub fn propagate_phase_sequence(
    resonances: &[(usize, u8, ResonanceTuple)],
    n: &BigInt,
    _params: &TunerParams,
) -> Vec<PhaseState> {
    let mut phase_states = Vec::new();
    
    if resonances.is_empty() {
        return phase_states;
    }
    
    // Initialize with first channel
    let mut current_state = PhaseState::from_resonance(&resonances[0].2, resonances[0].0);
    phase_states.push(current_state.clone());
    
    // Propagate through remaining channels
    for i in 1..resonances.len() {
        let (pos, _, res) = &resonances[i];
        
        // Calculate expected phase from propagation
        let propagated_state = current_state.propagate();
        
        // Calculate actual phase from resonance
        let actual_phase = &res.phase_offset;
        
        // Update velocity based on phase difference
        let phase_error = actual_phase - &propagated_state.accumulated_phase;
        let velocity_correction = &phase_error / BigInt::from(8); // Smooth correction
        
        // Create new state with corrections
        current_state = PhaseState {
            accumulated_phase: actual_phase.clone(),
            phase_velocity: &propagated_state.phase_velocity + &velocity_correction,
            phase_acceleration: velocity_correction.clone() / BigInt::from(16),
            channel_position: *pos,
        };
        
        // Apply damping to prevent instability
        current_state.apply_damping(0.95);
        
        phase_states.push(current_state.clone());
    }
    
    // Post-process: check for phase alignment with n
    for state in &mut phase_states {
        // Reduce phase modulo n for alignment checking
        state.accumulated_phase = &state.accumulated_phase % n;
    }
    
    phase_states
}

/// Detect phase relationships between adjacent channels
pub fn detect_phase_relations(
    phase_states: &[PhaseState],
    n: &BigInt,
    params: &TunerParams,
) -> Vec<PhaseRelation> {
    let mut relations = Vec::new();
    
    for i in 0..phase_states.len().saturating_sub(1) {
        let state1 = &phase_states[i];
        let state2 = &phase_states[i + 1];
        
        // Calculate phase difference
        let phase_diff = &state2.accumulated_phase - &state1.accumulated_phase;
        
        // Calculate coherence based on velocity consistency
        let vel_diff = &state2.phase_velocity - &state1.phase_velocity;
        let coherence = calculate_coherence(&vel_diff, n);
        
        // Check for phase lock (small phase difference relative to n)
        let threshold = BigInt::from(params.alignment_threshold as u64);
        let phase_diff_mod = &phase_diff % n;
        let is_locked = phase_diff_mod <= threshold || 
                       (n - &phase_diff_mod) <= threshold;
        
        relations.push(PhaseRelation {
            phase_diff,
            coherence,
            is_locked,
        });
    }
    
    relations
}

/// Calculate coherence value from velocity difference
fn calculate_coherence(vel_diff: &BigInt, n: &BigInt) -> f64 {
    // Coherence is high when velocity difference is small relative to n
    if let (Some(diff_f64), Some(n_f64)) = (vel_diff.to_f64(), n.to_f64()) {
        let normalized_diff = (diff_f64 / n_f64).abs();
        // Use exponential decay for coherence
        (-normalized_diff * 10.0).exp()
    } else {
        0.0
    }
}

/// Phase alignment pattern across multiple channels
#[derive(Debug, Clone)]
pub struct PhaseAlignment {
    /// Starting channel index
    pub start_channel: usize,
    /// Ending channel index
    pub end_channel: usize,
    /// Common phase period
    pub phase_period: BigInt,
    /// Alignment strength (0.0 to 1.0)
    pub alignment_strength: f64,
}

/// Detect phase alignment patterns
pub fn detect_phase_alignments(
    _phase_states: &[PhaseState],
    phase_relations: &[PhaseRelation],
    n: &BigInt,
    _params: &TunerParams,
) -> Vec<PhaseAlignment> {
    let mut alignments = Vec::new();
    
    // Look for sequences of phase-locked channels
    let mut i = 0;
    while i < phase_relations.len() {
        if phase_relations[i].is_locked {
            let start = i;
            let mut end = i;
            
            // Extend while phase remains locked
            while end < phase_relations.len() && phase_relations[end].is_locked {
                end += 1;
            }
            
            if end > start {
                // Calculate common phase period
                let phase_diffs: Vec<_> = phase_relations[start..end]
                    .iter()
                    .map(|r| &r.phase_diff)
                    .collect();
                
                let phase_period = if !phase_diffs.is_empty() {
                    // Find GCD of phase differences
                    phase_diffs.iter()
                        .fold(phase_diffs[0].clone(), |acc, diff| {
                            use num_integer::Integer;
                            acc.gcd(diff)
                        })
                } else {
                    BigInt::one()
                };
                
                // Calculate alignment strength
                let coherence_sum: f64 = phase_relations[start..end]
                    .iter()
                    .map(|r| r.coherence)
                    .sum();
                let alignment_strength = coherence_sum / (end - start) as f64;
                
                // Check if phase period relates to factors
                if phase_period > BigInt::one() && &phase_period < n {
                    alignments.push(PhaseAlignment {
                        start_channel: start,
                        end_channel: end,
                        phase_period: phase_period.clone(),
                        alignment_strength,
                    });
                }
            }
            
            i = end + 1;
        } else {
            i += 1;
        }
    }
    
    alignments
}

/// Extract potential factors from phase alignments
pub fn extract_factors_from_phase(
    alignments: &[PhaseAlignment],
    n: &BigInt,
) -> Vec<BigInt> {
    let mut potential_factors = Vec::new();
    
    for alignment in alignments {
        // Strong alignments might indicate factors
        if alignment.alignment_strength > 0.7 {
            // Check if phase period is a factor
            if n % &alignment.phase_period == BigInt::zero() {
                potential_factors.push(alignment.phase_period.clone());
            }
            
            // Check GCD with n
            use num_integer::Integer;
            let gcd = alignment.phase_period.gcd(n);
            if gcd > BigInt::one() && &gcd < n {
                potential_factors.push(gcd);
            }
        }
    }
    
    // Remove duplicates
    potential_factors.sort();
    potential_factors.dedup();
    
    potential_factors
}

/// Phase wave representation for visualization
#[derive(Debug, Clone)]
pub struct PhaseWave {
    /// Amplitude at each channel
    pub amplitudes: Vec<f64>,
    /// Phase at each channel
    pub phases: Vec<f64>,
    /// Frequency components
    pub frequencies: Vec<f64>,
}

/// Convert phase states to wave representation
pub fn phase_states_to_wave(phase_states: &[PhaseState], n: &BigInt) -> PhaseWave {
    let mut amplitudes = Vec::new();
    let mut phases = Vec::new();
    let mut frequencies = Vec::new();
    
    for (i, state) in phase_states.iter().enumerate() {
        // Convert phase to radians
        let phase_rad = if let (Some(phase_f64), Some(n_f64)) = 
            (state.accumulated_phase.to_f64(), n.to_f64()) {
            2.0 * PI * phase_f64 / n_f64
        } else {
            0.0
        };
        
        // Amplitude based on phase velocity
        let amplitude = if let Some(vel_f64) = state.phase_velocity.to_f64() {
            (1.0_f64 + vel_f64.abs()).ln()
        } else {
            1.0
        };
        
        // Frequency from velocity change
        let frequency = if i > 0 {
            if let (Some(v1), Some(v2)) = (
                phase_states[i-1].phase_velocity.to_f64(),
                state.phase_velocity.to_f64()
            ) {
                (v2 - v1).abs() / (2.0 * PI)
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        amplitudes.push(amplitude);
        phases.push(phase_rad);
        frequencies.push(frequency);
    }
    
    PhaseWave {
        amplitudes,
        phases,
        frequencies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phase_state_propagation() {
        let initial = PhaseState::new(BigInt::from(100));
        let mut state = initial.clone();
        state.phase_velocity = BigInt::from(10);
        
        let next = state.propagate();
        assert_eq!(next.accumulated_phase, BigInt::from(110));
        assert_eq!(next.channel_position, 1);
    }
    
    #[test]
    fn test_phase_damping() {
        let mut state = PhaseState::new(BigInt::zero());
        state.phase_velocity = BigInt::from(100);
        state.phase_acceleration = BigInt::from(20);
        
        state.apply_damping(0.5);
        
        assert_eq!(state.phase_velocity, BigInt::from(50));
        assert_eq!(state.phase_acceleration, BigInt::from(10));
    }
    
    #[test]
    fn test_coherence_calculation() {
        let n = BigInt::from(1000);
        
        // Small difference = high coherence
        let small_diff = BigInt::from(5);
        let coherence1 = calculate_coherence(&small_diff, &n);
        assert!(coherence1 > 0.5);
        
        // Large difference = low coherence
        let large_diff = BigInt::from(500);
        let coherence2 = calculate_coherence(&large_diff, &n);
        assert!(coherence2 < 0.1);
    }
}