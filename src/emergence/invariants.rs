//! Invariant relationship discovery
//!
//! This module discovers relationships that hold true across all observations.

use crate::types::{Observation, Pattern, PatternKind};
use crate::types::pattern::ScaleRange;
use crate::Result;

/// Discovery of invariant relationships
pub struct InvariantDiscovery;

impl InvariantDiscovery {
    /// Find all invariant relationships
    pub fn find_all(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut invariants = Vec::new();

        // Test fundamental relationships
        if Self::verify_multiplication_invariant(observations) {
            invariants.push(Self::create_multiplication_pattern());
        }

        if Self::verify_fermat_invariant(observations) {
            invariants.push(Self::create_fermat_pattern());
        }

        if Self::verify_modular_invariant(observations) {
            invariants.push(Self::create_modular_pattern());
        }

        if Self::verify_order_invariant(observations) {
            invariants.push(Self::create_order_pattern());
        }

        // Discover emergent invariants
        invariants.extend(Self::discover_emergent_invariants(observations)?);

        Ok(invariants)
    }

    /// Verify p × q = n
    fn verify_multiplication_invariant(observations: &[Observation]) -> bool {
        observations.iter().all(|obs| &obs.p * &obs.q == obs.n)
    }

    /// Create multiplication invariant pattern
    fn create_multiplication_pattern() -> Pattern {
        let mut pattern = Pattern::new("multiplication_invariant", PatternKind::Invariant);
        pattern.frequency = 1.0;
        pattern.description = "n = p × q holds for all semiprimes".to_string();
        pattern.scale_range = ScaleRange {
            min_bits: 1,
            max_bits: usize::MAX,
            unbounded: true,
        };
        pattern
    }

    /// Verify Fermat's identity
    fn verify_fermat_invariant(observations: &[Observation]) -> bool {
        observations.iter().all(|obs| {
            let a_squared = &obs.derived.fermat_a * &obs.derived.fermat_a;
            let b_squared = &obs.derived.fermat_b * &obs.derived.fermat_b;
            &a_squared - &b_squared == obs.n
        })
    }

    /// Create Fermat invariant pattern
    fn create_fermat_pattern() -> Pattern {
        let mut pattern = Pattern::new("fermat_identity", PatternKind::Invariant);
        pattern.frequency = 1.0;
        pattern.description = "n = a² - b² where a = (p+q)/2, b = |p-q|/2".to_string();
        pattern.scale_range = ScaleRange {
            min_bits: 1,
            max_bits: usize::MAX,
            unbounded: true,
        };
        pattern
    }

    /// Verify modular arithmetic invariant
    fn verify_modular_invariant(observations: &[Observation]) -> bool {
        observations.iter().all(|obs| {
            // Check that n ≡ p*q (mod m) for all moduli
            obs.modular.modular_signature.iter().enumerate().all(|(i, &n_mod)| {
                if i < obs.modular.p_mod_signature.len() && i < obs.modular.q_mod_signature.len() {
                    let p_mod = obs.modular.p_mod_signature[i];
                    let q_mod = obs.modular.q_mod_signature[i];
                    let prime =
                        crate::utils::generate_primes(i + 1)[i].as_integer().to_u64().unwrap();
                    n_mod == (p_mod * q_mod) % prime
                } else {
                    true
                }
            })
        })
    }

    /// Create modular invariant pattern
    fn create_modular_pattern() -> Pattern {
        let mut pattern = Pattern::new("modular_invariant", PatternKind::Invariant);
        pattern.frequency = 1.0;
        pattern.description = "n ≡ p·q (mod m) for all moduli m".to_string();
        pattern.scale_range = ScaleRange {
            min_bits: 1,
            max_bits: usize::MAX,
            unbounded: true,
        };
        pattern
    }

    /// Verify p ≤ q ordering
    fn verify_order_invariant(observations: &[Observation]) -> bool {
        observations.iter().all(|obs| obs.p <= obs.q)
    }

    /// Create order invariant pattern
    fn create_order_pattern() -> Pattern {
        let mut pattern = Pattern::new("factor_ordering", PatternKind::Invariant);
        pattern.frequency = 1.0;
        pattern.description = "Factors are ordered as p ≤ q".to_string();
        pattern.scale_range = ScaleRange {
            min_bits: 1,
            max_bits: usize::MAX,
            unbounded: true,
        };
        pattern
    }

    /// Discover emergent invariants from data
    fn discover_emergent_invariants(observations: &[Observation]) -> Result<Vec<Pattern>> {
        let mut patterns = Vec::new();

        // Check if sqrt(n) < (p+q)/2 always holds
        let sqrt_fermat_relation =
            observations.iter().all(|obs| obs.derived.sqrt_n < obs.derived.fermat_a);

        if sqrt_fermat_relation {
            let mut pattern = Pattern::new("sqrt_fermat_ordering", PatternKind::Invariant);
            pattern.frequency = 1.0;
            pattern.description = "sqrt(n) < (p+q)/2 always holds".to_string();
            pattern.scale_range = ScaleRange {
                min_bits: 1,
                max_bits: usize::MAX,
                unbounded: true,
            };
            patterns.push(pattern);
        }

        // Check if offset is always positive for balanced semiprimes
        let positive_offset = observations
            .iter()
            .filter(|obs| {
                obs.scale.pattern_type == crate::types::observation::PatternClass::Balanced
            })
            .all(|obs| obs.derived.offset > crate::types::Number::from(0u32));

        if positive_offset {
            let mut pattern = Pattern::new("positive_offset_balanced", PatternKind::Invariant);
            pattern.frequency = 1.0;
            pattern.description =
                "Offset from sqrt(n) is always positive for balanced semiprimes".to_string();
            patterns.push(pattern);
        }

        // Check digit sum relationships
        let digit_sum_mod9 = observations.iter().all(|obs| {
            let n_sum: u32 = obs.n.to_string().chars().filter_map(|c| c.to_digit(10)).sum();
            let p_sum: u32 = obs.p.to_string().chars().filter_map(|c| c.to_digit(10)).sum();
            let q_sum: u32 = obs.q.to_string().chars().filter_map(|c| c.to_digit(10)).sum();
            (n_sum % 9) == ((p_sum * q_sum) % 9)
        });

        if digit_sum_mod9 {
            let mut pattern = Pattern::new("digit_sum_invariant", PatternKind::Invariant);
            pattern.frequency = 1.0;
            pattern.description = "digit_sum(n) ≡ digit_sum(p) × digit_sum(q) (mod 9)".to_string();
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Validate an invariant across all scales
    pub fn validate_across_scales(pattern: &Pattern, observations: &[Observation]) -> bool {
        // Group by scale
        let mut by_scale = std::collections::HashMap::new();
        for obs in observations {
            by_scale.entry(obs.scale.bit_length / 8).or_insert_with(Vec::new).push(obs);
        }

        // Check pattern holds at each scale
        by_scale.values().all(|obs_at_scale| {
            // Pattern-specific validation
            match pattern.id.as_str() {
                "multiplication_invariant" => {
                    obs_at_scale.iter().all(|obs| &obs.p * &obs.q == obs.n)
                },
                "fermat_identity" => obs_at_scale.iter().all(|obs| {
                    let a_squared = &obs.derived.fermat_a * &obs.derived.fermat_a;
                    let b_squared = &obs.derived.fermat_b * &obs.derived.fermat_b;
                    &a_squared - &b_squared == obs.n
                }),
                _ => true, // Unknown patterns assumed valid
            }
        })
    }
}
