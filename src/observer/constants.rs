//! Universal constant discovery
//!
//! This module discovers constants that appear naturally in patterns.

use crate::types::{Observation, Pattern, UniversalConstant};
use std::collections::HashMap;

/// Constant discovery from patterns
#[derive(Debug)]
pub struct ConstantDiscovery;

impl ConstantDiscovery {
    /// Extract constants from patterns
    pub fn extract(patterns: &[Pattern]) -> Vec<UniversalConstant> {
        let mut constants = Vec::new();
        let mut ratio_counts: HashMap<String, (f64, usize)> = HashMap::new();

        // Look for recurring values in pattern parameters
        for pattern in patterns {
            for (i, &value) in pattern.parameters.iter().enumerate() {
                if value.abs() > 0.001 && value.is_finite() {
                    // Round to identify similar values
                    let rounded = (value * 1000000.0).round() / 1000000.0;
                    let key = format!("{:.6}_param{}", rounded, i);

                    let entry = ratio_counts.entry(key).or_insert((rounded, 0));
                    entry.1 += 1;
                }
            }
        }

        // Extract frequently appearing values
        let total_patterns = patterns.len() as f64;
        for (key, (value, count)) in ratio_counts {
            let frequency = count as f64 / total_patterns;

            if frequency > 0.1 {
                // Appears in >10% of patterns
                let constant = UniversalConstant {
                    name: format!("discovered_{}", key.replace('.', "_")),
                    value,
                    appearances: patterns
                        .iter()
                        .filter(|p| p.parameters.iter().any(|&v| (v - value).abs() < 0.000001))
                        .map(|p| p.id.clone())
                        .collect(),
                    universality: frequency,
                    meaning: None,
                };

                constants.push(constant);
            }
        }

        // Add known mathematical constants if they appear
        constants.extend(Self::check_known_constants(patterns));

        constants
    }

    /// Check for known mathematical constants
    fn check_known_constants(patterns: &[Pattern]) -> Vec<UniversalConstant> {
        let mut constants = Vec::new();

        let known = vec![
            ("phi", 1.618033988749895, "Golden ratio"),
            ("pi", std::f64::consts::PI, "Circle constant"),
            ("e", std::f64::consts::E, "Euler's number"),
            ("sqrt2", std::f64::consts::SQRT_2, "Square root of 2"),
            ("ln2", std::f64::consts::LN_2, "Natural log of 2"),
        ];

        for (name, value, meaning) in known {
            let appearances: Vec<String> = patterns
                .iter()
                .filter(|p| {
                    p.parameters.iter().any(|&v| {
                        let ratio = v / value;
                        (ratio - 1.0).abs() < 0.01 || // Direct appearance
                        (ratio - 2.0).abs() < 0.01 || // Double
                        (ratio - 0.5).abs() < 0.01 // Half
                    })
                })
                .map(|p| p.id.clone())
                .collect();

            if !appearances.is_empty() {
                let constant = UniversalConstant {
                    name: name.to_string(),
                    value,
                    appearances: appearances.clone(),
                    universality: appearances.len() as f64 / patterns.len() as f64,
                    meaning: Some(meaning.to_string()),
                };

                constants.push(constant);
            }
        }

        constants
    }

    /// Extract ratios from observations
    pub fn extract_ratios(observations: &[Observation]) -> Vec<(String, f64)> {
        let mut ratios = Vec::new();

        for obs in observations {
            // Offset to sqrt(n) ratio
            if obs.derived.sqrt_n.as_integer().to_u64().unwrap_or(0) > 0 {
                let ratio = obs.derived.offset.to_f64().unwrap_or(0.0)
                    / obs.derived.sqrt_n.to_f64().unwrap_or(1.0);
                ratios.push(("offset_sqrt_ratio".to_string(), ratio));
            }

            // Balance ratio
            ratios.push(("balance_ratio".to_string(), obs.scale.balance_ratio));

            // Fermat ratios
            if obs.derived.fermat_a.as_integer().to_u64().unwrap_or(0) > 0 {
                let a_n_ratio =
                    obs.derived.fermat_a.to_f64().unwrap_or(0.0) / obs.n.to_f64().unwrap_or(1.0);
                ratios.push(("fermat_a_n_ratio".to_string(), a_n_ratio));
            }
        }

        ratios
    }

    /// Find recurring values in patterns
    pub fn find_recurring_values(patterns: &[Pattern]) -> Vec<UniversalConstant> {
        Self::extract(patterns)
    }

    /// Validate constant universality
    pub fn validate_constant_universality(
        constant: f64,
        observations: &[Observation],
        threshold: f64,
    ) -> bool {
        // Check how often the constant appears in relationships
        let mut appearances = 0;
        let epsilon = constant * 0.01; // 1% tolerance

        for obs in observations {
            // Check various relationships
            let values = vec![
                obs.derived.offset_ratio,
                obs.scale.balance_ratio,
                obs.derived.fermat_a.to_f64().unwrap_or(0.0) / obs.n.to_f64().unwrap_or(1.0),
            ];

            for value in values {
                if (value - constant).abs() < epsilon {
                    appearances += 1;
                    break;
                }
            }
        }

        let frequency = appearances as f64 / observations.len() as f64;
        frequency >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Pattern, PatternKind};

    #[test]
    fn test_constant_discovery() {
        let mut patterns = vec![];

        // Add patterns with golden ratio
        let mut pattern1 = Pattern::new("test1", PatternKind::Invariant);
        pattern1.parameters = vec![1.618, 3.236]; // φ and 2φ
        patterns.push(pattern1);

        let mut pattern2 = Pattern::new("test2", PatternKind::ScaleDependent);
        pattern2.parameters = vec![0.809, 1.618]; // φ/2 and φ
        patterns.push(pattern2);

        let constants = ConstantDiscovery::extract(&patterns);

        // Should find golden ratio
        let phi = constants.iter().find(|c| c.name == "phi");
        assert!(phi.is_some());
    }
}
