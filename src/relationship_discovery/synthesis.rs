//! Synthesis of relationships into higher-order understanding

use crate::types::{Observation, Pattern, UniversalConstant};
use crate::Result;
use std::collections::HashMap;

/// Synthesized understanding of relationships
#[derive(Debug)]
pub struct RelationshipSynthesis {
    /// Meta-patterns discovered from pattern relationships
    pub meta_patterns: Vec<MetaPattern>,

    /// Emergent laws from observations
    pub emergent_laws: Vec<EmergentLaw>,

    /// Unified theory components
    pub unified_components: Vec<UnifiedComponent>,
}

/// Pattern that emerges from other patterns
#[derive(Debug, Clone)]
pub struct MetaPattern {
    /// Unique identifier
    pub id: String,

    /// Description
    pub description: String,

    /// Component patterns
    pub components: Vec<String>,

    /// Emergence strength
    pub strength: f64,

    /// Mathematical expression
    pub expression: Option<String>,
}

/// Law that emerges from observations
#[derive(Debug, Clone)]
pub struct EmergentLaw {
    /// Law name
    pub name: String,

    /// Mathematical statement
    pub statement: String,

    /// Supporting evidence count
    pub evidence_count: usize,

    /// Confidence level
    pub confidence: f64,

    /// Applicable scale range
    pub scale_range: Option<(usize, usize)>,
}

/// Component of unified theory
#[derive(Debug, Clone)]
pub struct UnifiedComponent {
    /// Component name
    pub name: String,

    /// Role in the theory
    pub role: String,

    /// Related constants
    pub constants: Vec<String>,

    /// Related patterns
    pub patterns: Vec<String>,

    /// Integration strength
    pub integration: f64,
}

impl RelationshipSynthesis {
    /// Synthesize relationships from all components
    pub fn synthesize(
        observations: &[Observation],
        patterns: &[Pattern],
        constants: &[UniversalConstant],
    ) -> Result<Self> {
        let meta_patterns = Self::discover_meta_patterns(patterns)?;
        let emergent_laws = Self::discover_emergent_laws(observations, patterns)?;
        let unified_components = Self::build_unified_theory(patterns, constants)?;

        Ok(Self {
            meta_patterns,
            emergent_laws,
            unified_components,
        })
    }

    /// Discover meta-patterns from pattern interactions
    fn discover_meta_patterns(patterns: &[Pattern]) -> Result<Vec<MetaPattern>> {
        let mut meta_patterns = Vec::new();

        // Group patterns by type
        let mut by_type: HashMap<String, Vec<&Pattern>> = HashMap::new();
        for pattern in patterns {
            let type_key = match &pattern.kind {
                crate::types::PatternKind::Invariant => "invariant",
                crate::types::PatternKind::ScaleDependent => "scale_dependent",
                crate::types::PatternKind::Universal => "universal",
                crate::types::PatternKind::TypeSpecific(_) => "type_specific",
                crate::types::PatternKind::Probabilistic => "probabilistic",
                crate::types::PatternKind::Emergent => "emergent",
            };
            by_type.entry(type_key.to_string()).or_insert_with(Vec::new).push(pattern);
        }

        // Look for patterns that appear together
        if let (Some(invariants), Some(universals)) =
            (by_type.get("invariant"), by_type.get("universal"))
        {
            if invariants.len() >= 3 && universals.len() >= 2 {
                meta_patterns.push(MetaPattern {
                    id: "invariant_universal_bridge".to_string(),
                    description: "Invariants connect to universal patterns through scale"
                        .to_string(),
                    components: invariants
                        .iter()
                        .take(3)
                        .map(|p| p.id.clone())
                        .chain(universals.iter().take(2).map(|p| p.id.clone()))
                        .collect(),
                    strength: 0.8,
                    expression: Some("∀n: Invariant(n) ∧ Scale(n) → Universal(n)".to_string()),
                });
            }
        }

        // Look for hierarchical patterns
        let hierarchical_patterns: Vec<&Pattern> = patterns
            .iter()
            .filter(|p| {
                p.scale_range.unbounded || (p.scale_range.max_bits - p.scale_range.min_bits > 100)
            })
            .collect();

        if hierarchical_patterns.len() >= 3 {
            meta_patterns.push(MetaPattern {
                id: "scale_hierarchy".to_string(),
                description: "Patterns form hierarchical structure across scales".to_string(),
                components: hierarchical_patterns.iter().take(5).map(|p| p.id.clone()).collect(),
                strength: 0.7,
                expression: Some("Pattern(scale_n) ⊆ Pattern(scale_n+1)".to_string()),
            });
        }

        // Look for complementary patterns
        let balanced_patterns: Vec<&Pattern> =
            patterns.iter().filter(|p| p.id.contains("balanced")).collect();
        let harmonic_patterns: Vec<&Pattern> =
            patterns.iter().filter(|p| p.id.contains("harmonic")).collect();

        if !balanced_patterns.is_empty() && !harmonic_patterns.is_empty() {
            meta_patterns.push(MetaPattern {
                id: "balance_harmony_duality".to_string(),
                description: "Balance and harmony are dual aspects of semiprime structure"
                    .to_string(),
                components: vec![
                    balanced_patterns[0].id.clone(),
                    harmonic_patterns[0].id.clone(),
                ],
                strength: 0.6,
                expression: Some("Balanced(n) ↔ ¬Harmonic(n)".to_string()),
            });
        }

        Ok(meta_patterns)
    }

    /// Discover emergent laws from observations
    fn discover_emergent_laws(
        observations: &[Observation],
        patterns: &[Pattern],
    ) -> Result<Vec<EmergentLaw>> {
        let mut laws = Vec::new();

        // Law 1: Offset ratio bounds
        let offset_ratios: Vec<f64> =
            observations.iter().map(|obs| obs.derived.offset_ratio.abs()).collect();

        if offset_ratios.len() >= 10 {
            let max_ratio = offset_ratios.iter().fold(0.0f64, |a, &b| a.max(b));
            let min_ratio = offset_ratios.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            laws.push(EmergentLaw {
                name: "offset_ratio_bounds".to_string(),
                statement: format!(
                    "∀ semiprimes n=pq: {:.6} ≤ |offset/sqrt(n)| ≤ {:.6}",
                    min_ratio, max_ratio
                ),
                evidence_count: offset_ratios.len(),
                confidence: 0.95,
                scale_range: None,
            });
        }

        // Law 2: Balance ratio distribution
        let mut balance_distribution = HashMap::new();
        for obs in observations {
            let bucket = (obs.scale.balance_ratio * 10.0).round() as i32;
            *balance_distribution.entry(bucket).or_insert(0) += 1;
        }

        if let Some((&mode_bucket, &mode_count)) =
            balance_distribution.iter().max_by_key(|(_, &count)| count)
        {
            let mode_ratio = mode_bucket as f64 / 10.0;
            let mode_freq = mode_count as f64 / observations.len() as f64;

            if mode_freq > 0.2 {
                laws.push(EmergentLaw {
                    name: "balance_ratio_mode".to_string(),
                    statement: format!(
                        "Mode(balance_ratio) ≈ {:.1} with frequency {:.1}%",
                        mode_ratio,
                        mode_freq * 100.0
                    ),
                    evidence_count: observations.len(),
                    confidence: mode_freq,
                    scale_range: None,
                });
            }
        }

        // Law 3: Factor gap growth
        let mut by_scale: HashMap<usize, Vec<&Observation>> = HashMap::new();
        for obs in observations {
            by_scale.entry(obs.scale.bit_length / 8).or_insert_with(Vec::new).push(obs);
        }

        let mut scale_gaps = Vec::new();
        for (&scale, obs_list) in &by_scale {
            if obs_list.len() >= 5 {
                let avg_gap =
                    obs_list.iter().filter_map(|obs| obs.scale.prime_gap.to_f64()).sum::<f64>()
                        / obs_list.len() as f64;
                scale_gaps.push((scale, avg_gap));
            }
        }

        if scale_gaps.len() >= 3 {
            scale_gaps.sort_by_key(|&(scale, _)| scale);
            let growth_observed = scale_gaps.windows(2).all(|w| w[1].1 >= w[0].1);

            if growth_observed {
                laws.push(EmergentLaw {
                    name: "factor_gap_monotonic_growth".to_string(),
                    statement: "Factor gap grows monotonically with number size".to_string(),
                    evidence_count: scale_gaps.len(),
                    confidence: 0.85,
                    scale_range: Some((scale_gaps[0].0 * 8, scale_gaps.last().unwrap().0 * 8)),
                });
            }
        }

        Ok(laws)
    }

    /// Build components of unified theory
    fn build_unified_theory(
        patterns: &[Pattern],
        constants: &[UniversalConstant],
    ) -> Result<Vec<UnifiedComponent>> {
        let mut components = Vec::new();

        // Component 1: Universal constant framework
        if !constants.is_empty() {
            let constant_names: Vec<String> = constants.iter().map(|c| c.name.clone()).collect();

            components.push(UnifiedComponent {
                name: "universal_constant_framework".to_string(),
                role: "Provides fundamental scaling and relationships".to_string(),
                constants: constant_names,
                patterns: patterns
                    .iter()
                    .filter(|p| matches!(p.kind, crate::types::PatternKind::Universal))
                    .map(|p| p.id.clone())
                    .collect(),
                integration: 0.9,
            });
        }

        // Component 2: Scale invariance principle
        let scale_invariant_patterns: Vec<String> = patterns
            .iter()
            .filter(|p| p.scale_range.unbounded)
            .map(|p| p.id.clone())
            .collect();

        if scale_invariant_patterns.len() >= 3 {
            components.push(UnifiedComponent {
                name: "scale_invariance_principle".to_string(),
                role: "Ensures patterns hold across all scales".to_string(),
                constants: vec![],
                patterns: scale_invariant_patterns,
                integration: 0.85,
            });
        }

        // Component 3: Pattern duality
        components.push(UnifiedComponent {
            name: "pattern_duality".to_string(),
            role: "Every pattern has complementary aspects".to_string(),
            constants: vec!["phi".to_string()], // Golden ratio often appears in dualities
            patterns: patterns.iter().take(6).map(|p| p.id.clone()).collect(),
            integration: 0.7,
        });

        // Component 4: Emergence cascade
        if patterns.len() > 10 {
            components.push(UnifiedComponent {
                name: "emergence_cascade".to_string(),
                role: "Simple patterns combine to create complex behaviors".to_string(),
                constants: vec!["e".to_string()], // Exponential growth
                patterns: patterns
                    .iter()
                    .filter(|p| matches!(p.kind, crate::types::PatternKind::ScaleDependent))
                    .map(|p| p.id.clone())
                    .take(5)
                    .collect(),
                integration: 0.75,
            });
        }

        Ok(components)
    }

    /// Get synthesis summary
    pub fn summarize(&self) -> String {
        format!(
            "Relationship Synthesis:\n\
             - {} meta-patterns discovered\n\
             - {} emergent laws identified\n\
             - {} unified theory components\n\
             \n\
             Strongest meta-pattern: {}\n\
             Most confident law: {}\n\
             Most integrated component: {}",
            self.meta_patterns.len(),
            self.emergent_laws.len(),
            self.unified_components.len(),
            self.meta_patterns
                .iter()
                .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|p| &p.id)
                .unwrap_or(&"none".to_string()),
            self.emergent_laws
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .map(|l| &l.name)
                .unwrap_or(&"none".to_string()),
            self.unified_components
                .iter()
                .max_by(|a, b| a.integration.partial_cmp(&b.integration).unwrap())
                .map(|c| &c.name)
                .unwrap_or(&"none".to_string()),
        )
    }
}
