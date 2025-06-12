//! Command-line interface for The Pattern

use clap::{Parser, Subcommand};
use log::info;
use rust_pattern_solver::{observer::ObservationCollector, types::Number, Result};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pattern-solver")]
#[command(about = "The Pattern - Recognition through empirical observation")]
#[command(version)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Observe patterns in factorizations
    Observe {
        /// Number to observe (if not provided, observes a range)
        number: Option<String>,

        /// Range start for observation
        #[arg(long, default_value = "1")]
        start: u64,

        /// Range end for observation
        #[arg(long, default_value = "1000000")]
        end: u64,

        /// Output file for observations
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of threads for parallel observation
        #[arg(short, long)]
        threads: Option<usize>,
    },

    /// Recognize factors of a number
    Recognize {
        /// Number to recognize
        number: String,

        /// Pattern data file
        #[arg(short, long, default_value = "data/analysis/universal.json")]
        pattern: PathBuf,

        /// Show detailed recognition process
        #[arg(long)]
        detailed: bool,
    },

    /// Discover patterns from observations
    Discover {
        /// Input observations file
        #[arg(short, long, default_value = "data/collection/observations.json")]
        input: PathBuf,

        /// Output patterns file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Analyze patterns at different scales
    Analyze {
        /// Scale to analyze
        #[arg(value_enum)]
        scale: Scale,

        /// Pattern type to analyze
        #[arg(short, long)]
        pattern: Option<String>,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Scale {
    Small,
    Medium,
    Large,
    Extreme,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    env_logger::Builder::new()
        .filter_level(match cli.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            2 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .init();

    match cli.command {
        Commands::Observe {
            number,
            start,
            end,
            output,
            threads,
        } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(t).build_global().unwrap();
            }

            let mut collector = ObservationCollector::new();

            let observations = if let Some(num_str) = number {
                // Observe a single number
                info!("Observing pattern for {}", num_str);
                let n = num_str.parse::<Number>().map_err(|_| {
                    rust_pattern_solver::error::PatternError::InvalidInput(
                        "Invalid number format".to_string(),
                    )
                })?;

                match collector.observe_single(n.clone()) {
                    Ok(obs) => {
                        println!("Number: {}", obs.n);
                        println!("Factors: {} × {}", obs.p, obs.q);
                        println!("Pattern type: {:?}", obs.scale.pattern_type);
                        vec![obs]
                    },
                    Err(e) => {
                        eprintln!("Failed to factor {}: {}", num_str, e);
                        return Err(e);
                    },
                }
            } else {
                // Observe a range
                info!("Observing patterns from {} to {}", start, end);
                collector.collect_range(start..end)
            };

            if let Some(output_path) = output {
                info!("Writing observations to {:?}", output_path);
                // Ensure directory exists
                if let Some(parent) = output_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(output_path, serde_json::to_string_pretty(&observations)?)?;
            } else {
                println!("Collected {} observations", observations.len());
            }
        },

        Commands::Recognize {
            number,
            pattern,
            detailed,
        } => {
            info!("Recognizing factors of {}", number);

            let n = number.parse::<Number>().map_err(|_| {
                rust_pattern_solver::error::PatternError::InvalidInput(
                    "Invalid number format".to_string(),
                )
            })?;

            // Load patterns from file
            let pattern_data = std::fs::read_to_string(&pattern)?;
            let patterns: Vec<rust_pattern_solver::types::Pattern> =
                serde_json::from_str(&pattern_data)?;

            if patterns.is_empty() {
                return Err(rust_pattern_solver::error::PatternError::PatternNotFound(
                    "No patterns found in file".to_string(),
                ));
            }

            // Use pattern recognition pipeline
            let recognition =
                rust_pattern_solver::pattern::recognition::recognize(n.clone(), &patterns)?;

            if detailed {
                println!("Recognition: {:#?}", recognition);
            }

            let formalization = rust_pattern_solver::pattern::formalization::formalize(
                recognition,
                &patterns,
                &[],
            )?;
            let factors =
                rust_pattern_solver::pattern::execution::execute(formalization, &patterns)?;

            println!(
                "Factors: {} × {} = {}",
                factors.p,
                factors.q,
                &factors.p * &factors.q
            );
        },

        Commands::Discover { input, output } => {
            info!("Discovering patterns from {:?}", input);

            let observations_data = std::fs::read_to_string(input)?;
            let observations: Vec<rust_pattern_solver::types::Observation> =
                serde_json::from_str(&observations_data)?;

            let patterns =
                rust_pattern_solver::pattern::Pattern::discover_from_observations(&observations)?;

            if let Some(output_path) = output {
                info!("Writing patterns to {:?}", output_path);
                std::fs::write(output_path, serde_json::to_string_pretty(&patterns)?)?;
            } else {
                println!("Discovered {} patterns", patterns.len());
                for pattern in patterns {
                    println!("  - {}: {}", pattern.id, pattern.description);
                }
            }
        },

        Commands::Analyze { scale, pattern } => {
            info!("Analyzing patterns at {:?} scale", scale);

            let bit_range = match scale {
                Scale::Small => (8, 16),
                Scale::Medium => (16, 32),
                Scale::Large => (32, 64),
                Scale::Extreme => (64, 128),
            };

            println!("Analyzing {}-{} bit numbers", bit_range.0, bit_range.1);

            if let Some(ref pattern_name) = pattern {
                println!("Focusing on pattern: {}", pattern_name);
            }

            // Create observations for the scale
            let mut collector = ObservationCollector::new();
            let mut test_numbers = Vec::new();

            // Generate sample numbers for the scale
            let count = 100;
            for _ in 0..count {
                let p = rust_pattern_solver::utils::generate_random_prime(bit_range.0 / 2)?;
                let q = rust_pattern_solver::utils::generate_random_prime(bit_range.0 / 2)?;
                test_numbers.push(&p * &q);
            }

            println!(
                "Collecting observations for {} numbers...",
                test_numbers.len()
            );
            let observations = collector.observe_parallel(&test_numbers)?;

            // Discover patterns at this scale
            let patterns =
                rust_pattern_solver::pattern::Pattern::discover_from_observations(&observations)?;

            println!("\nPatterns at {:?} scale:", scale);
            if let Some(pattern_filter) = pattern {
                // Filter to specific pattern
                let filtered: Vec<_> =
                    patterns.into_iter().filter(|p| p.id.contains(&pattern_filter)).collect();

                if filtered.is_empty() {
                    println!("Pattern '{}' not found at this scale", pattern_filter);
                } else {
                    for p in filtered {
                        println!("  {}: {}", p.id, p.description);
                        println!("    Frequency: {:.1}%", p.frequency * 100.0);
                    }
                }
            } else {
                // Show all patterns
                println!("Found {} patterns:", patterns.len());
                for p in patterns.iter().take(10) {
                    println!("  {}: {}", p.id, p.description);
                    println!("    Frequency: {:.1}%", p.frequency * 100.0);
                }
                if patterns.len() > 10 {
                    println!("  ... and {} more", patterns.len() - 10);
                }
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }
}
