//! Command-line interface for The Pattern

use clap::{Parser, Subcommand};
use log::info;
use rust_pattern_solver::{Observer, Pattern, Result};
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

#[derive(clap::ValueEnum, Clone)]
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
            start,
            end,
            output,
            threads,
        } => {
            info!("Observing patterns from {} to {}", start, end);

            let observer = Observer::new();
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build_global()
                    .unwrap();
            }

            let observations = observer.observe_range(start..end);

            if let Some(output_path) = output {
                info!("Writing observations to {:?}", output_path);
                std::fs::write(
                    output_path,
                    serde_json::to_string_pretty(&observations)?,
                )?;
            } else {
                println!("Collected {} observations", observations.len());
            }
        }

        Commands::Recognize {
            number,
            pattern,
            detailed,
        } => {
            info!("Recognizing factors of {}", number);

            let n = number.parse::<rust_pattern_solver::Number>()
                .map_err(|_| rust_pattern_solver::PatternError::InvalidInput(
                    "Invalid number format".to_string()
                ))?;

            let pattern_data = std::fs::read_to_string(pattern)?;
            let pattern = Pattern::from_json(&pattern_data)?;

            let recognition = pattern.recognize(&n)?;

            if detailed {
                println!("Recognition: {:#?}", recognition);
            }

            let formalization = pattern.formalize(recognition)?;
            let factors = pattern.execute(formalization)?;

            println!("Factors: {} × {} = {}", factors.p, factors.q, &factors.p * &factors.q);
        }

        Commands::Discover { input, output } => {
            info!("Discovering patterns from {:?}", input);

            let observations_data = std::fs::read_to_string(input)?;
            let observations: Vec<rust_pattern_solver::Observation> =
                serde_json::from_str(&observations_data)?;

            let patterns = rust_pattern_solver::observer::Analyzer::find_patterns(&observations)?;

            if let Some(output_path) = output {
                info!("Writing patterns to {:?}", output_path);
                std::fs::write(
                    output_path,
                    serde_json::to_string_pretty(&patterns)?,
                )?;
            } else {
                println!("Discovered {} patterns", patterns.len());
                for pattern in patterns {
                    println!("  - {}", pattern.describe());
                }
            }
        }

        Commands::Analyze { scale, pattern } => {
            info!("Analyzing patterns at {:?} scale", scale);

            let bit_range = match scale {
                Scale::Small => (8, 16),
                Scale::Medium => (16, 32),
                Scale::Large => (32, 64),
                Scale::Extreme => (64, 128),
            };

            println!("Analyzing {}-{} bit numbers", bit_range.0, bit_range.1);

            if let Some(pattern_name) = pattern {
                println!("Focusing on pattern: {}", pattern_name);
            }

            // TODO: Implement scale analysis
            println!("Scale analysis not yet implemented");
        }
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