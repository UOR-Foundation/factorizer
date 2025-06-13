//! Generate and save pre-computed basis files

use rust_pattern_solver::pattern::basis_persistence;
use std::path::Path;

fn main() {
    println!("=== Generating Pre-computed Basis Files ===\n");
    
    // Create data/basis directory if it doesn't exist
    let basis_dir = Path::new("data/basis");
    
    // Generate and save both standard and enhanced basis
    match basis_persistence::generate_basis_files() {
        Ok(()) => {
            println!("\n✓ Successfully generated basis files!");
            println!("\nFiles created:");
            println!("  - data/basis/universal_basis.json");
            println!("  - data/basis/enhanced_basis.json");
            
            // Check file sizes
            if let Ok(standard_meta) = std::fs::metadata(basis_dir.join("universal_basis.json")) {
                println!("\nStandard basis size: {} KB", standard_meta.len() / 1024);
            }
            if let Ok(enhanced_meta) = std::fs::metadata(basis_dir.join("enhanced_basis.json")) {
                println!("Enhanced basis size: {} KB", enhanced_meta.len() / 1024);
            }
        }
        Err(e) => {
            eprintln!("✗ Error generating basis files: {}", e);
        }
    }
    
    println!("\nThese files will be loaded automatically by the pattern implementation.");
    println!("No need to regenerate them unless the basis algorithms change.");
}