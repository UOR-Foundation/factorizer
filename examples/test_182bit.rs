use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;

fn main() {
    // The 182-bit perfect square that's hanging
    let n_str = "5316911983139663487012911223901898752222907725421778827677637920196241536907171387003542222693990401";
    let n = Number::from_str(n_str).unwrap();
    
    println!("Testing {}-bit number", n.bit_length());
    
    // First test: Check if sqrt is computed correctly
    let sqrt_n = rust_pattern_solver::utils::integer_sqrt(&n).unwrap();
    println!("sqrt_n = {}", sqrt_n);
    
    // Check if it's actually a perfect square
    let sqrt_squared = &sqrt_n * &sqrt_n;
    println!("sqrt² == n? {}", sqrt_squared == n);
    
    // Check division
    if &n % &sqrt_n == Number::from(0u32) {
        let quotient = &n / &sqrt_n;
        println!("n / sqrt_n = {}", quotient);
        println!("quotient == sqrt_n? {}", quotient == sqrt_n);
    }
    
    // Expected factors
    let expected = Number::from_str("72949702165160697506058194296497658716930225667548208555007761118604067061").unwrap();
    println!("Expected factor: {}", expected);
    println!("sqrt_n == expected? {}", sqrt_n == expected);
    
    // Try factoring with pattern
    let mut pattern = UniversalPattern::new();
    println!("\nTrying to factor with pattern...");
    
    // Recognition phase
    match pattern.recognize(&n) {
        Ok(recognition) => {
            println!("Recognition successful");
            
            // Formalization phase
            match pattern.formalize(recognition) {
                Ok(formalization) => {
                    println!("Formalization successful");
                    
                    // Execution phase
                    match pattern.execute(formalization) {
                        Ok(factors) => println!("Success: {} × {}", factors.p, factors.q),
                        Err(e) => println!("Execution error: {}", e),
                    }
                }
                Err(e) => println!("Formalization error: {}", e),
            }
        }
        Err(e) => println!("Recognition error: {}", e),
    }
}