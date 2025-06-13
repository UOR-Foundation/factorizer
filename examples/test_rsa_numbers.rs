//! Test RSA-style balanced semiprimes

use rust_pattern_solver::pattern::universal_pattern::UniversalPattern;
use rust_pattern_solver::types::Number;
use std::str::FromStr;
use std::time::Instant;

fn test_rsa_number(name: &str, n_str: &str, p_str: &str, q_str: &str) -> bool {
    let n = Number::from_str(n_str).expect("Invalid n");
    let expected_p = Number::from_str(p_str).expect("Invalid p");
    let expected_q = Number::from_str(q_str).expect("Invalid q");
    
    println!("\nTesting {}: {} bits", name, n.bit_length());
    
    let mut pattern = UniversalPattern::with_precomputed_basis();
    
    let start = Instant::now();
    
    // Recognition
    let recognition = match pattern.recognize(&n) {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ Recognition failed: {}", e);
            return false;
        }
    };
    
    // Formalization
    let formalization = match pattern.formalize(recognition) {
        Ok(f) => f,
        Err(e) => {
            println!("  ✗ Formalization failed: {}", e);
            return false;
        }
    };
    
    // Execution
    match pattern.execute(formalization) {
        Ok(factors) => {
            let elapsed = start.elapsed();
            let correct = (&factors.p == &expected_p && &factors.q == &expected_q) ||
                         (&factors.p == &expected_q && &factors.q == &expected_p);
            
            if correct {
                println!("  ✓ Success via {} in {:?}", factors.method, elapsed);
                true
            } else {
                println!("  ✗ Wrong factors: {} × {}", factors.p, factors.q);
                false
            }
        }
        Err(e) => {
            println!("  ✗ Execution failed: {} (after {:?})", e, start.elapsed());
            false
        }
    }
}

fn main() {
    println!("=== Testing RSA-style Balanced Semiprimes ===\n");
    
    let mut successes = 0;
    let mut total = 0;
    
    // RSA-100 (330 bits)
    total += 1;
    if test_rsa_number(
        "RSA-100",
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139",
        "37975227936943673922808872755445627854565536638199",
        "40094690950920881030683735292761468389214899724061"
    ) {
        successes += 1;
    }
    
    // RSA-110 (364 bits)
    total += 1;
    if test_rsa_number(
        "RSA-110",
        "35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667",
        "6122421090493547576937037317561418841225758554253106999",
        "5846418214406154678836553182979162384198610505601062333"
    ) {
        successes += 1;
    }
    
    // RSA-120 (397 bits)
    total += 1;
    if test_rsa_number(
        "RSA-120",
        "227010481295437363334259960947493668895875336466084780038173258247009162675779735389791151574049166747880487470296548479",
        "327414555693498015751146303749141488063642403240171463406883",
        "693342667110830181197325401899700641361965863127336680673013"
    ) {
        successes += 1;
    }
    
    // RSA-129 (426 bits)
    total += 1;
    if test_rsa_number(
        "RSA-129",
        "114381625757888867669235779976146612010218296721242362562561842935706935245733897830597123563958705058989075147599290026879543541",
        "3490529510847650949147849619903898133417764638493387843990820577",
        "32769132993266709549961988190834461413177642967992942539798288533"
    ) {
        successes += 1;
    }
    
    // Custom 256-bit balanced semiprime
    total += 1;
    if test_rsa_number(
        "256-bit balanced",
        "85070313213500165033452292858360183929161102774993270926993942594878016663201",
        "257135571819915593992997182071",
        "330821992834951303476866502231"
    ) {
        successes += 1;
    }
    
    // Custom 512-bit balanced semiprime (two 256-bit primes)
    total += 1;
    if test_rsa_number(
        "512-bit balanced",
        "7237005577332262213973186563042994240857116359379907606001950938285454250989121297166690784551211576763485732206612603489213347158775318339695457997278273",
        "85070313213500165033452292858360183929161102774993270926993942594878016663201",
        "85070313213500165033452292858360183929161102774993270926993942594878016663193"
    ) {
        successes += 1;
    }
    
    println!("\n=== Summary ===");
    println!("Success rate: {}/{} ({:.1}%)", 
             successes, total, 100.0 * successes as f64 / total as f64);
    
    if successes == total {
        println!("\n✓ All tests passed!");
    } else {
        println!("\n✗ Some tests failed.");
    }
}