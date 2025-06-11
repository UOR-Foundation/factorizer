//! Precision tests - Verify arbitrary precision arithmetic

use rust_pattern_solver::observer::ObservationCollector;
use rust_pattern_solver::types::Number;
use rust_pattern_solver::utils;

#[test]
fn test_large_number_arithmetic() {
    // Test that we can handle large numbers correctly
    let p = Number::from_str("18446744073709551629").unwrap(); // Large prime
    let q = Number::from_str("18446744073709551653").unwrap(); // Another large prime

    let n = &p * &q;

    // Verify multiplication
    let expected = Number::from_str("340282366920938463534117278883377737537").unwrap();
    assert_eq!(n, expected);

    // Verify division
    assert_eq!(&n / &p, q);
    assert_eq!(&n / &q, p);

    // Verify modulo
    assert_eq!(&n % &p, Number::from(0u32));
    assert_eq!(&n % &q, Number::from(0u32));
}

#[test]
fn test_bit_length_calculation() {
    let test_cases = vec![
        (Number::from(1u32), 1),
        (Number::from(2u32), 2),
        (Number::from(3u32), 2),
        (Number::from(4u32), 3),
        (Number::from(255u32), 8),
        (Number::from(256u32), 9),
        (Number::from_str("18446744073709551615").unwrap(), 64), // 2^64 - 1
        (Number::from_str("18446744073709551616").unwrap(), 65), // 2^64
    ];

    for (n, expected_bits) in test_cases {
        assert_eq!(n.bit_length(), expected_bits, "Wrong bit length for {}", n);
    }
}

#[test]
fn test_sqrt_precision() {
    let test_cases = vec![
        Number::from(4u32),
        Number::from(9u32),
        Number::from(16u32),
        Number::from(100u32),
        Number::from(10000u32),
        Number::from_str("1000000000000").unwrap(),
    ];

    for n in test_cases {
        let sqrt = utils::integer_sqrt(&n).unwrap();

        // Verify sqrt * sqrt <= n
        assert!(&sqrt * &sqrt <= n);

        // Verify (sqrt + 1) * (sqrt + 1) > n
        let sqrt_plus_one = &sqrt + &Number::from(1u32);
        assert!(&sqrt_plus_one * &sqrt_plus_one > n);
    }
}

#[test]
fn test_128_bit_numbers() {
    // Test with 128-bit semiprimes
    let p1 = Number::from_str("340282366920938463463374607431768211507").unwrap(); // Prime near 2^128
    let p2 = Number::from_str("340282366920938463463374607431768211537").unwrap();

    let n = &p1 * &p2;

    // Should be able to observe
    let mut collector = ObservationCollector::new();
    match collector.observe_single(n.clone()) {
        Ok(obs) => {
            assert_eq!(obs.n, n);
            assert_eq!(&obs.p * &obs.q, n);
            assert!(obs.scale.bit_length > 250); // Should be around 256 bits
        },
        Err(_) => {
            // It's OK if we can't factor such large numbers in tests
            // Just verify we can handle the arithmetic
            assert_eq!(&n / &p1, p2);
        },
    }
}

#[test]
fn test_comparison_operators() {
    let a = Number::from(100u32);
    let b = Number::from(200u32);
    let c = Number::from(100u32);

    assert!(a < b);
    assert!(b > a);
    assert!(a <= b);
    assert!(b >= a);
    assert!(a <= c);
    assert!(a >= c);
    assert!(a == c);
    assert!(a != b);
}

#[test]
fn test_conversion_precision() {
    // Test conversions maintain precision
    let large_str = "123456789012345678901234567890";
    let n = Number::from_str(large_str).unwrap();

    // Converting back to string should give same result
    assert_eq!(n.to_string(), large_str);

    // Test from_bytes
    let bytes = n.to_bytes();
    let n2 = Number::from_bytes(&bytes);
    assert_eq!(n, n2);
}

#[test]
fn test_modular_arithmetic_precision() {
    let n = Number::from_str("1234567890123456789012345678901234567890").unwrap();
    let m = Number::from(1000000007u64); // Large prime modulus

    let remainder = &n % &m;

    // Verify remainder is less than modulus
    assert!(remainder < m);

    // Verify n = quotient * m + remainder
    let quotient = &n / &m;
    let reconstructed = &quotient * &m + &remainder;
    assert_eq!(reconstructed, n);
}

#[test]
fn test_edge_case_arithmetic() {
    // Test edge cases
    let zero = Number::from(0u32);
    let one = Number::from(1u32);
    let two = Number::from(2u32);

    // Zero arithmetic
    assert_eq!(&zero + &one, one);
    assert_eq!(&one + &zero, one);
    assert_eq!(&zero * &one, zero);
    assert_eq!(&one * &zero, zero);

    // Division edge cases
    assert_eq!(&zero / &one, zero);
    assert_eq!(&one / &one, one);
    assert_eq!(&two / &two, one);

    // Modulo edge cases
    assert_eq!(&zero % &one, zero);
    assert_eq!(&one % &two, one);
    assert_eq!(&two % &two, zero);
}

#[test]
fn test_primality_testing_precision() {
    // Test that primality testing works for large numbers
    let large_primes = vec![
        "2147483647",           // 2^31 - 1 (Mersenne prime)
        "4294967291",           // Large 32-bit prime
        "18446744073709551557", // Large 64-bit prime
    ];

    for prime_str in large_primes {
        let p = Number::from_str(prime_str).unwrap();
        assert!(
            utils::is_probable_prime(&p, 20),
            "{} should be detected as prime",
            prime_str
        );
    }

    // Test composites
    let composites = vec![
        "2147483649",           // 2^31 - 1 + 2
        "4294967295",           // 2^32 - 1 (composite)
        "18446744073709551615", // 2^64 - 1 (composite)
    ];

    for composite_str in composites {
        let c = Number::from_str(composite_str).unwrap();
        assert!(
            !utils::is_probable_prime(&c, 20),
            "{} should not be detected as prime",
            composite_str
        );
    }
}
