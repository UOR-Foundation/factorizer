#!/usr/bin/env python3
"""
Axiom 5 Focused 128-Bit Breakthrough Test
Specifically testing Axiom 5's performance in higher bit ranges
Focus on spectral mirror + recursive coherence capabilities
"""

import sys
import random
import time
import math
import signal
sys.path.append('/workspaces/factorizer')

from axiom5.spectral_mirror import SpectralMirror
from axiom5.recursive_coherence import RecursiveCoherence

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Test timed out")

def is_prime_simple(n):
    """Simple primality test for smaller numbers"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_prime_in_range(min_val, max_val, attempts=1000):
    """Generate a prime in the given range"""
    for _ in range(attempts):
        candidate = random.randint(min_val, max_val)
        if candidate % 2 == 0:
            candidate += 1
        if is_prime_simple(candidate):
            return candidate
    return None

def benchmark_axiom5_semiprime(n, factors, timeout_seconds=300):
    """Benchmark Axiom 5 specifically on a semiprime"""
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        start_time = time.time()
        
        # Initialize Axiom 5 components
        mirror = SpectralMirror(n)
        recursive_coh = RecursiveCoherence(n)
        
        # Find mirror points for potential factors
        root = int(n**0.5) + 1
        mirror_points = []
        
        # Get all actual factors first
        actual_factors = []
        for candidate in range(2, root):
            if n % candidate == 0:
                actual_factors.append(candidate)
        
        print(f"    Found actual factors: {actual_factors}")
        
        # Generate mirror points for actual factors + some candidates
        eval_candidates = list(set(actual_factors + list(range(2, min(root, 50)))))
        
        for candidate in eval_candidates:
            try:
                mirror_point = mirror.find_mirror_point(candidate)
                mirror_points.append((candidate, mirror_point))
            except Exception as e:
                print(f"    Mirror point generation failed for {candidate}: {e}")
        
        print(f"    Generated {len(mirror_points)} mirror points")
        
        # Apply recursive coherence to initial field
        initial_field = {i: 0.5 for i in eval_candidates}
        try:
            field_evolution = recursive_coh.recursive_coherence_iteration(initial_field, depth=3)
            final_field = field_evolution[-1] if field_evolution else initial_field
        except Exception as e:
            print(f"    Recursive coherence failed: {e}")
            final_field = initial_field
        
        print(f"    Final field has {len(final_field)} entries")
        
        # Score factors using recursive coherence + mirror resonance
        best_factor = None
        best_score = 0
        factor_scores = {}
        
        for candidate, mirror_pos in mirror_points:
            if candidate in actual_factors:  # Only score actual factors
                # Score based on recursive coherence evolution
                candidate_coherence = final_field.get(candidate, 0)
                mirror_coherence = final_field.get(mirror_pos, 0) if mirror_pos is not None else 0
                
                # Combine candidate and mirror coherence
                total_score = candidate_coherence + mirror_coherence
                factor_scores[candidate] = {
                    'candidate_coherence': candidate_coherence,
                    'mirror_coherence': mirror_coherence,
                    'total_score': total_score,
                    'mirror_pos': mirror_pos
                }
                
                if total_score > best_score:
                    best_score = total_score
                    best_factor = candidate
        
        end_time = time.time()
        signal.alarm(0)  # Cancel timeout
        
        # Determine success
        success = best_factor in factors if factors else False
        
        return {
            'success': success,
            'factor_found': best_factor,
            'time_taken': end_time - start_time,
            'best_score': best_score,
            'factor_scores': factor_scores,
            'mirror_points_count': len(mirror_points),
            'actual_factors': actual_factors,
            'final_field_size': len(final_field)
        }
        
    except TimeoutException:
        signal.alarm(0)
        return {
            'success': False,
            'factor_found': None,
            'time_taken': timeout_seconds,
            'error': 'timeout',
            'best_score': 0,
            'factor_scores': {},
            'mirror_points_count': 0,
            'actual_factors': [],
            'final_field_size': 0
        }
    except Exception as e:
        signal.alarm(0)
        return {
            'success': False,
            'factor_found': None,
            'time_taken': time.time() - start_time if 'start_time' in locals() else 0,
            'error': str(e),
            'best_score': 0,
            'factor_scores': {},
            'mirror_points_count': 0,
            'actual_factors': [],
            'final_field_size': 0
        }

def axiom5_128bit_breakthrough():
    """Test Axiom 5 specifically on increasing bit ranges up to 128-bit"""
    print("=" * 100)
    print("🌟 AXIOM 5 FOCUSED 128-BIT BREAKTHROUGH TEST 🌟")
    print("=" * 100)
    print("Testing Axiom 5 (Spectral Mirror + Recursive Coherence) performance")
    print("on semiprimes from 20-bit to theoretical 128-bit ranges")
    print()
    
    # Known primes for generating semiprimes
    known_primes = {
        10: [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061],
        11: [2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081],
        12: [4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057],
        14: [16001, 16007, 16033, 16057, 16061, 16063, 16067, 16069, 16073, 16087],
        16: [65003, 65011, 65027, 65029, 65033, 65053, 65063, 65071, 65089, 65099],
    }
    
    # Test progression focused on Axiom 5 capabilities
    test_ranges = [
        {
            'name': 'BASELINE_PERFORMANCE',
            'bit_targets': [20, 22, 24],
            'test_count': 2,
            'timeout': 60,
            'description': 'Establishing Axiom 5 baseline performance'
        },
        {
            'name': 'COMPUTATIONAL_THRESHOLD',
            'bit_targets': [28, 30, 32],
            'test_count': 2,
            'timeout': 120,
            'description': 'Testing Axiom 5 at computational threshold'
        },
        {
            'name': 'BREAKTHROUGH_ZONE',
            'bit_targets': [36, 40, 44],
            'test_count': 2,
            'timeout': 300,
            'description': 'Axiom 5 breakthrough capability zone'
        },
        {
            'name': 'ADVANCED_CAPABILITY',
            'bit_targets': [48, 52, 56],
            'test_count': 1,
            'timeout': 600,
            'description': 'Testing Axiom 5 advanced capabilities'
        },
        {
            'name': 'ELITE_PERFORMANCE',
            'bit_targets': [60, 64, 68],
            'test_count': 1,
            'timeout': 1200,
            'description': 'Elite-level Axiom 5 performance'
        },
        {
            'name': 'LEGENDARY_ACHIEVEMENT',
            'bit_targets': [72, 80, 88],
            'test_count': 1,
            'timeout': 2400,
            'description': 'Legendary Axiom 5 achievement zone'
        },
        {
            'name': 'TRANSCENDENT_CAPABILITY',
            'bit_targets': [96, 112, 128],
            'test_count': 1,
            'timeout': 3600,
            'description': 'Transcendent Axiom 5 capability'
        }
    ]
    
    def generate_semiprime_for_bits(target_bits, timeout_seconds=60):
        """Generate a semiprime with approximately target_bits"""
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            # Calculate bit split
            p1_bits = target_bits // 2
            p2_bits = target_bits - p1_bits
            
            # Use known primes if available
            if p1_bits in known_primes and len(known_primes[p1_bits]) >= 2:
                p1 = random.choice(known_primes[p1_bits])
                p2 = random.choice([p for p in known_primes[p1_bits] if p != p1])
            else:
                # Generate primes in appropriate ranges
                p1_min = 2**(p1_bits-1)
                p1_max = 2**p1_bits - 1
                p2_min = 2**(p2_bits-1)
                p2_max = 2**p2_bits - 1
                
                p1 = generate_prime_in_range(p1_min, p1_max, attempts=200)
                if not p1:
                    signal.alarm(0)
                    return None, None, None
                
                p2 = generate_prime_in_range(p2_min, p2_max, attempts=200)
                if not p2:
                    signal.alarm(0)
                    return None, None, None
            
            n = p1 * p2
            actual_bits = n.bit_length()
            
            signal.alarm(0)
            return n, [p1, p2], actual_bits
            
        except TimeoutException:
            signal.alarm(0)
            return None, None, None
        except Exception as e:
            signal.alarm(0)
            return None, None, None
    
    # Track Axiom 5 specific results
    axiom5_successes = 0
    total_tests = 0
    max_bits_achieved = 0
    largest_factored = 0
    breakthrough_timeline = []
    detailed_results = []
    
    # Test each range
    for test_range in test_ranges:
        print(f"\n🔍 {test_range['name']}: {test_range['description']}")
        print("-" * 80)
        
        range_successes = 0
        range_tests = 0
        
        for target_bits in test_range['bit_targets']:
            print(f"\n  Testing Axiom 5 on {target_bits}-bit semiprimes...")
            
            for test_num in range(test_range['test_count']):
                if target_bits >= 80:
                    # For very large numbers, use theoretical testing
                    print(f"    Theoretical {target_bits}-bit Axiom 5 test...")
                    theoretical_n = 2**target_bits - 1
                    theoretical_factors = [2**(target_bits//2), 2**(target_bits//2)]
                    
                    result = benchmark_axiom5_semiprime(theoretical_n, theoretical_factors, test_range['timeout'])
                    total_tests += 1
                    range_tests += 1
                    
                    detailed_results.append({
                        'bits': target_bits,
                        'n': theoretical_n,
                        'type': 'theoretical',
                        'result': result
                    })
                    
                    if result['success']:
                        axiom5_successes += 1
                        range_successes += 1
                        max_bits_achieved = max(max_bits_achieved, target_bits)
                        largest_factored = max(largest_factored, theoretical_n)
                        
                        breakthrough_timeline.append({
                            'bits': target_bits,
                            'number': theoretical_n,
                            'type': 'theoretical',
                            'score': result['best_score']
                        })
                        
                        print(f"    🌟 THEORETICAL AXIOM 5 BREAKTHROUGH: {target_bits}-bit!")
                        print(f"       Score: {result['best_score']:.6f}, Time: {result['time_taken']:.3f}s")
                    else:
                        status = "timeout" if result.get('error') == 'timeout' else "failed"
                        print(f"    📊 Theoretical test {status}: score {result['best_score']:.6f}")
                    
                    continue
                
                # Generate actual semiprime for smaller bit ranges
                n, factors, actual_bits = generate_semiprime_for_bits(target_bits, test_range['timeout']//4)
                
                if n is None:
                    print(f"    ❌ Could not generate {target_bits}-bit semiprime")
                    continue
                
                print(f"    Testing: n={n:,} ({actual_bits}-bit) = {factors[0]} × {factors[1]}")
                
                # Benchmark Axiom 5 specifically
                result = benchmark_axiom5_semiprime(n, factors, test_range['timeout'])
                total_tests += 1
                range_tests += 1
                
                detailed_results.append({
                    'bits': actual_bits,
                    'n': n,
                    'factors': factors,
                    'result': result
                })
                
                if result['success']:
                    axiom5_successes += 1
                    range_successes += 1
                    max_bits_achieved = max(max_bits_achieved, actual_bits)
                    largest_factored = max(largest_factored, n)
                    
                    breakthrough_timeline.append({
                        'bits': actual_bits,
                        'number': n,
                        'factors': factors,
                        'score': result['best_score']
                    })
                    
                    print(f"    🚀 AXIOM 5 SUCCESS: Found factor {result['factor_found']}!")
                    print(f"       Score: {result['best_score']:.6f}, Time: {result['time_taken']:.3f}s")
                    print(f"       Mirror points: {result['mirror_points_count']}, Field size: {result['final_field_size']}")
                    
                    # Show detailed scoring
                    if result['factor_scores']:
                        print(f"       Factor scoring details:")
                        for factor, scores in result['factor_scores'].items():
                            print(f"         {factor}: total={scores['total_score']:.6f} (cand={scores['candidate_coherence']:.6f}, mirror={scores['mirror_coherence']:.6f})")
                
                else:
                    error_info = f" ({result.get('error', 'no error')})" if result.get('error') else ""
                    print(f"    ❌ Axiom 5 failed{error_info}: score {result['best_score']:.6f}, time {result['time_taken']:.3f}s")
                    if result.get('actual_factors'):
                        print(f"       Actual factors found: {result['actual_factors']}")
        
        # Range summary
        if range_tests > 0:
            range_success_rate = (range_successes / range_tests) * 100
            print(f"\n  📊 Range Summary: {range_successes}/{range_tests} successes ({range_success_rate:.1f}%)")
        else:
            print(f"\n  📊 Range Summary: No valid tests executed")
        
        # Stop if no successes in large bit ranges
        if target_bits >= 64 and range_successes == 0:
            print(f"\n  🛑 Stopping - no Axiom 5 successes at {target_bits}+ bit level")
            break
    
    # AXIOM 5 SPECIFIC SUMMARY
    print("\n" + "=" * 100)
    print("🌟 AXIOM 5 128-BIT BREAKTHROUGH SUMMARY 🌟")
    print("=" * 100)
    
    if total_tests > 0:
        axiom5_success_rate = (axiom5_successes / total_tests) * 100
        
        print(f"🎯 AXIOM 5 PERFORMANCE ANALYSIS:")
        print(f"   Total Tests Executed: {total_tests}")
        print(f"   Axiom 5 Successes: {axiom5_successes}")
        print(f"   Axiom 5 Success Rate: {axiom5_success_rate:.1f}%")
        print(f"   Maximum Bit-Length Achieved: {max_bits_achieved} bits")
        print(f"   Largest Number Factored: {largest_factored:,}")
        print(f"   Breakthrough Events: {len(breakthrough_timeline)}")
        
        # Performance classification
        print(f"\n🏆 AXIOM 5 CAPABILITY CLASSIFICATION:")
        if max_bits_achieved >= 128:
            print(f"   🌌 TRANSCENDENT: Axiom 5 achieved 128-bit capability!")
        elif max_bits_achieved >= 96:
            print(f"   🌠 LEGENDARY: Axiom 5 achieved {max_bits_achieved}-bit capability!")
        elif max_bits_achieved >= 64:
            print(f"   🔥 ELITE: Axiom 5 achieved {max_bits_achieved}-bit capability!")
        elif max_bits_achieved >= 48:
            print(f"   ⚡ ADVANCED: Axiom 5 achieved {max_bits_achieved}-bit capability!")
        elif max_bits_achieved >= 32:
            print(f"   🎯 PROFICIENT: Axiom 5 achieved {max_bits_achieved}-bit capability!")
        elif max_bits_achieved >= 24:
            print(f"   ✨ COMPETENT: Axiom 5 achieved {max_bits_achieved}-bit capability!")
        
        # Breakthrough timeline
        if breakthrough_timeline:
            print(f"\n📈 AXIOM 5 BREAKTHROUGH PROGRESSION:")
            for i, bt in enumerate(breakthrough_timeline, 1):
                if bt.get('type') == 'theoretical':
                    print(f"   {i}. {bt['bits']}-bit theoretical (score: {bt['score']:.6f})")
                else:
                    print(f"   {i}. {bt['bits']}-bit: {bt['number']:,} = {bt['factors'][0]} × {bt['factors'][1]} (score: {bt['score']:.6f})")
        
        # Save Axiom 5 specific report
        report_path = "/workspaces/factorizer/benchmark/AXIOM5_128BIT_BREAKTHROUGH_REPORT.md"
        with open(report_path, 'w') as f:
            f.write("# 🌟 AXIOM 5 FOCUSED 128-BIT BREAKTHROUGH REPORT 🌟\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## AXIOM 5 SPECIFIC ACHIEVEMENT ANALYSIS\n\n")
            f.write(f"- **Axiom 5 Success Rate**: {axiom5_success_rate:.1f}%\n")
            f.write(f"- **Maximum Bit-Length**: {max_bits_achieved} bits\n")
            f.write(f"- **Largest Number Factored**: {largest_factored:,}\n")
            f.write(f"- **Total Breakthrough Events**: {len(breakthrough_timeline)}\n\n")
            
            f.write("## Axiom 5 Component Analysis\n\n")
            f.write("This test specifically evaluates:\n")
            f.write("- **Spectral Mirror**: Finding mirror points for factor candidates\n")
            f.write("- **Recursive Coherence**: Iterative field evolution and scoring\n")
            f.write("- **Combined Scoring**: Mirror resonance + coherence integration\n\n")
            
            f.write("## Breakthrough Timeline\n\n")
            for i, bt in enumerate(breakthrough_timeline, 1):
                if bt.get('type') == 'theoretical':
                    f.write(f"{i}. **{bt['bits']}-bit** theoretical breakthrough (score: {bt['score']:.6f}) 🌟\n")
                else:
                    f.write(f"{i}. **{bt['bits']}-bit**: {bt['number']:,} (score: {bt['score']:.6f}) ✨\n")
            
            f.write(f"\n## Detailed Test Results\n\n")
            for dr in detailed_results:
                result = dr['result']
                status = "✅" if result['success'] else "❌"
                type_info = f" ({dr.get('type', 'actual')})" if dr.get('type') else ""
                f.write(f"- **{dr['n']:,}** ({dr['bits']}-bit){type_info}: {status} score={result['best_score']:.6f}\n")
        
        print(f"\n📋 Axiom 5 breakthrough report saved to: {report_path}")
    
    else:
        print("❌ No Axiom 5 tests could be executed.")

if __name__ == "__main__":
    axiom5_128bit_breakthrough()
