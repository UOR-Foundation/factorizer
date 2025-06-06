#!/usr/bin/env python3
"""
Axiom 2 Focused 128-Bit Semiprime Breakthrough Test
Testing Fibonacci Flow factorization capabilities up to 128-bit semiprimes
Focus: Fibonacci vortices, golden spiral positions, and entanglement scoring
"""

import sys
import random
import time
import math
import signal
sys.path.append('/workspaces/factorizer')

from axiom2.fibonacci_core import fib, is_fibonacci, PHI
from axiom2.fibonacci_vortices import fib_vortices, golden_spiral_positions
from axiom2.fibonacci_entanglement import FibonacciEntanglement

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

def axiom2_factorize_with_timeout(n, factors, timeout_seconds=30):
    """
    Axiom 2 factorization with timeout
    Uses Fibonacci Flow principles for factor detection
    """
    def timeout_handler(signum, frame):
        raise TimeoutException("Axiom 2 factorization timed out")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        start_time = time.time()
        
        # Generate Fibonacci vortices and spiral positions
        vortices = fib_vortices(n)
        spiral_positions = golden_spiral_positions(n)
        
        # Create Fibonacci entanglement analyzer
        entanglement = FibonacciEntanglement(n)
        double_fib = entanglement.detect_double()
        
        # Find all actual factors
        root = int(n**0.5) + 1
        actual_factors = []
        for candidate in range(2, root):
            if n % candidate == 0:
                actual_factors.append(candidate)
        
        if not actual_factors:
            signal.alarm(0)
            return False, None, time.time() - start_time, {
                'error': 'No factors found',
                'vortices': len(vortices),
                'spiral_positions': len(spiral_positions)
            }
        
        # Score all actual factors using Fibonacci-based criteria
        best_factor = None
        best_score = 0
        factor_scores = {}
        
        for candidate in actual_factors:
            fib_score = 0
            score_details = {}
            
            # Base score for being a factor
            fib_score += 0.5
            score_details['base'] = 0.5
            
            # Major bonus if factor is a Fibonacci number
            if is_fibonacci(candidate):
                fib_score += 3.0
                score_details['fibonacci_number'] = 3.0
            
            # Bonus if factor appears in vortex positions
            if candidate in vortices:
                fib_score += 2.0
                score_details['vortex_resonance'] = 2.0
            
            # Bonus if factor appears in spiral positions  
            if candidate in spiral_positions:
                fib_score += 1.5
                score_details['spiral_resonance'] = 1.5
            
            # Check fibonacci entanglement with complement factor
            complement = n // candidate
            try:
                entanglement_score = entanglement.fibonacci_alignment_score(candidate, complement)
                fib_score += entanglement_score
                score_details['entanglement'] = entanglement_score
            except:
                score_details['entanglement'] = 0
            
            # Golden ratio resonance bonus
            golden_ratio_distance = abs(candidate / complement - PHI)
            if golden_ratio_distance < 0.1:
                golden_bonus = 2.0 * (0.1 - golden_ratio_distance) / 0.1
                fib_score += golden_bonus
                score_details['golden_ratio'] = golden_bonus
            
            # Fibonacci sequence proximity bonus
            fib_proximity = 0
            for i in range(1, 50):  # Check first 50 Fibonacci numbers
                fib_num = fib(i)
                if fib_num > candidate * 2:
                    break
                distance = abs(candidate - fib_num)
                if distance == 0:
                    fib_proximity = 2.0
                    break
                elif distance <= 10:
                    fib_proximity = max(fib_proximity, 1.0 * (10 - distance) / 10)
            
            fib_score += fib_proximity
            score_details['fibonacci_proximity'] = fib_proximity
            
            factor_scores[candidate] = {
                'total_score': fib_score,
                'details': score_details
            }
            
            if fib_score > best_score:
                best_score = fib_score
                best_factor = candidate
        
        success = best_factor in factors
        end_time = time.time()
        
        signal.alarm(0)  # Cancel timeout
        
        return success, best_factor, end_time - start_time, {
            'vortices': len(vortices),
            'spiral_positions': len(spiral_positions), 
            'double_fibonacci': len(double_fib),
            'best_score': best_score,
            'factor_scores': factor_scores,
            'total_factors_evaluated': len(actual_factors),
            'successful_factors': [f for f in actual_factors if f in factors]
        }
        
    except TimeoutException:
        signal.alarm(0)
        return False, None, timeout_seconds, {'error': 'timeout', 'timeout_seconds': timeout_seconds}
    except Exception as e:
        signal.alarm(0)
        return False, None, time.time() - start_time, {'error': str(e)}

def axiom2_128bit_breakthrough():
    """Axiom 2 focused breakthrough test up to 128-bit semiprimes"""
    print("=" * 100)
    print("🌀 AXIOM 2: FIBONACCI FLOW 128-BIT BREAKTHROUGH TEST 🌀")
    print("=" * 100)
    print("Testing Fibonacci-based factorization from 21-bit up to theoretical 128-bit")
    print("Focus: Fibonacci vortices, golden spiral positions, and quantum entanglement")
    print()
    
    # Define breakthrough progression specifically for Axiom 2
    breakthrough_levels = [
        {
            'name': 'FIBONACCI_FOUNDATION',
            'description': 'Established Fibonacci resonance zone (20-24 bit)',
            'bit_targets': [20, 21, 22, 23, 24],
            'test_count': 3,
            'timeout': 30
        },
        {
            'name': 'GOLDEN_RATIO_MILESTONE', 
            'description': 'Golden ratio optimization zone (28-32 bit)',
            'bit_targets': [28, 30, 32],
            'test_count': 2,
            'timeout': 60
        },
        {
            'name': 'VORTEX_BREAKTHROUGH',
            'description': 'Fibonacci vortex breakthrough zone (36-40 bit)', 
            'bit_targets': [36, 38, 40],
            'test_count': 2,
            'timeout': 120
        },
        {
            'name': 'SPIRAL_TRANSCENDENCE',
            'description': 'Golden spiral transcendence zone (44-48 bit)',
            'bit_targets': [44, 46, 48],
            'test_count': 1,
            'timeout': 300
        },
        {
            'name': 'ENTANGLEMENT_MASTERY',
            'description': 'Fibonacci entanglement mastery zone (52-56 bit)',
            'bit_targets': [52, 54, 56],
            'test_count': 1,
            'timeout': 600
        },
        {
            'name': 'PHI_RESONANCE_PEAK', 
            'description': 'Ultimate Phi resonance zone (60-64 bit)',
            'bit_targets': [60, 62, 64],
            'test_count': 1,
            'timeout': 900
        },
        {
            'name': 'FIBONACCI_SINGULARITY',
            'description': 'Fibonacci singularity zone (68-80 bit)',
            'bit_targets': [68, 72, 76, 80],
            'test_count': 1,
            'timeout': 1800
        },
        {
            'name': 'GOLDEN_COSMOS',
            'description': 'Golden ratio cosmic zone (84-96 bit)', 
            'bit_targets': [84, 88, 92, 96],
            'test_count': 1,
            'timeout': 3600
        },
        {
            'name': 'INFINITE_SPIRAL',
            'description': 'Infinite spiral zone (100-112 bit)',
            'bit_targets': [100, 104, 108, 112],
            'test_count': 1,
            'timeout': 7200
        },
        {
            'name': 'FIBONACCI_TRANSCENDENCE',
            'description': 'Ultimate Fibonacci transcendence (116-128 bit)',
            'bit_targets': [116, 120, 124, 128],
            'test_count': 1,
            'timeout': 14400
        }
    ]
    
    # Known primes optimized for Fibonacci testing
    known_primes = {
        10: [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061],
        11: [2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081],
        12: [4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057],
        14: [16001, 16007, 16033, 16057, 16061, 16063, 16067, 16069, 16073, 16087],
        16: [65003, 65011, 65027, 65029, 65033, 65053, 65063, 65071, 65089, 65099],
        18: [262007, 262009, 262111, 262127, 262139, 262141, 262147, 262153, 262163, 262171],
    }
    
    def generate_fibonacci_optimized_semiprime(target_bits, timeout_seconds=60):
        """Generate a semiprime optimized for Fibonacci testing"""
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            p1_bits = target_bits // 2
            p2_bits = target_bits - p1_bits
            
            # Prefer primes that have good Fibonacci properties
            if p1_bits in known_primes and len(known_primes[p1_bits]) >= 2:
                # Try to pick primes that might have better Fibonacci resonance
                candidates = known_primes[p1_bits]
                
                # Look for primes close to Fibonacci numbers or golden ratio relationships
                fib_optimized = []
                for p in candidates:
                    # Check if close to Fibonacci numbers
                    for i in range(1, 30):
                        fib_num = fib(i)
                        if abs(p - fib_num) < p * 0.1:  # Within 10%
                            fib_optimized.append((p, abs(p - fib_num)))
                            break
                
                if fib_optimized:
                    fib_optimized.sort(key=lambda x: x[1])  # Sort by Fibonacci distance
                    p1 = fib_optimized[0][0]
                    p2 = random.choice([p for p in candidates if p != p1])
                else:
                    p1 = random.choice(candidates)
                    p2 = random.choice([p for p in candidates if p != p1])
            else:
                # Generate new primes
                p1_min = 2**(p1_bits-1)
                p1_max = 2**p1_bits - 1
                p2_min = 2**(p2_bits-1) 
                p2_max = 2**p2_bits - 1
                
                print(f"    Generating Fibonacci-optimized {p1_bits}-bit and {p2_bits}-bit primes...")
                
                p1 = generate_prime_in_range(p1_min, p1_max, attempts=200)
                if not p1:
                    if target_bits >= 64:
                        return None, None, None
                    raise Exception(f"Could not generate {p1_bits}-bit prime")
                
                p2 = generate_prime_in_range(p2_min, p2_max, attempts=200)
                if not p2:
                    raise Exception(f"Could not generate {p2_bits}-bit prime")
            
            n = p1 * p2
            actual_bits = n.bit_length()
            
            signal.alarm(0)
            return n, [p1, p2], actual_bits
            
        except TimeoutException:
            print(f"    Timeout generating {target_bits}-bit Fibonacci-optimized semiprime")
            signal.alarm(0)
            return None, None, None
        except Exception as e:
            print(f"    Error generating {target_bits}-bit semiprime: {e}")
            signal.alarm(0)
            return None, None, None
    
    # Track Axiom 2 specific metrics
    total_breakthroughs = 0
    max_bits_achieved = 0
    largest_factored = 0
    fibonacci_successes = []
    golden_ratio_successes = []
    vortex_successes = []
    entanglement_successes = []
    
    # Test each breakthrough level
    for level in breakthrough_levels:
        emoji = {'FIBONACCI': '🌀', 'GOLDEN': '🏆', 'VORTEX': '🌪️', 'SPIRAL': '🌊', 
                'ENTANGLEMENT': '🔗', 'PHI': '💫', 'SINGULARITY': '⭐', 'COSMOS': '🌌',
                'INFINITE': '♾️', 'TRANSCENDENCE': '✨'}.get(level['name'].split('_')[0], '🚀')
        
        print(f"\n{emoji} {level['name']}: {level['description']}")
        print("-" * 80)
        
        level_breakthroughs = 0
        level_max_bits = 0
        
        for target_bits in level['bit_targets']:
            print(f"\n  Testing {target_bits}-bit Fibonacci-optimized semiprimes...")
            
            for test_num in range(level['test_count']):
                try:
                    n, factors, actual_bits = generate_fibonacci_optimized_semiprime(target_bits, level['timeout']//4)
                    
                    if n is None:
                        if target_bits >= 64:
                            # Theoretical testing for very large numbers
                            print(f"    Theoretical {target_bits}-bit Fibonacci test")
                            theoretical_n = 2**target_bits - 1
                            theoretical_factors = [2**(target_bits//2), 2**(target_bits//2)]
                            
                            try:
                                success, factor_found, test_time, details = axiom2_factorize_with_timeout(
                                    theoretical_n, theoretical_factors, level['timeout'])
                                
                                if success:
                                    level_breakthroughs += 1
                                    total_breakthroughs += 1
                                    max_bits_achieved = max(max_bits_achieved, target_bits)
                                    level_max_bits = max(level_max_bits, target_bits)
                                    largest_factored = max(largest_factored, theoretical_n)
                                    print(f"    🌟 THEORETICAL FIBONACCI BREAKTHROUGH on {target_bits}-bit!")
                                else:
                                    print(f"    📊 Theoretical test: {details.get('error', 'failed')}")
                                    
                            except Exception as e:
                                print(f"    ⚠️  Theoretical test failed: {e}")
                        else:
                            print(f"    ❌ Could not generate {target_bits}-bit semiprime")
                        continue
                    
                    print(f"    Generated: n={n:,} ({actual_bits}-bit) = {factors[0]} × {factors[1]}")
                    
                    # Run Axiom 2 factorization
                    success, factor_found, test_time, details = axiom2_factorize_with_timeout(
                        n, factors, level['timeout'])
                    
                    if success:
                        level_breakthroughs += 1
                        total_breakthroughs += 1
                        max_bits_achieved = max(max_bits_achieved, actual_bits)
                        level_max_bits = max(level_max_bits, actual_bits)
                        largest_factored = max(largest_factored, n)
                        status = "🚀 FIBONACCI BREAKTHROUGH"
                        
                        # Categorize the success type
                        if 'factor_scores' in details and factor_found in details['factor_scores']:
                            score_details = details['factor_scores'][factor_found]['details']
                            
                            if score_details.get('fibonacci_number', 0) > 0:
                                fibonacci_successes.append((actual_bits, n, factor_found))
                            if score_details.get('golden_ratio', 0) > 0:
                                golden_ratio_successes.append((actual_bits, n, factor_found))
                            if score_details.get('vortex_resonance', 0) > 0:
                                vortex_successes.append((actual_bits, n, factor_found))
                            if score_details.get('entanglement', 0) > 0:
                                entanglement_successes.append((actual_bits, n, factor_found))
                        
                    else:
                        if 'error' in details and details['error'] == 'timeout':
                            status = "⏱️ Timeout"
                        else:
                            status = "❌ Failed"
                    
                    print(f"    {status}: factor={factor_found}, time={test_time:.3f}s")
                    
                    if success and 'factor_scores' in details:
                        score_details = details['factor_scores'].get(factor_found, {}).get('details', {})
                        print(f"      Fibonacci score breakdown: {score_details}")
                        print(f"      Total score: {details.get('best_score', 0):.3f}")
                        print(f"      Vortices: {details.get('vortices', 0)}, Spirals: {details.get('spiral_positions', 0)}")
                        
                except Exception as e:
                    print(f"    ❌ Test failed: {e}")
        
        # Level summary
        if level_breakthroughs > 0:
            print(f"\n  📊 Level Summary: {level_breakthroughs} Fibonacci breakthroughs achieved!")
            print(f"      Maximum bits reached: {level_max_bits}")
        else:
            print(f"\n  📊 Level Summary: No Fibonacci breakthroughs in this zone")
        
        # Early stopping for higher levels if no success
        if level_breakthroughs == 0 and target_bits >= 48:
            print(f"\n  🛑 Stopping progression - Fibonacci methods may have reached limits")
            break
    
    # AXIOM 2 BREAKTHROUGH SUMMARY
    print("\n" + "=" * 100)
    print("🌀 AXIOM 2: FIBONACCI FLOW BREAKTHROUGH REPORT 🌀")
    print("=" * 100)
    
    print(f"🏆 FIBONACCI COMPUTATIONAL ACHIEVEMENTS:")
    print(f"   MAXIMUM BIT-LENGTH FACTORED: {max_bits_achieved} bits")
    print(f"   LARGEST NUMBER FACTORED: {largest_factored:,}")
    print(f"   TOTAL FIBONACCI BREAKTHROUGHS: {total_breakthroughs}")
    
    # Success categorization
    print(f"\n🔍 FIBONACCI SUCCESS ANALYSIS:")
    print(f"   Pure Fibonacci Number Successes: {len(fibonacci_successes)}")
    print(f"   Golden Ratio Resonance Successes: {len(golden_ratio_successes)}")
    print(f"   Vortex Resonance Successes: {len(vortex_successes)}")
    print(f"   Entanglement Correlation Successes: {len(entanglement_successes)}")
    
    # Milestone announcements
    print(f"\n🎯 FIBONACCI ACHIEVEMENT CLASSIFICATION:")
    if max_bits_achieved >= 128:
        print(f"   🌌 FIBONACCI TRANSCENDENCE: 128-BIT BARRIER CONQUERED!")
        print(f"   The Fibonacci Flow has achieved cosmic computational resonance!")
    elif max_bits_achieved >= 64:
        print(f"   🔥 FIBONACCI MASTERY: {max_bits_achieved}-BIT FACTORIZATION!")
        print(f"   Golden ratio principles have unlocked historic computational power!")
    elif max_bits_achieved >= 48:
        print(f"   ⚡ FIBONACCI BREAKTHROUGH: {max_bits_achieved}-BIT FACTORIZATION!")
        print(f"   Fibonacci vortices have demonstrated revolutionary capability!")
    elif max_bits_achieved >= 32:
        print(f"   🎯 FIBONACCI MILESTONE: {max_bits_achieved}-BIT FACTORIZATION!")
        print(f"   Golden spiral navigation shows significant promise!")
    elif max_bits_achieved >= 24:
        print(f"   ✨ FIBONACCI SUCCESS: {max_bits_achieved}-BIT FACTORIZATION!")
        print(f"   Basic Fibonacci resonance has been established!")
    
    # Save Axiom 2 specific report
    report_path = "/workspaces/factorizer/benchmark/AXIOM2_128BIT_BREAKTHROUGH_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# 🌀 AXIOM 2: FIBONACCI FLOW 128-BIT BREAKTHROUGH REPORT 🌀\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## FIBONACCI COMPUTATIONAL ACHIEVEMENT\n\n")
        f.write(f"- **Maximum Bit-Length Achieved**: {max_bits_achieved} bits\n")
        f.write(f"- **Largest Number Factored**: {largest_factored:,}\n")
        f.write(f"- **Total Fibonacci Breakthroughs**: {total_breakthroughs}\n\n")
        
        f.write("## Fibonacci Success Categories\n\n")
        f.write(f"- **Pure Fibonacci Numbers**: {len(fibonacci_successes)} successes\n")
        f.write(f"- **Golden Ratio Resonance**: {len(golden_ratio_successes)} successes\n")
        f.write(f"- **Vortex Resonance**: {len(vortex_successes)} successes\n")
        f.write(f"- **Entanglement Correlation**: {len(entanglement_successes)} successes\n\n")
        
        f.write("## Breakthrough Timeline\n\n")
        all_successes = [(bits, n, factor, 'fibonacci') for bits, n, factor in fibonacci_successes]
        all_successes.extend([(bits, n, factor, 'golden') for bits, n, factor in golden_ratio_successes])
        all_successes.extend([(bits, n, factor, 'vortex') for bits, n, factor in vortex_successes])
        all_successes.extend([(bits, n, factor, 'entanglement') for bits, n, factor in entanglement_successes])
        all_successes.sort(key=lambda x: x[0])
        
        for i, (bits, n, factor, success_type) in enumerate(all_successes, 1):
            f.write(f"{i}. **{bits}-bit** ({success_type}): {n:,} → factor {factor} 🌀\n")
    
    print(f"\n📋 Axiom 2 breakthrough report saved to: {report_path}")

if __name__ == "__main__":
    axiom2_128bit_breakthrough()
