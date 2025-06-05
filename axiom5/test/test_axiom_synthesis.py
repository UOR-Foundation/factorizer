"""
Tests for Axiom Synthesis functionality
Validates pattern fusion and hybrid method creation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.axiom_synthesis import (
    AxiomSynthesizer,
    pattern_fusion,
    create_hybrid_method,
    cross_axiom_resonance,
    emergent_pattern_detection,
    synthesize_from_failures
)

def test_axiom_synthesizer_init():
    """Test AxiomSynthesizer initialization"""
    n = 143  # 11 × 13
    synth = AxiomSynthesizer(n)
    
    assert synth.n == 143
    assert synth.root == 11
    assert len(synth.success_patterns) == 0
    assert len(synth.hybrid_methods) == 0
    
    print("✓ AxiomSynthesizer initialization")

def test_record_success():
    """Test success pattern recording"""
    n = 77
    synth = AxiomSynthesizer(n)
    
    # Record some successes
    synth.record_success(['axiom1', 'axiom3'], position=7, 
                        method_description="Prime-coherence combination")
    synth.record_success(['axiom2'], position=5,
                        method_description="Fibonacci method")
    
    assert len(synth.success_patterns) == 2
    assert synth.success_patterns[0]['position'] == 7
    assert 'axiom1' in synth.success_patterns[0]['axioms']
    
    print("✓ Success pattern recording")

def test_synthesize_method():
    """Test hybrid method synthesis"""
    n = 91  # 7 × 13
    synth = AxiomSynthesizer(n)
    
    # Create hybrid with weights
    weights = {
        'axiom1': 0.3,
        'axiom2': 0.2,
        'axiom3': 0.4,
        'axiom4': 0.1
    }
    
    hybrid = synth.synthesize_method(weights)
    
    assert callable(hybrid)
    assert len(synth.hybrid_methods) == 1
    
    # Test evaluation
    score = hybrid(7)  # Should score well for factor
    assert isinstance(score, float)
    assert score >= 0
    
    print("✓ Hybrid method synthesis")

def test_learn_weights():
    """Test weight learning from patterns"""
    n = 55
    synth = AxiomSynthesizer(n)
    
    # Add success patterns
    synth.record_success(['axiom1', 'axiom3'], 5)
    synth.record_success(['axiom1'], 11)
    synth.record_success(['axiom3', 'axiom4'], 7)
    
    weights = synth.learn_weights()
    
    assert isinstance(weights, dict)
    assert 'axiom1' in weights
    assert 'axiom3' in weights
    
    # Weights should sum to 1
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01
    
    # axiom1 should have high weight (appears often)
    assert weights['axiom1'] > 0.2
    
    print("✓ Weight learning")

def test_pattern_fusion():
    """Test pattern fusion"""
    n = 143
    patterns = [
        {'position': 5, 'axioms': ['axiom1']},
        {'position': 7, 'axioms': ['axiom2']},
        {'position': 11, 'axioms': ['axiom3']}
    ]
    
    fused = pattern_fusion(patterns, n)
    
    assert isinstance(fused, list)
    assert len(fused) > len(patterns)  # Should generate combinations
    
    # Should include original positions
    for p in patterns:
        assert p['position'] in fused
    
    # All should be valid
    for pos in fused:
        assert 2 <= pos <= 11
    
    print("✓ Pattern fusion")

def test_create_hybrid_method_func():
    """Test create_hybrid_method function"""
    n = 77
    combinations = [
        ('axiom1', 'axiom3'),
        ('axiom2', 'axiom3'),
        ('axiom1', 'axiom4')
    ]
    
    hybrid = create_hybrid_method(combinations, n)
    
    assert callable(hybrid)
    
    # Test evaluation
    score = hybrid(7)
    assert isinstance(score, float)
    
    print("✓ Create hybrid method function")

def test_cross_axiom_resonance():
    """Test cross-axiom resonance detection"""
    n = 91
    axiom_results = {
        'axiom1': [7, 9, 11],
        'axiom2': [5, 7, 8],
        'axiom3': [6, 7, 10],
        'axiom4': [7, 12]
    }
    
    resonant = cross_axiom_resonance(n, axiom_results)
    
    assert isinstance(resonant, list)
    
    # Position 7 appears in all axioms, should be top
    if resonant:
        assert 7 in resonant[:2]  # Should be near top
    
    print("✓ Cross-axiom resonance")

def test_emergent_pattern_detection():
    """Test emergent pattern detection"""
    n = 143
    
    # Create observation history
    history = [
        {'axiom': 'axiom1', 'position': 11},
        {'axiom': 'axiom1', 'position': 13},
        {'axiom': 'axiom2', 'position': 8},
        {'axiom': 'axiom2', 'position': 13},
        {'axiom': 'axiom3', 'position': 5},
        {'axiom': 'axiom3', 'position': 10},
        {'axiom': 'axiom4', 'position': 7},
        {'axiom': 'axiom4', 'position': 7},
        {'axiom': 'axiom4', 'position': 7}
    ]
    
    emergent = emergent_pattern_detection(n, history)
    
    assert isinstance(emergent, list)
    
    # Should detect axiom4 fixed point at 7
    assert 7 in emergent
    
    print("✓ Emergent pattern detection")

def test_synthesize_from_failures():
    """Test synthesis from failure patterns"""
    n = 77
    failed_positions = [4, 6, 8, 10, 20, 22, 24]  # Many evens
    
    candidates = synthesize_from_failures(n, failed_positions)
    
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    
    # Should avoid failed positions
    for cand in candidates:
        assert cand not in failed_positions
    
    # Should include some odd positions (since evens failed)
    odd_count = sum(1 for c in candidates if c % 2 == 1)
    assert odd_count > len(candidates) // 2
    
    print("✓ Synthesis from failures")

def test_synthesis_determinism():
    """Test that synthesis is deterministic"""
    n = 143
    
    # Create two synthesizers
    synth1 = AxiomSynthesizer(n)
    synth2 = AxiomSynthesizer(n)
    
    # Same patterns
    for synth in [synth1, synth2]:
        synth.record_success(['axiom1'], 11)
        synth.record_success(['axiom2', 'axiom3'], 13)
    
    # Learn weights
    weights1 = synth1.learn_weights()
    weights2 = synth2.learn_weights()
    
    assert weights1 == weights2
    
    print("✓ Synthesis determinism")

def run_all_tests():
    """Run all axiom synthesis tests"""
    print("Testing Axiom Synthesis (Axiom 5)...")
    print("-" * 40)
    
    test_axiom_synthesizer_init()
    test_record_success()
    test_synthesize_method()
    test_learn_weights()
    test_pattern_fusion()
    test_create_hybrid_method_func()
    test_cross_axiom_resonance()
    test_emergent_pattern_detection()
    test_synthesize_from_failures()
    test_synthesis_determinism()
    
    print("-" * 40)
    print("All Axiom Synthesis tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
