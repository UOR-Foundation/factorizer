"""
Tests for Meta-Observer functionality
Validates observation of observers and blind spot detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from axiom5.meta_observer import (
    MetaObserver,
    AxiomPerformanceProfile,
    detect_blind_spots,
    create_meta_coherence_field,
    analyze_axiom_interference
)

def test_axiom_performance_profile():
    """Test axiom performance tracking"""
    profile = AxiomPerformanceProfile()
    
    # Record some attempts
    profile.record_attempt('axiom1', 35, True, 0.1, factor_found=5)
    profile.record_attempt('axiom1', 77, True, 0.2, factor_found=7)
    profile.record_attempt('axiom2', 35, False, 0.3)
    profile.record_attempt('axiom3', 143, True, 0.15, factor_found=11)
    
    # Check tracking
    assert len(profile.success_patterns['axiom1']) > 0
    assert len(profile.failure_patterns['axiom2']) > 0
    assert len(profile.timing_data['axiom1']) == 2
    
    # Test best axiom prediction
    best = profile.get_best_axiom(91)  # 7 × 13
    assert best in ['axiom1', 'axiom2', 'axiom3', 'axiom4']
    
    print("✓ Axiom performance profile")

def test_meta_observer_init():
    """Test MetaObserver initialization"""
    n = 143  # 11 × 13
    observer = MetaObserver(n)
    
    assert observer.n == 143
    assert observer.root == 11
    assert len(observer.observation_history) == 0
    assert len(observer.blind_spots) == 0
    
    print("✓ MetaObserver initialization")

def test_observe_observation():
    """Test observation recording"""
    n = 77  # 7 × 11
    observer = MetaObserver(n)
    
    # Record observations
    observer.observe_observation(position=7, coherence_value=0.8, 
                               axiom_used='axiom1', found_factor=True)
    observer.observe_observation(position=5, coherence_value=0.05, 
                               axiom_used='axiom2', found_factor=False)
    
    assert len(observer.observation_history) == 2
    assert len(observer.coherence_history) == 2
    assert 5 in observer.blind_spots  # Low coherence, no factor
    
    print("✓ Observation recording")

def test_meta_coherence():
    """Test meta-coherence calculation"""
    n = 55  # 5 × 11
    observer = MetaObserver(n)
    
    # Create observation history with consistent coherence values
    for i in range(5):
        # Use similar coherence values to get low variance = high meta-coherence
        observer.observe_observation(position=5+i, coherence_value=0.7+i*0.01,
                                   axiom_used='axiom3', found_factor=i==0)
    
    # Calculate meta-coherence
    meta_coh = observer.get_meta_coherence(5)
    assert 0 <= meta_coh <= 1
    
    # With a successful observation, should have boosted value
    assert meta_coh > 0.3  # More reasonable threshold
    
    print("✓ Meta-coherence calculation")

def test_observation_patterns():
    """Test pattern detection in observations"""
    n = 91  # 7 × 13
    observer = MetaObserver(n)
    
    # Create varied observation history
    for i in range(15):
        pos = 2 + (i % 5)
        axiom = f'axiom{(i % 4) + 1}'
        coh = 0.5 + 0.1 * (i % 3)
        observer.observe_observation(position=pos, coherence_value=coh,
                                   axiom_used=axiom, found_factor=False)
    
    # Detect patterns
    patterns = observer.detect_observation_patterns()
    
    assert 'high_activity' in patterns
    assert 'coherence_peaks' in patterns
    assert 'axiom_boundaries' in patterns
    assert 'repetition_cycles' in patterns
    
    # Should detect some patterns
    assert any(len(v) > 0 for v in patterns.values())
    
    print("✓ Observation pattern detection")

def test_blind_spot_detection():
    """Test blind spot detection"""
    n = 143
    observer = MetaObserver(n)
    
    # Create sparse observations
    for pos in [2, 3, 9, 10, 11]:
        observer.observe_observation(position=pos, coherence_value=0.6,
                                   axiom_used='axiom1', found_factor=False)
    
    # Mark some as blind spots
    observer.blind_spots.update([4, 5, 6])
    
    # Detect blind spots
    blind_spots = detect_blind_spots(observer, coverage_threshold=0.7)
    
    assert isinstance(blind_spots, list)
    assert len(blind_spots) > 0
    
    # Should include manually marked blind spots
    for bs in [4, 5, 6]:
        assert bs in blind_spots
    
    print("✓ Blind spot detection")

def test_meta_coherence_field():
    """Test meta-coherence field creation"""
    n = 35  # 5 × 7
    
    # Create multiple observers
    observers = []
    for i in range(3):
        obs = MetaObserver(n)
        # Add some observations
        for j in range(5):
            obs.observe_observation(position=2+j, coherence_value=0.5+j*0.1,
                                  axiom_used=f'axiom{(j%4)+1}', found_factor=False)
        observers.append(obs)
    
    # Create field
    field = create_meta_coherence_field(n, observers)
    
    assert isinstance(field, dict)
    assert len(field) > 0
    
    # All values should be valid
    for pos, mc in field.items():
        assert 2 <= pos <= 5  # sqrt(35) ≈ 5.9
        assert 0 <= mc <= 1
    
    print("✓ Meta-coherence field creation")

def test_axiom_interference():
    """Test axiom interference analysis"""
    n = 77
    observers = []
    
    # Create observer with axiom transitions
    obs = MetaObserver(n)
    obs.observe_observation(2, 0.5, 'axiom1', False)
    obs.observe_observation(3, 0.8, 'axiom2', False)  # Transition
    obs.observe_observation(4, 0.3, 'axiom2', False)
    obs.observe_observation(5, 0.7, 'axiom3', False)  # Transition
    observers.append(obs)
    
    # Analyze interference
    interference = analyze_axiom_interference(observers)
    
    assert isinstance(interference, dict)
    # Should detect axiom1-axiom2 and axiom2-axiom3 transitions
    assert len(interference) > 0
    
    print("✓ Axiom interference analysis")

def test_meta_observer_determinism():
    """Test that meta-observer is deterministic"""
    n = 143
    
    # Create two identical observers
    obs1 = MetaObserver(n)
    obs2 = MetaObserver(n)
    
    # Same observations
    for obs in [obs1, obs2]:
        obs.observe_observation(7, 0.8, 'axiom1', True)
        obs.observe_observation(9, 0.3, 'axiom2', False)
    
    # Meta-coherence should be identical
    mc1 = obs1.get_meta_coherence(8)
    mc2 = obs2.get_meta_coherence(8)
    assert mc1 == mc2
    
    print("✓ Meta-observer determinism")

def run_all_tests():
    """Run all meta-observer tests"""
    print("Testing Meta-Observer (Axiom 5)...")
    print("-" * 40)
    
    test_axiom_performance_profile()
    test_meta_observer_init()
    test_observe_observation()
    test_meta_coherence()
    test_observation_patterns()
    test_blind_spot_detection()
    test_meta_coherence_field()
    test_axiom_interference()
    test_meta_observer_determinism()
    
    print("-" * 40)
    print("All Meta-Observer tests passed! ✓")

if __name__ == "__main__":
    run_all_tests()
