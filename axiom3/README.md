# Axiom 3: Duality Principle

## Foundation
The Duality Principle recognizes that numbers exist simultaneously as waves (spectral patterns) and particles (discrete factors). This wave-particle duality reveals factor relationships through spectral analysis and coherence measurements.

## Core Principle
**"Every number has a spectral signature that resonates with its factors"**

## Implementation Components

### 1. Spectral Core (`spectral_core.py`)
- **Binary Spectrum**: Bit pattern analysis and autocorrelation
- **Modular Spectrum**: Residue patterns across prime moduli
- **Digital Spectrum**: Digit sum and digital root analysis
- **Harmonic Spectrum**: Golden ratio phase relationships

### 2. Coherence (`coherence.py`)
- **Coherence Calculation**: Measure spectral alignment between numbers
- **Coherence Caching**: Efficient storage of spectral calculations
- **Triple Coherence**: How well a×b matches n spectrally

### 3. Fold Topology (`fold_topology.py`)
- **Fold Energy**: Spectral distance between x, n/x, and n
- **Sharp Folds**: Local minima in fold energy landscape
- **Topological Navigation**: Connected components in energy space

### 4. Interference (`interference.py`)
- **Prime×Fibonacci Waves**: Interference patterns from dual sources
- **Extrema Detection**: Peaks and valleys in interference
- **Resonance Source**: Identify which prime/fib pair creates patterns

## Mathematical Foundation
- Spectral vector S(n) combines multiple spectral views
- Coherence C(a,b,n) = exp(-||S(a)+S(b)-2S(n)||²)
- Fold energy E(x) = ||S(x) + S(n/x) - S(n)||²
- Interference I(x) = Σ cos(2πpx/n) × Σ cos(2πfx/nφ)

## Key Algorithms

### Spectral Vector Construction
```
S(n) = [binary_spectrum(n), modular_spectrum(n), 
        digital_spectrum(n), harmonic_spectrum(n)]
```

### Sharp Fold Detection
```
curvature = E[i-1] - 2×E[i] + E[i+1]
candidates = positions with minimum curvature
```

### Interference Pattern
```
prime_wave = Σ cos(2π × prime × x / n)
fib_wave = Σ cos(2π × fib × x / (n×φ))
interference = prime_wave × fib_wave
```

## Axiom Compliance
- **NO FALLBACKS**: Pure spectral mathematics
- **NO RANDOMIZATION**: Deterministic spectral analysis
- **NO SIMPLIFICATION**: Full spectrum computation
- **NO HARDCODING**: All spectra derived mathematically

## Integration with Other Axioms
- Uses Axiom 1's primes for modular spectrum
- Incorporates Axiom 2's golden ratio in harmonic spectrum
- Provides coherence fields for Axiom 4's observer
- Axiom 5 creates meta-spectra from spectral patterns
