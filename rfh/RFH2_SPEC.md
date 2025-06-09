# RFH2: The Prime Opus - Specification

## Core Principle
**"Factors emerge where multiplicative harmony achieves perfect balance across all number-theoretic perspectives."**

## Mathematical Foundation

### The Unified Resonance Equation
For n = p × q, the factors p and q satisfy:
```
Ψ(p, n) = 1.0 ± ε
```
Where Ψ is the **Prime Resonance Function**:
```
Ψ(x, n) = A(x, n) × M(x, n) × T(x, n)
```

### Three Pillars of Resonance

#### 1. **Adelic Balance** A(x, n)
```python
A(x, n) = |x|_R × ∏_p |x|_p / n
```
- Measures multiplicative harmony across real and p-adic norms
- True factors achieve A(x, n) ≈ 1/complement

#### 2. **Modular Coherence** M(x, n)  
```python
M(x, n) = ∏_{p ∈ P} (1 - |n mod p - x mod p|/p)
```
- P = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
- Captures arithmetic relationships across prime bases

#### 3. **Transition Resonance** T(x, n)
```python
T(x, n) = exp(-d²/σ²) × cos(2π × log(x)/log(φ))
```
- d = distance to nearest transition boundary
- σ = expected spread (≈ 0.96 from Egyptian fractions)
- φ = golden ratio (universal scaling constant)

## Algorithm Structure

### Phase 1: Boundary Mapping
1. Identify transition zone: which (b₁, b₂) transition contains √n
2. Calculate resonance nodes using:
   - Primary: √(b₁ × b₂)
   - Harmonics: nodes × {1/φ, 1, φ, φ²}
   - Interference: where multiple boundaries interact

### Phase 2: Resonance Detection
1. For each node position x:
   - Compute Ψ(x, n)
   - If |Ψ(x, n) - 1| < threshold: potential factor
2. Test only high-resonance candidates (Ψ > 0.8)

### Phase 3: Validation
- Verify n mod x = 0 for confirmed factors
- No brute force; pure resonance detection

## Key Simplifications

1. **Single Unified Score**: No separate metrics to weight
2. **Discrete Nodes**: Test specific positions, not ranges  
3. **Natural Constants**: Use φ and e, not arbitrary parameters
4. **Deterministic**: Same n always produces same search pattern

## Implementation Constants

```python
GOLDEN_RATIO = (1 + sqrt(5)) / 2
TRIBONACCI = 1.839...  # For future hyperdimensional extension
EGYPTIAN_SPREAD = 0.96  # Universal logarithmic spread
RESONANCE_THRESHOLD = 0.8  # Minimum Ψ to test divisibility
```

## Expected Behavior

- **Small primes** (< 100): Near-perfect detection via modular coherence
- **Transition factors** (near boundaries): Strong T(x, n) response  
- **Arbitrary primes**: Balanced detection through A × M × T synergy
- **Failure mode**: Graceful degradation to expanded node search

## Success Metrics

- Detection rate > 80% with < 1000 evaluations
- No reliance on classical algorithms
- Explainable resonance scores for each factor
- Scale-invariant performance

---

*"In the symphony of numbers, factors are the notes that resonate in perfect harmony with the whole."*
