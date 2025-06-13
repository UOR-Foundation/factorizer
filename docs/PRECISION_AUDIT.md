# Precision Audit Report

## Critical Precision Loss Points

### 1. Direct Empirical Pattern (src/pattern/direct_empirical.rs)
**Critical Issues:**
- Line 233-234: `Number::from((p_scaled.round() as u128).max(1))` - **Caps at 128 bits!**
- Line 226: `n.to_f64().unwrap_or(1e100)` - Loses precision for numbers > 2^53
- Line 278-279: Converting to u128 for center calculations
- Line 283-284: Using u128 and i128 for factor candidates

**Impact**: This explains the 224-bit cutoff - any number requiring factors > 128 bits fails.

### 2. Pattern Adaptation Logic
- Uses f64 for all scaling operations
- Converts ratios through floating point
- Loses precision in sqrt calculations

### 3. Basis Module (src/pattern/basis.rs)
- Line 156: `n.to_f64().unwrap_or(1e100)` for large numbers
- Line 379: `as u128` conversion for pattern radius
- Multiple f64 conversions for scaling

### 4. Stream Processor (src/pattern/stream_processor.rs)
- Line 231: Converts to f64 for byte extraction (this is OK, stays under 256)
- Line 260, 322: Maps bytes to u64 (should be fine for individual bytes)
- Uses f64 for all resonance calculations

### 5. Expression Module (src/pattern/expression.rs)
- Line 503-504: Converts numbers to f64 for calculations
- Line 389-390: Stores constants as f64

## Summary of Issues

1. **Hard 128-bit limit**: Using `as u128` caps our factors at 128 bits
2. **Float precision loss**: Using f64 (53-bit mantissa) for large number operations
3. **Scaling through floats**: All pattern adaptation uses floating point
4. **Constants stored as f64**: Mathematical constants limited to 64-bit precision

## Files Needing Updates

### High Priority (Breaking at 224 bits):
- `/src/pattern/direct_empirical.rs` - Remove all u128 conversions
- `/src/pattern/empirical_pattern.rs` - Remove f64 scaling
- `/src/pattern/basis.rs` - Keep calculations in Number domain

### Medium Priority:
- `/src/pattern/expression.rs` - Store constants as Number
- `/src/pattern/formalization.rs` - Use rational arithmetic
- `/src/pattern/stream_processor.rs` - Exact resonance calculations

### Low Priority:
- `/src/optimization/arithmetic.rs` - Already has some Number operations
- Various test files using f64 for verification

The 224-bit limit is definitely due to the u128 conversions in pattern adaptation!