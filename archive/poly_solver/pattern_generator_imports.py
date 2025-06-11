"""
Helper module to properly import pattern generator classes
"""

# Execute the pattern generator code to define all classes
exec(open('pattern-generator.py').read(), globals())

# Export the main classes
__all__ = ['PatternFactorizer', 'PatternEngine', 'PatternSignature', 'UniversalConstants']