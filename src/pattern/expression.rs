//! Mathematical expression engine for pattern formalization
//!
//! This module provides symbolic representation and manipulation
//! of patterns discovered through empirical observation.

use crate::types::{Number, Pattern, PatternKind};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Symbolic expression representing a mathematical relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Constant value
    Constant(f64),

    /// Variable (e.g., n, p, q)
    Variable(String),

    /// Binary operation
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    /// Unary operation
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },

    /// Function application
    Function { name: String, args: Vec<Expression> },

    /// Pattern reference
    PatternRef {
        pattern_id: String,
        component: String,
    },
}

/// Binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
}

/// Unary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnaryOperator {
    Negate,
    Sqrt,
    Log,
    Exp,
    Sin,
    Cos,
}

/// Pattern constraint representing a mathematical relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConstraint {
    /// Unique identifier
    pub id: String,

    /// Left-hand side of the constraint
    pub lhs: Expression,

    /// Constraint relation
    pub relation: ConstraintRelation,

    /// Right-hand side of the constraint
    pub rhs: Expression,

    /// Confidence in this constraint
    pub confidence: f64,
}

/// Constraint relations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintRelation {
    Equal,
    LessThan,
    GreaterThan,
    LessEqual,
    GreaterEqual,
    Approximately(f64), // tolerance
}

impl Expression {
    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        Expression::Constant(value)
    }

    /// Create a variable expression
    pub fn var(name: impl Into<String>) -> Self {
        Expression::Variable(name.into())
    }

    /// Create a binary operation
    pub fn binary(op: BinaryOperator, left: Expression, right: Expression) -> Self {
        Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create addition
    pub fn add(left: Expression, right: Expression) -> Self {
        Self::binary(BinaryOperator::Add, left, right)
    }

    /// Create multiplication
    pub fn mul(left: Expression, right: Expression) -> Self {
        Self::binary(BinaryOperator::Multiply, left, right)
    }

    /// Create square root
    pub fn sqrt(expr: Expression) -> Self {
        Expression::UnaryOp {
            op: UnaryOperator::Sqrt,
            operand: Box::new(expr),
        }
    }

    /// Evaluate the expression with given variable bindings
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Expression::Constant(val) => Ok(*val),

            Expression::Variable(name) => bindings.get(name).copied().ok_or_else(|| {
                crate::error::PatternError::FormalizationError(format!(
                    "Undefined variable: {}",
                    name
                ))
            }),

            Expression::BinaryOp { op, left, right } => {
                let left_val = left.evaluate(bindings)?;
                let right_val = right.evaluate(bindings)?;

                match op {
                    BinaryOperator::Add => Ok(left_val + right_val),
                    BinaryOperator::Subtract => Ok(left_val - right_val),
                    BinaryOperator::Multiply => Ok(left_val * right_val),
                    BinaryOperator::Divide => {
                        if right_val.abs() < 1e-10 {
                            Err(crate::error::PatternError::ArithmeticError(
                                "Division by zero".to_string(),
                            ))
                        } else {
                            Ok(left_val / right_val)
                        }
                    },
                    BinaryOperator::Power => Ok(left_val.powf(right_val)),
                    BinaryOperator::Modulo => Ok(left_val % right_val),
                }
            },

            Expression::UnaryOp { op, operand } => {
                let val = operand.evaluate(bindings)?;

                match op {
                    UnaryOperator::Negate => Ok(-val),
                    UnaryOperator::Sqrt => {
                        if val < 0.0 {
                            Err(crate::error::PatternError::ArithmeticError(
                                "Square root of negative number".to_string(),
                            ))
                        } else {
                            Ok(val.sqrt())
                        }
                    },
                    UnaryOperator::Log => {
                        if val <= 0.0 {
                            Err(crate::error::PatternError::ArithmeticError(
                                "Logarithm of non-positive number".to_string(),
                            ))
                        } else {
                            Ok(val.ln())
                        }
                    },
                    UnaryOperator::Exp => Ok(val.exp()),
                    UnaryOperator::Sin => Ok(val.sin()),
                    UnaryOperator::Cos => Ok(val.cos()),
                }
            },

            Expression::Function { name, args } => {
                let arg_vals: Result<Vec<f64>> =
                    args.iter().map(|arg| arg.evaluate(bindings)).collect();
                let arg_vals = arg_vals?;

                // Evaluate built-in functions
                match name.as_str() {
                    "min" => Ok(arg_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
                    "max" => Ok(arg_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
                    "abs" => {
                        if arg_vals.len() == 1 {
                            Ok(arg_vals[0].abs())
                        } else {
                            Err(crate::error::PatternError::FormalizationError(
                                "abs expects 1 argument".to_string(),
                            ))
                        }
                    },
                    _ => Err(crate::error::PatternError::FormalizationError(format!(
                        "Unknown function: {}",
                        name
                    ))),
                }
            },

            Expression::PatternRef { .. } => Err(crate::error::PatternError::FormalizationError(
                "Pattern references require pattern context".to_string(),
            )),
        }
    }

    /// Simplify the expression
    pub fn simplify(&self) -> Expression {
        match self {
            Expression::BinaryOp { op, left, right } => {
                let left_simp = left.simplify();
                let right_simp = right.simplify();

                // Constant folding
                if let (Expression::Constant(l), Expression::Constant(r)) =
                    (&left_simp, &right_simp)
                {
                    match op {
                        BinaryOperator::Add => return Expression::Constant(l + r),
                        BinaryOperator::Subtract => return Expression::Constant(l - r),
                        BinaryOperator::Multiply => return Expression::Constant(l * r),
                        BinaryOperator::Divide if r.abs() > 1e-10 => {
                            return Expression::Constant(l / r)
                        },
                        BinaryOperator::Power => return Expression::Constant(l.powf(*r)),
                        _ => {},
                    }
                }

                // Identity simplifications
                match (op, &left_simp, &right_simp) {
                    (BinaryOperator::Add, expr, Expression::Constant(0.0))
                    | (BinaryOperator::Add, Expression::Constant(0.0), expr) => expr.clone(),

                    (BinaryOperator::Multiply, expr, Expression::Constant(1.0))
                    | (BinaryOperator::Multiply, Expression::Constant(1.0), expr) => expr.clone(),

                    (BinaryOperator::Multiply, _, Expression::Constant(0.0))
                    | (BinaryOperator::Multiply, Expression::Constant(0.0), _) => {
                        Expression::Constant(0.0)
                    },

                    _ => Expression::binary(op.clone(), left_simp, right_simp),
                }
            },

            Expression::UnaryOp { op, operand } => {
                let operand_simp = operand.simplify();

                // Constant folding
                if let Expression::Constant(val) = &operand_simp {
                    match op {
                        UnaryOperator::Negate => return Expression::Constant(-val),
                        UnaryOperator::Sqrt if *val >= 0.0 => {
                            return Expression::Constant(val.sqrt())
                        },
                        UnaryOperator::Log if *val > 0.0 => return Expression::Constant(val.ln()),
                        UnaryOperator::Exp => return Expression::Constant(val.exp()),
                        UnaryOperator::Sin => return Expression::Constant(val.sin()),
                        UnaryOperator::Cos => return Expression::Constant(val.cos()),
                        _ => {},
                    }
                }

                Expression::UnaryOp {
                    op: op.clone(),
                    operand: Box::new(operand_simp),
                }
            },

            _ => self.clone(),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Constant(val) => write!(f, "{}", val),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::BinaryOp { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            },
            Expression::UnaryOp { op, operand } => {
                write!(f, "{}({})", op, operand)
            },
            Expression::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            },
            Expression::PatternRef {
                pattern_id,
                component,
            } => {
                write!(f, "{}::{}", pattern_id, component)
            },
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Power => write!(f, "^"),
            BinaryOperator::Modulo => write!(f, "%"),
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOperator::Negate => write!(f, "-"),
            UnaryOperator::Sqrt => write!(f, "sqrt"),
            UnaryOperator::Log => write!(f, "log"),
            UnaryOperator::Exp => write!(f, "exp"),
            UnaryOperator::Sin => write!(f, "sin"),
            UnaryOperator::Cos => write!(f, "cos"),
        }
    }
}

/// Generate mathematical expression from pattern
pub fn pattern_to_expression(pattern: &Pattern) -> Expression {
    match &pattern.kind {
        PatternKind::Power { base, exponent } => Expression::binary(
            BinaryOperator::Power,
            Expression::Constant(*base as f64),
            Expression::Constant(*exponent as f64),
        ),

        PatternKind::Mersenne { p, .. } => Expression::binary(
            BinaryOperator::Subtract,
            Expression::binary(
                BinaryOperator::Power,
                Expression::Constant(2.0),
                Expression::Constant(*p as f64),
            ),
            Expression::Constant(1.0),
        ),

        PatternKind::Harmonic {
            base_frequency,
            harmonics,
        } => {
            let mut expr = Expression::Constant(0.0);
            for (i, &amplitude) in harmonics.iter().enumerate() {
                let harmonic = Expression::mul(
                    Expression::Constant(amplitude),
                    Expression::UnaryOp {
                        op: UnaryOperator::Sin,
                        operand: Box::new(Expression::mul(
                            Expression::Constant(
                                2.0 * std::f64::consts::PI * base_frequency * (i + 1) as f64,
                            ),
                            Expression::var("x"),
                        )),
                    },
                );
                expr = Expression::add(expr, harmonic);
            }
            expr
        },

        _ => {
            // Default expression for other pattern types
            Expression::Function {
                name: "pattern".to_string(),
                args: vec![
                    Expression::Constant(pattern.frequency),
                    Expression::var("n"),
                ],
            }
        },
    }
}

/// Create constraints from pattern observations
pub fn create_pattern_constraints(
    pattern: &Pattern,
    observations: &[(Number, Number, Number)],
) -> Vec<PatternConstraint> {
    let mut constraints = Vec::new();

    // Basic factorization constraint: p * q = n
    constraints.push(PatternConstraint {
        id: "factorization".to_string(),
        lhs: Expression::mul(Expression::var("p"), Expression::var("q")),
        relation: ConstraintRelation::Equal,
        rhs: Expression::var("n"),
        confidence: 1.0,
    });

    // Pattern-specific constraints
    match &pattern.kind {
        PatternKind::Emergent => {
            // For balanced semiprimes: |p - q| < sqrt(n) / 10
            constraints.push(PatternConstraint {
                id: "balanced_constraint".to_string(),
                lhs: Expression::Function {
                    name: "abs".to_string(),
                    args: vec![Expression::binary(
                        BinaryOperator::Subtract,
                        Expression::var("p"),
                        Expression::var("q"),
                    )],
                },
                relation: ConstraintRelation::LessThan,
                rhs: Expression::binary(
                    BinaryOperator::Divide,
                    Expression::sqrt(Expression::var("n")),
                    Expression::Constant(10.0),
                ),
                confidence: 0.9,
            });
        },

        PatternKind::Harmonic { base_frequency, .. } => {
            // For harmonic patterns: frequency relationship
            constraints.push(PatternConstraint {
                id: "harmonic_frequency".to_string(),
                lhs: Expression::binary(
                    BinaryOperator::Modulo,
                    Expression::var("n"),
                    Expression::Constant(base_frequency * 1000.0),
                ),
                relation: ConstraintRelation::Approximately(0.01),
                rhs: Expression::Constant(0.0),
                confidence: 0.8,
            });
        },

        _ => {},
    }

    // Add empirical constraints from observations
    if !observations.is_empty() {
        // Average offset ratio constraint
        let avg_offset_ratio = observations
            .iter()
            .map(|(n, p, q)| {
                let sqrt_n = n.to_f64().unwrap_or(1.0).sqrt();
                let avg = (p.to_f64().unwrap_or(0.0) + q.to_f64().unwrap_or(0.0)) / 2.0;
                (avg - sqrt_n) / sqrt_n
            })
            .sum::<f64>()
            / observations.len() as f64;

        constraints.push(PatternConstraint {
            id: "empirical_offset".to_string(),
            lhs: Expression::binary(
                BinaryOperator::Divide,
                Expression::binary(
                    BinaryOperator::Subtract,
                    Expression::binary(
                        BinaryOperator::Divide,
                        Expression::add(Expression::var("p"), Expression::var("q")),
                        Expression::Constant(2.0),
                    ),
                    Expression::sqrt(Expression::var("n")),
                ),
                Expression::sqrt(Expression::var("n")),
            ),
            relation: ConstraintRelation::Approximately(0.1),
            rhs: Expression::Constant(avg_offset_ratio),
            confidence: 0.7,
        });
    }

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_evaluation() {
        // Test: 2 * 3 + 4
        let expr = Expression::add(
            Expression::mul(Expression::Constant(2.0), Expression::Constant(3.0)),
            Expression::Constant(4.0),
        );

        let bindings = HashMap::new();
        assert_eq!(expr.evaluate(&bindings).unwrap(), 10.0);
    }

    #[test]
    fn test_expression_simplification() {
        // Test: x * 1 + 0
        let expr = Expression::add(
            Expression::mul(Expression::var("x"), Expression::Constant(1.0)),
            Expression::Constant(0.0),
        );

        let simplified = expr.simplify();
        match simplified {
            Expression::Variable(name) => assert_eq!(name, "x"),
            _ => panic!("Expected variable x"),
        }
    }

    #[test]
    fn test_pattern_constraint() {
        let constraint = PatternConstraint {
            id: "test".to_string(),
            lhs: Expression::var("p"),
            relation: ConstraintRelation::Equal,
            rhs: Expression::Constant(17.0),
            confidence: 1.0,
        };

        let mut bindings = HashMap::new();
        bindings.insert("p".to_string(), 17.0);

        assert_eq!(constraint.lhs.evaluate(&bindings).unwrap(), 17.0);
    }
}
