//
//  Analysis.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.02.25.
//

import Foundation


indirect enum MathExpression: CustomStringConvertible {
    case variable(String)
    case constant(Double)
    case sum([MathExpression])
    case product([MathExpression])
    case power(base: MathExpression, exponent: MathExpression)
    case sin(MathExpression)
    case cos(MathExpression)
    case ln(MathExpression)
    case integral(MathExpression, withRespectTo: String)
    
    var description: String {
        switch self {
        case .variable(let name):
            return name
        case .constant(let value):
            return String(format: "%.2f", value)
        case .sum(let exprs):
            return "(\(exprs.map { $0.description }.joined(separator: " + ")))"
        case .product(let exprs):
            return "(\(exprs.map { $0.description }.joined(separator: " * ")))"
        case .power(let base, let exponent):
            return "(\(base.description)^\(exponent.description))"
        case .sin(let expr):
            return "sin(\(expr.description))"
        case .cos(let expr):
            return "cos(\(expr.description))"
        case .ln(let expr):
            return "ln(\(expr.description))"
        case .integral(let expr, let variable):
            return "∫\(expr.description)d\(variable)"
        }
    }
    
    
}

// MARK: - Differentiation
func derivative(of expr: MathExpression, withRespectTo variable: String) -> MathExpression {
    switch expr {
    case .variable(let name):
        return name == variable ? .constant(1) : .constant(0)
        
    case .constant:
        return .constant(0)
        
    case .sum(let exprs):
        return .sum(exprs.map { derivative(of: $0, withRespectTo: variable) })
        
    case .product(let exprs):
        var terms = [MathExpression]()
        for i in exprs.indices {
            var factors = exprs
            factors[i] = derivative(of: factors[i], withRespectTo: variable)
            terms.append(.product(factors))
        }
        return .sum(terms)
        
    case .power(let base, let exponent):
        return handlePowerRule(base: base, exponent: exponent, variable: variable)
        
    case .sin(let inner):
        return .product([
            .cos(inner),
            derivative(of: inner, withRespectTo: variable)
        ])
        
    case .cos(let inner):
        return .product([
            .constant(-1),
            .sin(inner),
            derivative(of: inner, withRespectTo: variable)
        ])
        
    default:
        return .constant(0) // Handle other cases as needed
    }
}

private func handlePowerRule(base: MathExpression, exponent: MathExpression, variable: String) -> MathExpression {
    let dBase = derivative(of: base, withRespectTo: variable)
    
    // Power rule: d/dx [u^n] = n*u^(n-1)*du/dx
    return .product([
        exponent,
        .power(base: base, exponent: simplified(.sum([exponent, .constant(-1)]))),
        dBase
    ])
}

// MARK: - Integration
func integral(of expr: MathExpression, withRespectTo variable: String) -> MathExpression {
    switch expr {
    case .variable(let name) where name == variable:
        return .power(base: .variable(variable), exponent: .constant(2)) / .constant(2)
        
    case .constant(let c):
        return .product([.constant(c), .variable(variable)])
        
    case .sum(let exprs):
        return .sum(exprs.map { integral(of: $0, withRespectTo: variable) })
        
    case .power(let base, let exponent) where isVariable(base, variable):
        if case .constant(let n) = exponent {
            let newExponent = n + 1
            if newExponent != 0 {
                return (.power(base: base, exponent: .constant(newExponent)) / .constant(newExponent))
            }
        }
        return .integral(expr, withRespectTo: variable)
        
    default:
        return .integral(expr, withRespectTo: variable)
    }
    
    
    
}

// MARK: - Helper Functions
func isVariable(_ expr: MathExpression, _ variable: String) -> Bool {
    if case .variable(let name) = expr {
        return name == variable
    }
    return false
}

func simplified(_ expr: MathExpression) -> MathExpression {
    // Implement simplification rules here
    return expr
}

// MARK: - Operators
func + (lhs: MathExpression, rhs: MathExpression) -> MathExpression {
    if case .sum(let exprs) = lhs {
        return .sum(exprs + [rhs])
    }
    return .sum([lhs, rhs])
}

func * (lhs: MathExpression, rhs: MathExpression) -> MathExpression {
    if case .product(let exprs) = lhs {
        return .product(exprs + [rhs])
    }
    return .product([lhs, rhs])
}

func / (lhs: MathExpression, rhs: MathExpression) -> MathExpression {
    .product([lhs, .power(base: rhs, exponent: .constant(-1))])
}
