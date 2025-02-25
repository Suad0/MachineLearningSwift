//
//  StochasticCalculus.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 24.02.25.
//

import Foundation

// MARK: - MathExpression Extensions
extension MathExpression {
    func evaluate(with variables: [String: Double]) -> Double {
        switch self {
        case .variable(let name):
            return variables[name] ?? 0
        case .constant(let value):
            return value
        case .sum(let exprs):
            return exprs.reduce(0) { $0 + $1.evaluate(with: variables) }
        case .product(let exprs):
            return exprs.reduce(1) { $0 * $1.evaluate(with: variables) }
        case .power(let base, let exponent):
            return pow(base.evaluate(with: variables), exponent.evaluate(with: variables))
        case .ln(let expr):
            return log(expr.evaluate(with: variables))
        case .integral(_, _):
            // Implement numerical integration if needed
            return 0
        @unknown default:
            return 0
        }
    }
}

// MARK: - Stochastic Calculus Core
struct StochasticCalculus {
    // Improved Wiener process with proper normal distribution
    private static func generateDeltaW(deltaT: Double) -> Double {
        let u1 = Double.random(in: 0.0...1.0)
        let u2 = Double.random(in: 0.0...1.0)
        let z = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        return z * sqrt(deltaT)
    }
    
    func wienerProcess(steps: Int, deltaT: Double) -> [Double] {
        var W = [0.0]
        (1..<steps).forEach { _ in
            W.append(W.last! + Self.generateDeltaW(deltaT: deltaT))
        }
        return W
    }
    
    // Generalized Euler-Maruyama
    func solveSDE(steps: Int, deltaT: Double,
                  drift: (Double, Double) -> Double,
                  diffusion: (Double, Double) -> Double,
                  X0: Double) -> [Double] {
        var X = [X0]
        (1..<steps).forEach { i in
            let t = Double(i) * deltaT
            let dW = Self.generateDeltaW(deltaT: deltaT)
            X.append(X.last! + drift(X.last!, t) * deltaT + diffusion(X.last!, t) * dW)
        }
        return X
    }
}

// MARK: - Ito Calculus
struct ItoCalculus {
    static func lemma(f: MathExpression,
                    variable: String,
                    time: String,
                    driftTerm: MathExpression,
                    diffusionTerm: MathExpression) -> (drift: MathExpression, diffusion: MathExpression) {
        
        // Break down complex expression
        let dfdt = derivative(of: f, withRespectTo: time)
        let dfdx = derivative(of: f, withRespectTo: variable)
        let d2fdx2 = derivative(of: dfdx, withRespectTo: variable)
        
        let firstTerm = dfdt
        let secondTerm = driftTerm * dfdx
        let thirdTerm = MathExpression.constant(0.5) * (diffusionTerm * diffusionTerm) * d2fdx2
        
        return (
            drift: firstTerm + secondTerm + thirdTerm,
            diffusion: diffusionTerm * dfdx
        )
    }
}

// MARK: - Black-Scholes Model
struct BlackScholes {
    private static func normcdf(_ x: Double) -> Double {
        let a1 = 0.31938153
        let a2 = -0.356563782
        let a3 = 1.781477937
        let a4 = -1.821255978
        let a5 = 1.330274429
        let l = abs(x)
        let k = 1.0 / (1.0 + 0.2316419 * l)
        var w = 1.0 - 1.0 / sqrt(2 * .pi) * exp(-l * l / 2) * (a1 * k + a2 * k * k + a3 * pow(k, 3) + a4 * pow(k, 4) + a5 * pow(k, 5))
        if x < 0 { w = 1.0 - w }
        return w
    }
    
    static func callPrice(S: Double, K: Double, T: Double, r: Double, sigma: Double) -> Double {
        let d1 = (log(S/K) + (r + pow(sigma, 2)/2) * T)/(sigma * sqrt(T))
        let d2 = d1 - sigma * sqrt(T)
        return S * normcdf(d1) - K * exp(-r*T) * normcdf(d2)
    }
    
    static func putPrice(S: Double, K: Double, T: Double, r: Double, sigma: Double) -> Double {
        let d1 = (log(S/K) + (r + pow(sigma, 2)/2) * T)/(sigma * sqrt(T))
        let d2 = d1 - sigma * sqrt(T)
        return K * exp(-r*T) * normcdf(-d2) - S * normcdf(-d1)
    }
}

// MARK: - Risk Management
struct RiskManager {
    // Value-at-Risk implementations
    static func historicalVaR(returns: [Double], confidence: Double) -> Double {
        let sorted = returns.sorted()
        let index = Int((1 - confidence) * Double(sorted.count))
        return -sorted[index]
    }
    
    // Expected Shortfall/CVaR
    static func expectedShortfall(returns: [Double], confidence: Double) -> Double {
        let varLevel = historicalVaR(returns: returns, confidence: confidence)
        let tailLosses = returns.filter { $0 <= -varLevel }
        return -tailLosses.reduce(0, +) / Double(tailLosses.count)
    }
}


/*

// MARK: - Kalman Filter
class KalmanFilter {
    // State vector and covariance matrix
    var x: [Double]
    var P: [[Double]]
    
    init(initialState: [Double], initialCovariance: [[Double]]) {
        self.x = initialState
        self.P = initialCovariance
    }
    
    func predict(F: [[Double]], Q: [[Double]]) {
        x = matrixMultiply(F, x)
        P = matrixAdd(matrixMultiply(matrixMultiply(F, P), matrixTranspose(F)), Q)
    }
    
    func update(z: [Double], H: [[Double]], R: [[Double]]) {
        let y = matrixSubtract(z, matrixMultiply(H, x))
        let S = matrixAdd(matrixMultiply(matrixMultiply(H, P), matrixTranspose(H)), R)
        let K = matrixMultiply(matrixMultiply(P, matrixTranspose(H)), matrixInverse(S))
        x = matrixAdd(x, matrixMultiply(K, y))
        P = matrixMultiply(matrixSubtract(matrixIdentity(P.count), matrixMultiply(K, H)), P)
    }
    
    // Matrix operations (simplified 1D implementation)
    private func matrixMultiply(_ a: [[Double]], _ b: [Double]) -> [Double] {
        // Implementation for 1D case
    }
    
    private func matrixInverse(_ matrix: [[Double]]) -> [[Double]] {
        // Simplified inversion for 2x2
        
    }
    
}
 
 */

