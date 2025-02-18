//
//  DifferentialEquation.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.02.25.
//

import Foundation
import Numerics


extension Complex where RealType == Double {
    // Complex-Double operations
    static func * (lhs: Complex, rhs: Double) -> Complex {
        Complex(lhs.real * rhs, lhs.imaginary * rhs)
    }
    
    static func * (lhs: Double, rhs: Complex) -> Complex {
        Complex(lhs * rhs.real, lhs * rhs.imaginary)
    }
    
    // Complex-Int operations
    static func * (lhs: Int, rhs: Complex) -> Complex {
        Double(lhs) * rhs
    }
    
    static func * (lhs: Complex, rhs: Int) -> Complex {
        lhs * Double(rhs)
    }
    
    // Complex-Complex operations
    static func * (lhs: Complex, rhs: Complex) -> Complex {
        Complex(
            lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
            lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
        )
    }
}


// MARK: - ODE Solver Core
struct ODESolver {
    
    // MARK: - ODE Methods
    static func eulerMethod(x0: Double, y0: Double, h: Double, steps: Int, f: (Double, Double) -> Double) {
        var x = x0
        var y = y0
        print("\nEuler's Method:")
        for i in 0..<steps {
            let slope = f(x, y)
            y += h * slope
            x += h
            print(String(format: "Step %2d: x = %6.4f, y = %8.6f", i+1, x, y))
        }
    }
    
    static func rungeKutta4(x0: Double, y0: Double, h: Double, steps: Int, f: (Double, Double) -> Double) {
        var x = x0
        var y = y0
        print("\nRunge-Kutta 4th Order:")
        for i in 0..<steps {
            let k1 = h * f(x, y)
            let k2 = h * f(x + h/2, y + k1/2)
            let k3 = h * f(x + h/2, y + k2/2)
            let k4 = h * f(x + h, y + k3)
            
            y += (k1 + 2*k2 + 2*k3 + k4)/6
            x += h
            print(String(format: "Step %2d: x = %6.4f, y = %8.6f", i+1, x, y))
        }
    }
    
    // MARK: - Adaptive RKF45
    static func rkf45(x0: Double, y0: Double, h: Double, tolerance: Double, maxSteps: Int, f: (Double, Double) -> Double) {
        var x = x0
        var y = y0
        var currentH = h
        print("\nAdaptive RKF45:")
        
        for i in 0..<maxSteps {
            let k1 = currentH * f(x, y)
            let k2 = currentH * f(x + currentH/4, y + k1/4)
            let k3 = currentH * f(x + 3*currentH/8, y + 3*k1/32 + 9*k2/32)
            let k4 = currentH * f(x + 12*currentH/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            let k5 = currentH * f(x + currentH, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
            let k6 = currentH * f(x + currentH/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            
            // Error estimate
            let error = abs(k1/360 - 128*k3/4275 - 2197*k4/75240 + k5/50 + 2*k6/55)
            
            // Adaptive step size control
            if error > tolerance {
                currentH /= 2
                continue
            }
            
            y += 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
            x += currentH
            
            if error < tolerance/10 {
                currentH *= 2
            }
            
            print(String(format: "Step %2d: x = %6.4f, y = %8.6f, h = %6.4f", i+1, x, y, currentH))
        }
    }
    
    // MARK: - Numerical Integration
    static func simpsonsRule(a: Double, b: Double, n: Int, f: (Double) -> Double) throws -> Double {
        guard n > 0 && n % 2 == 0 else {
            throw NSError(domain: "ODESolver", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Number of intervals must be even and positive"])
        }
        
        let h = (b - a) / Double(n)
        var sum = f(a) + f(b)
        
        for i in 1..<n {
            let x = a + Double(i) * h
            sum += (i % 2 == 0) ? 2 * f(x) : 4 * f(x)
        }
        
        return sum * h / 3
    }
    
    // MARK: - Inverse Laplace Approximation
    static func inverseLaplace(t: Double, gamma: Double, n: Int, F: (Complex<Double>) -> Complex<Double>) throws -> Double {
        let uMin = -100.0
        let uMax = 100.0
        let h = (uMax - uMin) / Double(n)
        var sum = Complex<Double>.zero
        
        for i in 0...n {
            let u = uMin + Double(i) * h
            let s = Complex(gamma, u)
            let exponent = s * t
            let expTerm = Complex.exp(exponent)
            let integrand = F(s) * expTerm
            
            switch i {
            case 0,
                _ where i == n:
                sum += integrand
            case _ where i % 2 == 0:
                sum += 2 * integrand
            default:
                sum += 4 * integrand
            }
        }
        
        return (sum * h / (2 * .pi)).real
    }
    // MARK: - User Interface
    func mainMenu() {
        var running = true
        
        while running {
            print("\nSwift ODE Solver")
            print("1. Solve ODE")
            print("2. Numerical Integration")
            print("3. Inverse Laplace Transform")
            print("4. Exit")
            print("Enter choice: ", terminator: "")
            
            guard let choice = readLine(), let option = Int(choice) else {
                print("Invalid input")
                continue
            }
            
            switch option {
            case 1:
                solveODEMenu()
            case 2:
                integrationMenu()
            case 3:
                laplaceMenu()
            case 4:
                running = false
            default:
                print("Invalid choice")
            }
        }
    }
    
    // MARK: - Menu Handlers
    private func solveODEMenu() {
        print("\nEnter ODE parameters:")
        
        guard let (x0, y0, h, steps) = getNumericalInputs() else { return }
        
        // Example ODE: dy/dx = -2x*y
        let odeFunction: (Double, Double) -> Double = { x, y in
            return -2 * x * y
        }
        
        ODESolver.eulerMethod(x0: x0, y0: y0, h: h, steps: steps, f: odeFunction)
        ODESolver.rungeKutta4(x0: x0, y0: y0, h: h, steps: steps, f: odeFunction)
        ODESolver.rkf45(x0: x0, y0: y0, h: h, tolerance: 1e-6, maxSteps: steps, f: odeFunction)
    }
    
    private func integrationMenu() {
        print("\nEnter integration limits and number of intervals:")
        guard let a = getDoubleInput(prompt: "Lower limit a: "),
              let b = getDoubleInput(prompt: "Upper limit b: "),
              let n = getIntInput(prompt: "Intervals (even): ") else { return }
        
        // Example function: ∫x^2 dx
        do {
            let result = try ODESolver.simpsonsRule(a: a, b: b, n: n) { x in
                return x * x
            }
            print(String(format: "Integral result: %.6f", result))
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
    
    private func laplaceMenu() {
        print("\nEnter Laplace transform parameters:")
        guard let t = getDoubleInput(prompt: "t value: "),
              let gamma = getDoubleInput(prompt: "Gamma: "),
              let n = getIntInput(prompt: "Intervals (even): ") else { return }
        
        // Example transform: L{1} = 1/s
        do {
            let result = try ODESolver.inverseLaplace(t: t, gamma: gamma, n: n) { s in
                return 1 / s
            }
            print(String(format: "Inverse Laplace result: %.6f", result))
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Input Helpers
    private func getDoubleInput(prompt: String) -> Double? {
        print(prompt, terminator: "")
        guard let input = readLine(), let value = Double(input) else {
            print("Invalid numerical input")
            return nil
        }
        return value
    }
    
    private func getIntInput(prompt: String) -> Int? {
        print(prompt, terminator: "")
        guard let input = readLine(), let value = Int(input) else {
            print("Invalid integer input")
            return nil
        }
        return value
    }
    
    private func getNumericalInputs() -> (x0: Double, y0: Double, h: Double, steps: Int)? {
        guard let x0 = getDoubleInput(prompt: "Initial x: "),
              let y0 = getDoubleInput(prompt: "Initial y: "),
              let h = getDoubleInput(prompt: "Step size: "),
              let steps = getIntInput(prompt: "Number of steps: ") else {
            return nil
        }
        return (x0, y0, h, steps)
    }
    
}

