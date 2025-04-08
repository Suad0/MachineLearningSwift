//
//  Tensor.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 20.01.25.
//

import Foundation
import Accelerate


enum TensorError: Error {
    case dimensionMismatch(message: String)
    case invalidOperation(message: String)
}

class Tensor: Hashable {
    
    var value: [[Float]]
    var grad: [[Float]]
    var gradFn: (() -> Void)?
    var requires_grad: Bool
    
    
    
    
    // Computation graph tracking
    var children: [Tensor] = []
    var backward_called = false
    
    // Unique identifier for hashable conformance
    private let id = UUID()
    
    // Initializers
    init(_ value: Float, requires_grad: Bool = true) {
        self.value = [[value]]
        self.grad = Array(repeating: Array(repeating: 0.0, count: 1), count: 1)
        self.requires_grad = requires_grad
        self.gradFn = nil
    }
    
    init(_ value: [[Float]], requires_grad: Bool = true) {
        self.value = value
        self.grad = Array(repeating: Array(repeating: 0.0, count: value[0].count), count: value.count)
        self.requires_grad = requires_grad
        self.gradFn = nil
    }
    
    // Hashable conformance
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // Equatable conformance
    static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        return lhs.id == rhs.id
    }
    
    // Topological sort for backward pass
    func backward(gradient: [[Float]]? = nil) {
        // Prevent multiple backward calls
        guard !backward_called else { return }
        backward_called = true
        
        // Use provided gradient or default to 1.0
        let seed = gradient ?? Array(repeating: Array(repeating: 1.0, count: value[0].count), count: value.count)
        
        // Accumulate gradient
        for i in 0..<value.count {
            for j in 0..<value[0].count {
                grad[i][j] += seed[i][j]
            }
        }
        
        // Topological sort and backward pass
        var visited = Set<Tensor>()
        var stack: [Tensor] = []
        
        func topologicalSort(_ node: Tensor) {
            if visited.contains(node) { return }
            visited.insert(node)
            
            for child in node.children {
                topologicalSort(child)
            }
            
            stack.append(node)
        }
        
        topologicalSort(self)
        
        // Reverse order for backward pass
        for node in stack.reversed() {
            node.gradFn?()
        }
    }
    
    // Utility methods
    func zero_grad() {
        grad = Array(repeating: Array(repeating: 0.0, count: value[0].count), count: value.count)
    }
    
    
    
    func matmul(_ other: Tensor) throws -> Tensor {
        // Validate matrix multiplication dimensions
        guard value[0].count == other.value.count else {
            throw TensorError.dimensionMismatch(
                message: "Matrix multiplication dimensions incompatible. " +
                "First matrix columns (\(value[0].count)) must equal " +
                "second matrix rows (\(other.value.count))"
            )
        }
        
        // Perform matrix multiplication
        let resultRows = value.count
        let resultCols = other.value[0].count
        
        var resultValue = Array(
            repeating: Array(repeating: Float(0), count: resultCols),
            count: resultRows
        )
        
        for i in 0..<resultRows {
            for j in 0..<resultCols {
                resultValue[i][j] = (0..<value[0].count).reduce(0) { sum, k in
                    sum + value[i][k] * other.value[k][j]
                }
            }
        }
        
        let result = Tensor(resultValue)
        
        // Gradient computation for matrix multiplication
        result.gradFn = {
            // Gradient for first matrix
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        for k in 0..<result.value[0].count {
                            self.grad[i][j] += result.grad[i][k] * other.value[j][k]
                        }
                    }
                }
            }
            
            // Gradient for second matrix
            if other.requires_grad {
                for i in 0..<other.value.count {
                    for j in 0..<other.value[0].count {
                        for k in 0..<result.value.count {
                            other.grad[i][j] += result.grad[k][j] * self.value[k][i]
                        }
                    }
                }
            }
        }
        
        result.children = [self, other]
        return result
    }
    
    
    
    // Trigonometric Operations
    func sin() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { Foundation.sin($0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += Foundation.cos(self.value[i][j]) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    func cos() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { Foundation.cos($0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += -Foundation.sin(self.value[i][j]) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // Logarithmic Operations
    func log() throws -> Tensor {
        // Check for non-positive values
        guard !value.contains(where: { $0.contains { $0 <= 0 } }) else {
            throw TensorError.invalidOperation(
                message: "Logarithm is undefined for non-positive values"
            )
        }
        
        let result = Tensor(
            value.map { row in
                row.map { Foundation.log($0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += (1.0 / self.value[i][j]) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // Dimension checking utility
    func checkDimensions(with other: Tensor) throws {
        guard value.count == other.value.count &&
                value[0].count == other.value[0].count else {
            throw TensorError.dimensionMismatch(
                message: "Tensor dimensions must match. " +
                "First tensor: \(value.count)x\(value[0].count), " +
                "Second tensor: \(other.value.count)x\(other.value[0].count)"
            )
        }
    }
    
    
    func createNeuralNetworkLayer(inputSize: Int, outputSize: Int) -> (weights: Tensor, bias: Tensor) {
        let weights = Tensor.xavier(rows: inputSize, cols: outputSize)
        let bias = Tensor(Array(repeating: Array(repeating: 0.0, count: outputSize), count: 1))
        return (weights, bias)
    }
    
    
    
}



// Operator Overloading
// Addition
func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    // Ensure compatible dimensions
    guard lhs.value.count == rhs.value.count &&
            lhs.value[0].count == rhs.value[0].count else {
        fatalError("Tensor dimensions must match for addition")
    }
    
    let result = Tensor(
        lhs.value.enumerated().map { i, row in
            row.enumerated().map { j, val in
                val + rhs.value[i][j]
            }
        }
    )
    
    result.gradFn = {
        if lhs.requires_grad {
            for i in 0..<lhs.value.count {
                for j in 0..<lhs.value[0].count {
                    lhs.grad[i][j] += result.grad[i][j]
                }
            }
        }
        if rhs.requires_grad {
            for i in 0..<rhs.value.count {
                for j in 0..<rhs.value[0].count {
                    rhs.grad[i][j] += result.grad[i][j]
                }
            }
        }
    }
    
    result.children = [lhs, rhs]
    return result
}

// Multiplication
func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    // Ensure compatible dimensions for element-wise multiplication
    guard lhs.value.count == rhs.value.count &&
            lhs.value[0].count == rhs.value[0].count else {
        fatalError("Tensor dimensions must match for multiplication")
    }
    
    let result = Tensor(
        lhs.value.enumerated().map { i, row in
            row.enumerated().map { j, val in
                val * rhs.value[i][j]
            }
        }
    )
    
    result.gradFn = {
        if lhs.requires_grad {
            for i in 0..<lhs.value.count {
                for j in 0..<lhs.value[0].count {
                    lhs.grad[i][j] += rhs.value[i][j] * result.grad[i][j]
                }
            }
        }
        if rhs.requires_grad {
            for i in 0..<rhs.value.count {
                for j in 0..<rhs.value[0].count {
                    rhs.grad[i][j] += lhs.value[i][j] * result.grad[i][j]
                }
            }
        }
    }
    
    result.children = [lhs, rhs]
    return result
}

// Power operation
func pow(_ tensor: Tensor, _ power: Float) -> Tensor {
    let result = Tensor(
        tensor.value.map { row in
            row.map { val in
                Foundation.pow(val, power)
            }
        }
    )
    
    result.gradFn = {
        if tensor.requires_grad {
            for i in 0..<tensor.value.count {
                for j in 0..<tensor.value[0].count {
                    tensor.grad[i][j] += power *
                    Foundation.pow(tensor.value[i][j], power - 1) *
                    result.grad[i][j]
                }
            }
        }
    }
    
    result.children = [tensor]
    return result
}

extension Tensor {
    
    /// Activation Functions
    
    // ReLU (Rectified Linear Unit) Activation
    func relu() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { max(0, $0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += self.value[i][j] > 0 ? result.grad[i][j] : 0
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // Sigmoid Activation
    func sigmoid() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { 1 / (1 + _math.exp(-$0)) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        let sigmoid = result.value[i][j]
                        self.grad[i][j] += sigmoid * (1 - sigmoid) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // Tanh Activation
    func tanh() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { Foundation.tanh($0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        let tanhValue = result.value[i][j]
                        self.grad[i][j] += (1 - tanhValue * tanhValue) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    /// Statistical Operations
    
    func mean() -> Tensor {
        let totalSum = value.flatMap { $0 }.reduce(0, +)
        let count = value.count * value[0].count
        let meanValue = totalSum / Float(count)
        
        let result = Tensor(meanValue)
        
        result.gradFn = {
            if self.requires_grad {
                let gradientValue = result.grad[0][0] / Float(self.value.count * self.value[0].count)
                
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += gradientValue
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // Standard Deviation
    func std() -> Tensor {
        let mean = self.mean().value[0][0]
        
        let variance = value.flatMap { row in
            row.map { pow($0 - mean, 2) }
        }.reduce(0, +) / Float(value.count * value[0].count)
        
        let stdDev = sqrt(variance)
        
        let result = Tensor(stdDev)
        
        result.gradFn = {
            if self.requires_grad {
                let gradientValue = result.grad[0][0]
                
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += gradientValue * (self.value[i][j] - mean) / (stdDev * Float(self.value.count * self.value[0].count))
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    /// Initialization Methods
    
    static func xavier(rows: Int, cols: Int) -> Tensor {
        let limit = sqrt(6.0 / Float(rows + cols))
        let randomValues = (0..<rows).map { _ in
            (0..<cols).map { _ in
                Float.random(in: -limit...limit)
            }
        }
        return Tensor(randomValues)
    }
    
    // He initialization (for ReLU networks)
    static func he(rows: Int, cols: Int) -> Tensor {
        let stddev = sqrt(2.0 / Float(rows))
        let randomValues = (0..<rows).map { _ in
            (0..<cols).map { _ in
                Float.random(in: -stddev...stddev)
            }
        }
        return Tensor(randomValues)
    }
    
    /// Regularization Methods
    
    // L1 Regularization (Lasso)
    func l1Regularization(lambda: Float) -> Float {
        return lambda * value.flatMap { $0 }.reduce(0) { $0 + Swift.abs($1) }
    }
    
    // L2 Regularization (Ridge)
    func l2Regularization(lambda: Float) -> Float {
        return lambda * value.flatMap { $0 }.reduce(0) { $0 + $1 * $1 }
    }
    
    
    
    
}




extension Tensor {
    // MARK: - Advanced Numerical Operations
    
    /// Element-wise absolute value
    func abs() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { Float(Foundation.abs(Int32($0))) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += (self.value[i][j] >= 0 ? 1 : -1) * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    /// Element-wise exponential
    func exp() -> Tensor {
        let result = Tensor(
            value.map { row in
                row.map { Foundation.exp($0) }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += result.value[i][j] * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // MARK: - Performance Optimizations
    
    /// Parallel matrix multiplication using Accelerate framework
    func parallelMatmul(_ other: Tensor) throws -> Tensor {
        // Validate matrix multiplication dimensions
        guard value[0].count == other.value.count else {
            throw TensorError.dimensionMismatch(
                message: "Matrix multiplication dimensions incompatible"
            )
        }
        
        let resultRows = value.count
        let resultCols = other.value[0].count
        let commonDim = value[0].count
        
        var resultValue = Array(
            repeating: Array(repeating: Float(0), count: resultCols),
            count: resultRows
        )
        
        // Prepare matrices for BLAS
        var flatA = value.flatMap { $0 }
        var flatB = other.value.flatMap { $0 }
        var flatC = Array(repeating: Float(0), count: resultRows * resultCols)
        
        // BLAS parameters
        let m = Int32(resultRows)     // Number of rows in A
        let n = Int32(resultCols)     // Number of columns in B
        let k = Int32(commonDim)      // Number of columns in A / rows in B
        let alpha: Float = 1.0
        let beta: Float = 0.0
        
        // Use row-major order with correct leading dimensions
        cblas_sgemm(
            CblasRowMajor,           // Matrix layout
            CblasNoTrans,            // TransA
            CblasNoTrans,            // TransB
            m,                       // Rows in A
            n,                       // Columns in B
            k,                       // Columns in A / Rows in B
            alpha,                   // Scalar multiplier
            flatA,                   // Matrix A
            k,                       // Leading dimension of A (number of columns)
            flatB,                   // Matrix B
            n,                       // Leading dimension of B (number of columns)
            beta,                    // Scalar beta
            &flatC,                  // Result matrix
            n                        // Leading dimension of C (number of columns)
        )
        
        // Convert back to 2D array
        resultValue = stride(from: 0, to: flatC.count, by: resultCols).map {
            Array(flatC[$0..<min($0 + resultCols, flatC.count)])
        }
        
        let result = Tensor(resultValue)
        
        // Gradient computation
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        for k in 0..<result.value[0].count {
                            self.grad[i][j] += result.grad[i][k] * other.value[j][k]
                        }
                    }
                }
            }
            
            if other.requires_grad {
                for i in 0..<other.value.count {
                    for j in 0..<other.value[0].count {
                        for k in 0..<result.value.count {
                            other.grad[i][j] += result.grad[k][j] * self.value[k][i]
                        }
                    }
                }
            }
        }
        
        result.children = [self, other]
        return result
    }
    
    // MARK: - Advanced Statistical Methods
    
    /// Compute variance
    func variance() -> Tensor {
        let mean = self.mean().value[0][0]
        
        let varianceValue = value.flatMap { row in
            row.map { pow($0 - mean, 2) }
        }.reduce(0, +) / Float(value.count * value[0].count)
        
        let result = Tensor(varianceValue)
        
        result.gradFn = {
            if self.requires_grad {
                let gradientValue = result.grad[0][0]
                
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        self.grad[i][j] += gradientValue * 2 * (self.value[i][j] - mean) / Float(self.value.count * self.value[0].count)
                    }
                }
            }
        }
        
        result.children = [self]
        return result
    }
    
    // MARK: - Numerical Stability Improvements
    
    /// Safe division to prevent divide-by-zero errors
    func divide(by other: Tensor, epsilon: Float = 1e-7) throws -> Tensor {
        // Check if tensors have compatible dimensions
        guard value.count == other.value.count &&
                value[0].count == other.value[0].count else {
            throw TensorError.dimensionMismatch(
                message: "Tensor dimensions must match for element-wise division. " +
                "First tensor: \(value.count)x\(value[0].count), " +
                "Second tensor: \(other.value.count)x\(other.value[0].count)"
            )
        }
        
        let result = Tensor(
            value.enumerated().map { i, row in
                row.enumerated().map { j, val in
                    let denominator = other.value[i][j]
                    return val / (Swift.abs(denominator) > epsilon ? denominator : (denominator >= 0 ? epsilon : -epsilon))
                }
            }
        )
        
        result.gradFn = {
            if self.requires_grad {
                for i in 0..<self.value.count {
                    for j in 0..<self.value[0].count {
                        let denominator = other.value[i][j]
                        let safeDerivative = 1 / (Swift.abs(denominator) > epsilon ? denominator : (denominator >= 0 ? epsilon : -epsilon))
                        self.grad[i][j] += safeDerivative * result.grad[i][j]
                    }
                }
            }
            
            if other.requires_grad {
                for i in 0..<other.value.count {
                    for j in 0..<other.value[0].count {
                        let denominator = other.value[i][j]
                        let safeDerivative = -self.value[i][j] / pow(Swift.abs(denominator) > epsilon ? denominator : (denominator >= 0 ? epsilon : -epsilon), 2)
                        other.grad[i][j] += safeDerivative * result.grad[i][j]
                    }
                }
            }
        }
        
        result.children = [self, other]
        return result
    }
    
    // MARK: - Debugging and Introspection
    
    /// Detailed tensor description
    func tensorDescription() -> String {
        let shapeDescription = "Shape: \(value.count)x\(value[0].count)"
        let dataTypeDescription = "Type: Float"
        let gradDescription = requires_grad ? "Gradient tracking: Enabled" : "Gradient tracking: Disabled"
        
        let valuePreview = value.prefix(3).map { row in
            row.prefix(3).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        }.joined(separator: "; ")
        
        return """
        Tensor Description:
        \(shapeDescription)
        \(dataTypeDescription)
        \(gradDescription)
        First values: [\(valuePreview)]
        Memory footprint: \(memoryFootprint()) bytes
        """
    }
    
    /// Calculate memory footprint
    func memoryFootprint() -> Int {
        let valueMemory = value.count * value[0].count * MemoryLayout<Float>.stride
        let gradMemory = grad.count * grad[0].count * MemoryLayout<Float>.stride
        return valueMemory + gradMemory
    }
}

// MARK: - Serialization Extension
extension Tensor {
    /// Convert tensor to JSON
    func toJSON() -> [String: Any] {
        return [
            "shape": [value.count, value[0].count],
            "values": value,
            "requires_grad": requires_grad
        ]
    }
    
    /// JSON deserialization with more robust error handling
    static func fromJSON(_ json: [String: Any]) -> Tensor? {
        guard let values = json["values"] as? [[Float]],
              let requiresGrad = json["requires_grad"] as? Bool else {
            print("Invalid JSON format for Tensor reconstruction")
            return nil
        }
        
        return Tensor(values, requires_grad: requiresGrad)
    }
}

