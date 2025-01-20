//
//  Tensor.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 20.01.25.
//

import Foundation

enum TensorError: Error {
    case dimensionMismatch(message: String)
    case invalidOperation(message: String)
}

class Tensor: Hashable {
    // Use a multi-dimensional array for values
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

// Example Usage
func testTensor() {
    // Scalar tensor
    let x = Tensor(3.0)
    let y = Tensor(2.0)
    
    let z = x * y + y
    z.backward()
    
    print("Scalar Tensor Example:")
    print("Value of z: \(z.value)")
    print("Gradient of x: \(x.grad)")
    print("Gradient of y: \(y.grad)")
    
    // Multi-dimensional tensor
    let a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = Tensor([[2.0, 3.0], [4.0, 5.0]])
    
    let c = a * b + b
    c.backward()
    
    print("\nMulti-dimensional Tensor Example:")
    print("Value of c: \(c.value)")
    print("Gradient of a: \(a.grad)")
    print("Gradient of b: \(b.grad)")
}

