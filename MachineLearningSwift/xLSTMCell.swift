//
//  xLSTMCell.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.01.25.
//

import Foundation
import Accelerate

class xLSTMCell {
    
    var inputSize: Int
    var hiddenSize: Int
    var memorySize: Int // Matrix memory size (d)
    
    // Weight matrices
    var Wf, Wi, Wo, Wv, Wk, Wq: [Float] // Weights for forget, input, output, value, key, and query
    var bf, bi, bo: [Float] // Biases for forget, input, and output gates
    
    // Hidden and memory states
    var h: [Float] // Hidden state
    var C: [[Float]] // Matrix memory cell
    
    init(inputSize: Int, hiddenSize: Int, memorySize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.memorySize = memorySize
        
        // Initialize weights and biases randomly
        Wf = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wi = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wo = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wv = (0..<memorySize * (inputSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wk = (0..<memorySize * (inputSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wq = (0..<memorySize * (inputSize)).map { _ in Float.random(in: -0.1...0.1) }
        
        bf = [Float](repeating: 0, count: hiddenSize)
        bi = [Float](repeating: 0, count: hiddenSize)
        bo = [Float](repeating: 0, count: hiddenSize)
        
        h = [Float](repeating: 0, count: hiddenSize)
        C = [[Float]](repeating: [Float](repeating: 0, count: memorySize), count: memorySize)
    }
    
    // Forward pass for xLSTM cell
    func forward(input: [Float]) -> [Float] {
        // Combine input and hidden state
        let concat = h + input
        
        // Forget gate with exponential gating
        let ft = elementWiseAdd(matrixMultiply(Wf, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1), bf).map { exp($0) }
        
        // Input gate with exponential gating
        let it = elementWiseAdd(matrixMultiply(Wi, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1), bi).map { exp($0) }
        
        // Output gate with sigmoid activation
        let ot = sigmoid(matrixMultiply(Wo, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1) + bo)
        
        // Generate key, value, and query vectors
        let vt = matrixMultiply(Wv, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        let kt = matrixMultiply(Wk, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        let qt = matrixMultiply(Wq, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        
        // Update matrix memory cell
        for i in 0..<memorySize {
            for j in 0..<memorySize {
                C[i][j] = ft[i] * C[i][j] + it[i] * vt[i] * kt[j]
            }
        }
        
        // Compute the hidden state
        let Ctq = C.map { row in elementWiseMultiply(row, qt).reduce(0, +) }
        let norm = max(Ctq.reduce(0, +), 1.0)
        h = elementWiseMultiply(ot, Ctq.map { $0 / norm })
        
        return h
    }
}

