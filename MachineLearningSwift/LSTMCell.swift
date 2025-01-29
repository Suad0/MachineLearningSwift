//
//  LSTMCell.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 08.10.24.
//


/*
 
 Step-by-Step LSTM Implementation in Swift:
 1. LSTM Cell Overview
 LSTM cells operate with three main gates:
 
 Forget Gate: Decides how much of the past information should be forgotten.
 Input Gate: Decides how much of new information should be stored.
 Output Gate: Decides what part of the cell state should be output as the hidden state.
 The equations for these gates and state updates are as follows:
 
 */




import Foundation
import Accelerate

// Define a sigmoid activation function
func sigmoid(_ x: [Float]) -> [Float] {
    return x.map { 1 / (1 + exp(-$0)) }
}

// Define a tanh activation function
func tanh(_ x: [Float]) -> [Float] {
    return x.map { (exp(2 * $0) - 1) / (exp(2 * $0) + 1) }
}

// Vector operations (element-wise multiplication, addition)
func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
    return zip(a, b).map(*)
}

func elementWiseAdd(_ a: [Float], _ b: [Float]) -> [Float] {
    return zip(a, b).map(+)
}



// Matrix multiply: uses Accelerate framework for fast computations
private func matrixMultiply(_ A: [Float], _ B: [Float], rowsA: Int, colsA: Int, colsB: Int) -> [Float] {
    var C = [Float](repeating: 0.0, count: rowsA * colsB)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rowsA), Int32(colsB), Int32(colsA), 1.0, A, Int32(colsA), B, Int32(colsB), 0.0, &C, Int32(colsB))
    return C
}



class LSTMCell {
    
    var inputSize: Int
    var hiddenSize: Int
    
    // Weight matrices
    var Wf, Wi, Wo, Wc: [Float] // Weight matrices for forget, input, output, and candidate cell state
    var bf, bi, bo, bc: [Float] // Bias terms for the gates and cell state
    
    var h: [Float] // Hidden state
    var C: [Float] // Cell state
    
    init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        // Initialize weights and biases randomly
        Wf = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wi = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wo = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        Wc = (0..<hiddenSize * (inputSize + hiddenSize)).map { _ in Float.random(in: -0.1...0.1) }
        
        bf = [Float](repeating: 0, count: hiddenSize)
        bi = [Float](repeating: 0, count: hiddenSize)
        bo = [Float](repeating: 0, count: hiddenSize)
        bc = [Float](repeating: 0, count: hiddenSize)
        
        h = [Float](repeating: 0, count: hiddenSize)
        C = [Float](repeating: 0, count: hiddenSize)
    }
    
    // Forward pass of the LSTM cell
    func forward(input: [Float]) -> [Float] {
        // Combine hidden state and input
        let concat = h + input
        
        // Forget gate: controls how much of the previous cell state to forget
        // Formula: ft = σ(Wf * [h, x] + bf)
        let ft = sigmoid(matrixMultiply(Wf, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1) + bf)
        
        
        // Input gate: controls how much of the new input to allow
        // Formula: it = σ(Wi * [h, x] + bi)
        
        let it = sigmoid(matrixMultiply(Wi, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1) + bi)
        
        
        // Output gate: controls how much of the cell state to output
        // Formula: ot = σ(Wo * [h, x] + bo)
        
        let ot = sigmoid(matrixMultiply(Wo, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1) + bo)
        
        
        // Candidate cell state: proposed new cell state
        // Formula: C_hat = tanh(Wc * [h, x] + bc)
        
        let C_hat = tanh(matrixMultiply(Wc, concat, rowsA: hiddenSize, colsA: inputSize + hiddenSize, colsB: 1) + bc)
        
        
        // Update cell state: combines the previous cell state with the candidate cell state
        // Formula: C = ft * C + it * C_hat
        
        C = elementWiseAdd(elementWiseMultiply(ft, C), elementWiseMultiply(it, C_hat))
        
        // Update hidden state: the output of the LSTM cell
        // Formula: h = ot * tanh(C)
        
        h = elementWiseMultiply(ot, tanh(C))
        
        return h
    }
}






