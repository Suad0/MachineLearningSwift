//
//  Transformer.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 16.12.24.
//

import Foundation

class Transformer {
    
    private var inputSize: Int
    private var hiddenSize: Int
    private var outputSize: Int
    private var numHeads: Int
    private var ffHiddenSize: Int
    
    // Neural network for feedforward layers
    private var feedForwardNetwork: NeuralNetwork
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int, numHeads: Int, ffHiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numHeads = numHeads
        self.ffHiddenSize = ffHiddenSize
        
        // Initialize the feedforward network
        self.feedForwardNetwork = NeuralNetwork(inputSize: hiddenSize, hiddenSize: ffHiddenSize, outputSize: hiddenSize)
    }
    
    // MARK: - Positional Encoding
    func positionalEncoding(sequenceLength: Int, embeddingSize: Int) -> [[Double]] {
        var positionalEncoding = Array(repeating: Array(repeating: 0.0, count: embeddingSize), count: sequenceLength)
        for pos in 0..<sequenceLength {
            for i in 0..<embeddingSize {
                if i % 2 == 0 {
                    positionalEncoding[pos][i] = sin(Double(pos) / pow(10000, Double(i) / Double(embeddingSize)))
                } else {
                    positionalEncoding[pos][i] = cos(Double(pos) / pow(10000, Double(i) / Double(embeddingSize)))
                }
            }
        }
        return positionalEncoding
    }
    
    // MARK: - Scaled Dot-Product Attention
    private func scaledDotProductAttention(query: [[Double]], key: [[Double]], value: [[Double]]) -> [[Double]] {
        let d_k = Double(key[0].count)
        var attentionScores = matrixMultiply(query, transpose(key))
        attentionScores = attentionScores.map { $0.map { $0 / sqrt(d_k) } }
        let attentionWeights = softmax2D(attentionScores)
        return matrixMultiply(attentionWeights, value)
    }
    
    // MARK: - Multi-Head Attention
    func multiHeadAttention(query: [[Double]], key: [[Double]], value: [[Double]]) -> [[Double]] {
        // Split into multiple heads
        let headSize = hiddenSize / numHeads
        var concatenatedOutputs = [[Double]]()
        
        for _ in 0..<numHeads {
            // Implement splitting query, key, and value here
            let output = scaledDotProductAttention(query: query, key: key, value: value)
            concatenatedOutputs.append(contentsOf: output)
        }
        
        // Concatenate outputs and apply linear transformation
        return concatenatedOutputs
    }
    
    // MARK: - Encoder
    func encoderLayer(input: [[Double]]) -> [[Double]] {
        // 1. Multi-Head Self-Attention
        let attentionOutput = multiHeadAttention(query: input, key: input, value: input)
        
        // 2. Add Residual Connection + Layer Normalization
        let normalizedAttention = layerNormalization(input + attentionOutput)
        
        // 3. Feedforward Neural Network
        let ffOutput = feedForwardNetwork.predict(normalizedAttention.flatMap { $0 })
        
        // Reshape ffOutput and add to normalizedAttention
        let reshapedFFOutput = normalizedAttention.map { _ in ffOutput }
        let finalOutput = zip(normalizedAttention, reshapedFFOutput).map { zip($0, $1).map(+) }
        
        // 4. Add Residual Connection + Layer Normalization
        return layerNormalization(finalOutput)
    }
    
    
    // MARK: - Utility Functions
    private func matrixMultiply(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
        let rowsA = a.count, colsA = a[0].count
        let rowsB = b.count, colsB = b[0].count
        
        guard colsA == rowsB else {
            fatalError("Matrix dimensions do not match for multiplication")
        }
        
        var result = Array(repeating: Array(repeating: 0.0, count: colsB), count: rowsA)
        
        for i in 0..<rowsA {
            for j in 0..<colsB {
                for k in 0..<colsA {
                    result[i][j] += a[i][k] * b[k][j]
                }
            }
        }
        
        return result
    }
    
    private func transpose(_ matrix: [[Double]]) -> [[Double]] {
        guard let firstRow = matrix.first else { return [] }
        return (0..<firstRow.count).map { col in matrix.map { $0[col] } }
    }
    
    private func softmax2D(_ matrix: [[Double]]) -> [[Double]] {
        return matrix.map { row in
            let maxVal = row.max() ?? 0
            let expRow = row.map { exp($0 - maxVal) }
            let sumExp = expRow.reduce(0, +)
            return expRow.map { $0 / sumExp }
        }
    }
    
    private func layerNormalization(_ input: [[Double]]) -> [[Double]] {
        return input // Placeholder: implement mean/variance normalization
    }
}
