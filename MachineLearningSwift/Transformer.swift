//
//  Transformer.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 16.12.24.
//

import Foundation
import Accelerate

enum TransformerError: Error {
    case invalidMatrixDimensions
    case incompatibleInputSize
    case configurationError(String)
}

extension Array where Element == [Double] {
    func validateDimensions() throws {
        guard !self.isEmpty && Set(self.map { $0.count }).count == 1 else {
            throw TransformerError.invalidMatrixDimensions
        }
    }
}

class Transformer {
    private let config: TransformerConfiguration
    private var isTraining: Bool = true
    
    // Learnable parameters for attention and FFN
    private var queryWeights: [[Double]]
    private var keyWeights: [[Double]]
    private var valueWeights: [[Double]]
    private var outputWeights: [[Double]]
    private var outputBias: [Double]
    private var ffWeights1: [[Double]]
    private var ffWeights2: [[Double]]
    private var ffBias1: [Double]
    private var ffBias2: [Double]
    
    struct TransformerConfiguration {
        let inputSize: Int
        let hiddenSize: Int
        let outputSize: Int
        let numHeads: Int
        let ffHiddenSize: Int
        let dropout: Double
        let epsilon: Double
        let learningRate: Double
        
        init(
            inputSize: Int,
            hiddenSize: Int,
            outputSize: Int,
            numHeads: Int,
            ffHiddenSize: Int,
            dropout: Double = 0.1,
            epsilon: Double = 1e-5,
            learningRate: Double = 0.001
        ) throws {
            guard hiddenSize % numHeads == 0 else {
                throw TransformerError.configurationError("hiddenSize must be divisible by numHeads")
            }
            guard inputSize > 0, hiddenSize > 0, outputSize > 0,
                  numHeads > 0, ffHiddenSize > 0,
                  dropout >= 0 && dropout < 1,
                  learningRate > 0 else {
                throw TransformerError.configurationError("Invalid configuration parameters")
            }
            
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.outputSize = outputSize
            self.numHeads = numHeads
            self.ffHiddenSize = ffHiddenSize
            self.dropout = dropout
            self.epsilon = epsilon
            self.learningRate = learningRate
        }
    }
    
    init(config: TransformerConfiguration) throws {
        self.config = config
        
        // Initialize attention parameters with Xavier
        self.queryWeights = Self.xavierInitializedWeights(rows: config.hiddenSize, cols: config.hiddenSize)
        self.keyWeights = Self.xavierInitializedWeights(rows: config.hiddenSize, cols: config.hiddenSize)
        self.valueWeights = Self.xavierInitializedWeights(rows: config.hiddenSize, cols: config.hiddenSize)
        self.outputWeights = Self.xavierInitializedWeights(rows: config.hiddenSize, cols: config.hiddenSize)
        self.outputBias = [Double](repeating: 0.0, count: config.hiddenSize)
        
        // Initialize FFN parameters
        self.ffWeights1 = Self.xavierInitializedWeights(rows: config.hiddenSize, cols: config.ffHiddenSize)
        self.ffWeights2 = Self.xavierInitializedWeights(rows: config.ffHiddenSize, cols: config.hiddenSize)
        self.ffBias1 = [Double](repeating: 0.0, count: config.ffHiddenSize)
        self.ffBias2 = [Double](repeating: 0.0, count: config.hiddenSize)
    }
    
    private static func xavierInitializedWeights(rows: Int, cols: Int) -> [[Double]] {
        let limit = sqrt(6.0 / Double(rows + cols))
        return (0..<rows).map { _ in
            (0..<cols).map { _ in Double.random(in: -limit...limit) }
        }
    }
    
    func setTrainingMode(_ isTraining: Bool) {
        self.isTraining = isTraining
    }
    
    // MARK: - Positional Encoding
    private func generateSinusoidalEncoding(sequenceLength: Int, embeddingSize: Int) -> [[Double]] {
        var positionalEncoding = Array(repeating: Array(repeating: 0.0, count: embeddingSize), count: sequenceLength)
        for pos in 0..<sequenceLength {
            for i in 0..<embeddingSize {
                let exponent = 2.0 * Double(i / 2) / Double(embeddingSize)
                let divTerm = pow(10000.0, exponent)
                positionalEncoding[pos][i] = (i % 2 == 0) ? sin(Double(pos) / divTerm) : cos(Double(pos) / divTerm)
            }
        }
        return positionalEncoding
    }
    
    // MARK: - Core Transformer Components
    func encoderLayer(input: [[Double]], mask: [[Double]]? = nil) throws -> [[Double]] {
        // Project input to Q, K, V
        let query = try performLinearTransformation(input, weights: queryWeights, bias: [])
        let key = try performLinearTransformation(input, weights: keyWeights, bias: [])
        let value = try performLinearTransformation(input, weights: valueWeights, bias: [])
        
        let attentionOutput = try multiHeadAttention(query: query, key: key, value: value, mask: mask)
        let residual1 = try addVectors(input, attentionOutput)
        let layerNorm1 = layerNormalization(residual1)
        
        // Feed Forward Network
        let ffOutput = try feedForwardNetwork(layerNorm1)
        let residual2 = try addVectors(layerNorm1, ffOutput)
        let layerNorm2 = layerNormalization(residual2)
        
        return layerNorm2
    }
    
    private func multiHeadAttention(
        query: [[Double]],
        key: [[Double]],
        value: [[Double]],
        mask: [[Double]]? = nil
    ) throws -> [[Double]] {
        let headSize = config.hiddenSize / config.numHeads
        var outputs = [[[Double]]]()
        
        for head in 0..<config.numHeads {
            let headQuery = splitHeads(query, headIndex: head, headSize: headSize)
            let headKey = splitHeads(key, headIndex: head, headSize: headSize)
            let headValue = splitHeads(value, headIndex: head, headSize: headSize)
            
            let headOutput = try scaledDotProductAttention(
                query: headQuery,
                key: headKey,
                value: headValue,
                mask: mask,
                dropout: isTraining ? config.dropout : 0.0
            )
            outputs.append(headOutput)
        }
        
        let concatenated = concatenateHeads(outputs)
        return try performLinearTransformation(concatenated, weights: outputWeights, bias: outputBias)
    }
    
    private func scaledDotProductAttention(
        query: [[Double]],
        key: [[Double]],
        value: [[Double]],
        mask: [[Double]]?,
        dropout: Double
    ) throws -> [[Double]] {
        let d_k = Double(query[0].count)
        var scores = try performMatrixMultiplication(query, try transpose(key))
        scores = scores.map { $0.map { $0 / sqrt(d_k) } }
        
        if let mask = mask {
            scores = applyMask(scores, mask: mask)
        }
        
        let weights = softmax(scores)
        let droppedWeights = applyDropout(weights, rate: dropout)
        return try performMatrixMultiplication(droppedWeights, value)
    }
    
    private func feedForwardNetwork(_ input: [[Double]]) throws -> [[Double]] {
        // First linear layer + ReLU
        let layer1 = try performLinearTransformation(input, weights: ffWeights1, bias: ffBias1)
            .map { $0.map { max($0, 0) } }  // ReLU
        
        // Second linear layer
        return try performLinearTransformation(layer1, weights: ffWeights2, bias: ffBias2)
    }
    
    // MARK: - Helper Functions
    private func performLinearTransformation(
        _ input: [[Double]],
        weights: [[Double]],
        bias: [Double]
    ) throws -> [[Double]] {
        try input.validateDimensions()
        try weights.validateDimensions()
        
        let result = try performMatrixMultiplication(input, weights)
        return bias.isEmpty ? result : addBias(result, bias: bias)
    }
    
    private func addBias(_ matrix: [[Double]], bias: [Double]) -> [[Double]] {
        matrix.map { row in zip(row, bias).map { $0 + $1 } }
    }
    
    private func addVectors(_ a: [[Double]], _ b: [[Double]]) throws -> [[Double]] {
        guard a.count == b.count, a[0].count == b[0].count else {
            throw TransformerError.invalidMatrixDimensions
        }
        return zip(a, b).map { zip($0, $1).map { $0 + $1 } }
    }
    
    private func applyMask(_ scores: [[Double]], mask: [[Double]]) -> [[Double]] {
        zip(scores, mask).map { zip($0, $1).map { $0 + $1 } }
    }
    
    private func softmax(_ matrix: [[Double]]) -> [[Double]] {
        matrix.map { row in
            let maxVal = row.max() ?? 0
            let expRow = vForce.exp(row.map { $0 - maxVal })
            let sumExp = expRow.reduce(0, +)
            return expRow.map { $0 / sumExp }
        }
    }
    
    private func layerNormalization(_ input: [[Double]]) -> [[Double]] {
        input.map { row in
            let mean = row.reduce(0, +) / Double(row.count)
            let variance = row.map { pow($0 - mean, 2) }.reduce(0, +) / Double(row.count)
            return row.map { ($0 - mean) / sqrt(variance + config.epsilon) }
        }
    }
    
    private func splitHeads(_ input: [[Double]], headIndex: Int, headSize: Int) -> [[Double]] {
        input.map { Array($0[headIndex*headSize..<(headIndex+1)*headSize]) }
    }
    
    private func concatenateHeads(_ heads: [[[Double]]]) -> [[Double]] {
        (0..<heads[0].count).map { rowIdx in
            heads.flatMap { $0[rowIdx] }
        }
    }
    
    private func applyDropout(_ matrix: [[Double]], rate: Double) -> [[Double]] {
        guard rate > 0 else { return matrix }
        return matrix.map { row in
            row.map { Double.random(in: 0...1) < rate ? 0 : $0 / (1 - rate) }
        }
    }
    
    // MARK: - Matrix Operations (Accelerate optimized)
    private func performMatrixMultiplication(_ a: [[Double]], _ b: [[Double]]) throws -> [[Double]] {
        try a.validateDimensions()
        try b.validateDimensions()
        guard a[0].count == b.count else { throw TransformerError.invalidMatrixDimensions }
        
        let rowsA = a.count, colsA = a[0].count, colsB = b[0].count
        var result = [Double](repeating: 0, count: rowsA * colsB)
        var aFlat = a.flatMap { $0 }, bFlat = b.flatMap { $0 }
        
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(rowsA), Int32(colsB), Int32(colsA),
            1.0,
            &aFlat, Int32(colsA),
            &bFlat, Int32(colsB),
            0.0,
            &result, Int32(colsB)
        )
        
        return stride(from: 0, to: result.count, by: colsB).map {
            Array(result[$0..<$0 + colsB])
        }
    }
    
    private func transpose(_ matrix: [[Double]]) throws -> [[Double]] {
        try matrix.validateDimensions()
        guard !matrix.isEmpty else { return [] }
        return (0..<matrix[0].count).map { col in matrix.map { $0[col] } }
    }
}


