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
    private let feedForwardNetwork: NeuralNetwork
    
    // Comprehensive configuration structure
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
    
    // Enhanced positional encoding strategies
    enum PositionalEncodingType {
        case sinusoidal
        case learned
        case hybrid
    }
    
    // Initializer with comprehensive setup
    init(config: TransformerConfiguration) throws {
        self.config = config
        self.feedForwardNetwork = NeuralNetwork(
            inputSize: config.hiddenSize,
            hiddenSize: config.ffHiddenSize,
            outputSize: config.hiddenSize
        )
    }
    
    // Advanced positional encoding method
    func positionalEncoding(
        sequenceLength: Int,
        embeddingSize: Int,
        type: PositionalEncodingType = .sinusoidal
    ) throws -> [[Double]] {
        switch type {
        case .sinusoidal:
            return generateSinusoidalEncoding(sequenceLength: sequenceLength, embeddingSize: embeddingSize)
        case .learned:
            return generateLearnedEncoding(sequenceLength: sequenceLength, embeddingSize: embeddingSize)
        case .hybrid:
            let sinusoidal = generateSinusoidalEncoding(sequenceLength: sequenceLength, embeddingSize: embeddingSize)
            let learned = generateLearnedEncoding(sequenceLength: sequenceLength, embeddingSize: embeddingSize)
            return zip(sinusoidal, learned).map { zip($0, $1).map { ($0 + $1) / 2 } }
        }
    }
    
    // Sinusoidal positional encoding with improved accuracy
    private func generateSinusoidalEncoding(sequenceLength: Int, embeddingSize: Int) -> [[Double]] {
        var positionalEncoding = Array(repeating: Array(repeating: 0.0, count: embeddingSize), count: sequenceLength)
        
        for pos in 0..<sequenceLength {
            for i in 0..<embeddingSize {
                let div_term = pow(10000, Double(i) / Double(embeddingSize))
                positionalEncoding[pos][i] = (i % 2 == 0)
                ? sin(Double(pos) / div_term)
                : cos(Double(pos) / div_term)
            }
        }
        
        return positionalEncoding
    }
    
    // Learned positional encoding with advanced initialization
    private func generateLearnedEncoding(sequenceLength: Int, embeddingSize: Int) -> [[Double]] {
        var positionalEmbeddings = Array(
            repeating: Array(repeating: 0.0, count: embeddingSize),
            count: sequenceLength
        )
        
        let stddev = sqrt(2.0 / Double(sequenceLength + embeddingSize))
        
        for i in 0..<sequenceLength {
            positionalEmbeddings[i] = (0..<embeddingSize).map { _ in
                Double.random(in: -stddev...stddev)
            }
        }
        
        return positionalEmbeddings
    }
    
    // High-performance matrix multiplication
    private func performMatrixMultiplication(_ a: [[Double]], _ b: [[Double]]) throws -> [[Double]] {
        try a.validateDimensions()
        try b.validateDimensions()
        
        guard a[0].count == b.count else {
            throw TransformerError.invalidMatrixDimensions
        }
        
        let rowsA = a.count
        let colsA = a[0].count
        let colsB = b[0].count
        
        var result = [Double](repeating: 0, count: rowsA * colsB)
        var aFlat = a.flatMap { $0 }
        var bFlat = b.flatMap { $0 }
        
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(rowsA), Int32(colsB), Int32(colsA),
            1.0,
            aFlat, Int32(colsA),
            bFlat, Int32(colsB),
            0.0,
            &result, Int32(colsB)
        )
        
        return stride(from: 0, to: result.count, by: colsB).map { start in
            Array(result[start..<start + colsB])
        }
    }
    
    // Comprehensive multi-head attention implementation
    func multiHeadAttention(
        query: [[Double]],
        key: [[Double]],
        value: [[Double]]
    ) throws -> [[Double]] {
        try query.validateDimensions()
        try key.validateDimensions()
        try value.validateDimensions()
        
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
                dropout: config.dropout
            )
            outputs.append(headOutput)
        }
        
        let concatenatedOutputs = concatenateHeads(outputs)
        return try applyLinearTransformation(concatenatedOutputs)
    }
    
    // Linear transformation with adaptive weight initialization
    private func applyLinearTransformation(_ input: [[Double]]) throws -> [[Double]] {
        let weightRows = input[0].count
        let weightCols = config.hiddenSize
        
        // Adaptive Xavier/Glorot initialization
        let weights = (0..<weightRows).map { _ in
            (0..<weightCols).map { _ in
                let limit = sqrt(6.0 / Double(weightRows + weightCols))
                return Double.random(in: -limit...limit)
            }
        }
        
        // Add bias term
        let biasWeights = (0..<weightCols).map { _ in
            Double.random(in: -0.1...0.1)
        }
        
        // Perform matrix multiplication with learned weights and add bias
        var result = try performMatrixMultiplication(input, weights)
        
        // Apply bias
        result = zip(result, 0..<result.count).map { row, rowIndex in
            zip(row, biasWeights).map { $0 + $1 }
        }
        
        return result
    }
    
    // Sophisticated encoder layer
    func encoderLayer(input: [[Double]]) throws -> [[Double]] {
        let attentionOutput = try multiHeadAttention(query: input, key: input, value: input)
        
        // Layer normalization and residual connection
        let layerNormOutput = layerNormalization(input + attentionOutput)
        
        // Feedforward processing
        let ffOutput = feedForwardNetwork.predict(layerNormOutput.flatMap { $0 })
        let reshapedFFOutput = layerNormOutput.map { _ in ffOutput }
        
        let finalOutput = zip(layerNormOutput, reshapedFFOutput)
            .map { zip($0, $1).map(+) }
        
        return layerNormalization(finalOutput)
    }
    
    // Utility methods (softmax, layer normalization, etc.)
    private func softmax2D(_ matrix: [[Double]]) -> [[Double]] {
        return matrix.map { row in
            let maxVal = row.max() ?? 0
            let expRow = row.map { exp($0 - maxVal) }
            let sumExp = expRow.reduce(0, +)
            return expRow.map { $0 / sumExp }
        }
    }
    
    private func layerNormalization(_ input: [[Double]]) -> [[Double]] {
        return input.map { row in
            let mean = row.reduce(0, +) / Double(row.count)
            let variance = row.map { pow($0 - mean, 2) }.reduce(0, +) / Double(row.count)
            let epsilon = config.epsilon
            return row.map { ($0 - mean) / sqrt(variance + epsilon) }
        }
    }
    
    // Additional utility methods
    private func splitHeads(_ input: [[Double]], headIndex: Int, headSize: Int) -> [[Double]] {
        return input.map { Array($0[headIndex * headSize..<min((headIndex + 1) * headSize, $0.count)]) }
    }
    
    private func concatenateHeads(_ heads: [[[Double]]]) -> [[Double]] {
        return zip(heads[0].indices, heads).map { (rowIdx, head) in
            head.compactMap { $0[rowIdx] }
        }
    }
    
    private func scaledDotProductAttention(
        query: [[Double]],
        key: [[Double]],
        value: [[Double]],
        dropout: Double
    ) throws -> [[Double]] {
        let d_k = Double(key[0].count)
        var attentionScores = try performMatrixMultiplication(query, try transpose(key))
        
        attentionScores = attentionScores.map { $0.map { $0 / sqrt(d_k) } }
        
        let attentionWeights = softmax2D(attentionScores)
        let droppedAttentionWeights = dropout > 0 ? applyDropout(attentionWeights, dropRate: dropout) : attentionWeights
        
        return try performMatrixMultiplication(droppedAttentionWeights, value)
    }
    
    private func transpose(_ matrix: [[Double]]) throws -> [[Double]] {
        try matrix.validateDimensions()
        guard let firstRow = matrix.first else { return [] }
        return (0..<firstRow.count).map { col in matrix.map { $0[col] } }
    }
    
    private func applyDropout(_ matrix: [[Double]], dropRate: Double) -> [[Double]] {
        return matrix.map { row in
            row.map { value in
                Double.random(in: 0...1) < dropRate ? 0 : value / (1 - dropRate)
            }
        }
    }
}





/*
 
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
 /// Generates positional encodings for a sequence of length `sequenceLength` with embedding size `embeddingSize`.
 /// - Parameters:
 ///   - sequenceLength: Length of the input sequence.
 ///   - embeddingSize: Size of each embedding vector.
 /// - Returns: A 2D array of positional encodings.
 func positionalEncoding(sequenceLength: Int, embeddingSize: Int) -> [[Double]] {
 var positionalEncoding = Array(repeating: Array(repeating: 0.0, count: embeddingSize), count: sequenceLength)
 for pos in 0..<sequenceLength {
 for i in 0..<embeddingSize {
 let angle = Double(pos) / pow(10000, Double(i) / Double(embeddingSize))
 positionalEncoding[pos][i] = (i % 2 == 0) ? sin(angle) : cos(angle)
 }
 }
 return positionalEncoding
 }
 
 // MARK: - Scaled Dot-Product Attention
 /// Computes the scaled dot-product attention with optional masking.
 /// - Parameters:
 ///   - query: Query matrix.
 ///   - key: Key matrix.
 ///   - value: Value matrix.
 ///   - mask: Optional mask to block certain positions.
 /// - Returns: Attention output matrix.
 private func scaledDotProductAttention(query: [[Double]], key: [[Double]], value: [[Double]], mask: [[Double]]? = nil) -> [[Double]] {
 let d_k = Double(key[0].count)
 var attentionScores = matrixMultiply(query, transpose(key))
 attentionScores = attentionScores.map { $0.map { $0 / sqrt(d_k) } }
 
 if let mask = mask {
 // Apply mask (assuming mask has the same dimensions as attentionScores)
 attentionScores = zip(attentionScores, mask).map { (row, maskRow) in
 zip(row, maskRow).map { $1 == 0 ? -Double.infinity : $0 }
 }
 }
 
 let attentionWeights = softmax2D(attentionScores)
 return matrixMultiply(attentionWeights, value)
 }
 
 // MARK: - Multi-Head Attention
 /// Splits the input into multiple heads, processes each head, and combines the results.
 /// - Parameters:
 ///   - query: Query matrix.
 ///   - key: Key matrix.
 ///   - value: Value matrix.
 /// - Returns: Output of multi-head attention.
 func multiHeadAttention(query: [[Double]], key: [[Double]], value: [[Double]]) -> [[Double]] {
 let headSize = hiddenSize / numHeads
 var outputs = [[[Double]]]() // Outputs for each head
 
 for head in 0..<numHeads {
 let headQuery = splitHeads(query, headIndex: head, headSize: headSize)
 let headKey = splitHeads(key, headIndex: head, headSize: headSize)
 let headValue = splitHeads(value, headIndex: head, headSize: headSize)
 
 let headOutput = scaledDotProductAttention(query: headQuery, key: headKey, value: headValue)
 outputs.append(headOutput)
 }
 
 // Concatenate outputs from all heads
 let concatenatedOutputs = concatenateHeads(outputs)
 return applyLinearTransformation(concatenatedOutputs)
 }
 
 /// Splits the input matrix into a specific head's slice.
 private func splitHeads(_ input: [[Double]], headIndex: Int, headSize: Int) -> [[Double]] {
 return input.map { Array($0[headIndex * headSize..<min((headIndex + 1) * headSize, $0.count)]) }
 }
 
 /// Concatenates outputs from all heads back into a single matrix.
 private func concatenateHeads(_ heads: [[[Double]]]) -> [[Double]] {
 return zip(heads[0].indices, heads).map { (rowIdx, head) in
 head.compactMap { $0[rowIdx] }
 }
 }
 
 /// Applies a linear transformation to the concatenated multi-head output.
 private func applyLinearTransformation(_ input: [[Double]]) -> [[Double]] {
 // Placeholder: implement a linear transformation.
 return input
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
 assert(a[0].count == b.count, "Matrix dimensions do not match for multiplication")
 
 let rowsA = a.count, colsA = a[0].count
 let colsB = b[0].count
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
 return input.map { row in
 let mean = row.reduce(0, +) / Double(row.count)
 let variance = row.map { pow($0 - mean, 2) }.reduce(0, +) / Double(row.count)
 let epsilon = 1e-5 // To prevent division by zero
 return row.map { ($0 - mean) / sqrt(variance + epsilon) }
 }
 }
 }
 
 */
