//
//  xLSTMCell.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.01.25.
//

import Foundation
import Accelerate

class xLSTMCell {
    
    // Network architecture parameters
    private let inputSize: Int
    private let hiddenSize: Int
    private let memorySize: Int
    
    // Weights and biases
    private var Wf, Wi, Wo, Wv, Wk, Wq: [Float]
    private var bf, bi, bo: [Float]
    
    // Gradient storage
    private var gradWf, gradWi, gradWo, gradWv, gradWk, gradWq: [Float]
    private var gradbf, gradbi, gradbo: [Float]
    
    // Training hyperparameters
    private let learningRate: Float
    private let l2RegularizationFactor: Float
    private let gradientClippingThreshold: Float
    
    // State tracking
    private var h: [Float]
    private var C: [Float]
    private var lastInput: [Float] = []
    private var lastOutputs: [[Float]] = []
    
    init(inputSize: Int,
         hiddenSize: Int,
         memorySize: Int,
         learningRate: Float = 0.001,
         l2RegFactor: Float = 1e-5,
         gradClipThreshold: Float = 1.0) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.memorySize = memorySize
        self.learningRate = learningRate
        self.l2RegularizationFactor = l2RegFactor
        self.gradientClippingThreshold = gradClipThreshold
        
        // Weight initialization with Xavier method
        func xavierInitialize(inputSize: Int, outputSize: Int) -> [Float] {
            let limit = sqrt(6.0 / Float(inputSize + outputSize))
            return (0..<inputSize * outputSize).map { _ in Float.random(in: -limit...limit) }
        }
        
        Wf = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: hiddenSize)
        Wi = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: hiddenSize)
        Wo = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: hiddenSize)
        Wv = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        Wk = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        Wq = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        
        bf = [Float](repeating: 0, count: hiddenSize)
        bi = [Float](repeating: 0, count: hiddenSize)
        bo = [Float](repeating: 0, count: hiddenSize)
        
        // Initialize gradient storage
        gradWf = [Float](repeating: 0, count: Wf.count)
        gradWi = [Float](repeating: 0, count: Wi.count)
        gradWo = [Float](repeating: 0, count: Wo.count)
        gradWv = [Float](repeating: 0, count: Wv.count)
        gradWk = [Float](repeating: 0, count: Wk.count)
        gradWq = [Float](repeating: 0, count: Wq.count)
        gradbf = [Float](repeating: 0, count: bf.count)
        gradbi = [Float](repeating: 0, count: bi.count)
        gradbo = [Float](repeating: 0, count: bo.count)
        
        h = [Float](repeating: 0, count: hiddenSize)
        C = [Float](repeating: 0, count: memorySize * memorySize)
    }
    
    // Sequence processing forward pass
    func processSequence(inputs: [[Float]]) -> [[Float]] {
        lastOutputs = []
        lastInput = []
        
        return inputs.map { input in
            lastInput = input
            let output = forward(input: input)
            lastOutputs.append(output)
            return output
        }
    }
    
    // Forward pass with detailed state tracking
    private func forward(input: [Float]) -> [Float] {
        guard input.count == inputSize else {
            fatalError("Input size mismatch")
        }
        
        let concat = h + input
        
        // Compute gates with exponential and sigmoid activations
        let ft = computeExponentialGate(Wf, concat, bf)
        let it = computeExponentialGate(Wi, concat, bi)
        let ot = computeSigmoidGate(Wo, concat, bo)
        
        // Compute key, value, query vectors
        let vt = matrixMultiply(Wv, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        let kt = matrixMultiply(Wk, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        let qt = matrixMultiply(Wq, input, rowsA: memorySize, colsA: inputSize, colsB: 1)
        
        // Update matrix memory cell
        updateMemoryCell(ft: ft, it: it, vt: vt, kt: kt)
        
        // Compute hidden state
        h = computeHiddenState(ot: ot, qt: qt)
        
        return h
    }
    
    // Backpropagation through time (BPTT)
    func backpropagate(targetSequence: [[Float]]) {
        var dhnext = [Float](repeating: 0, count: hiddenSize)
        
        // Reverse iteration through sequence
        for (index, targetOutput) in targetSequence.reversed().enumerated() {
            let currentOutput = lastOutputs[lastOutputs.count - 1 - index]
            let currentInput = lastInput
            
            // Compute output error
            let dh = computeOutputGradient(currentOutput, targetOutput, dhnext)
            
            // Gradient computation for gates and weights
            accumulateGradients(dh: dh, input: currentInput)
            
            dhnext = dh // For next iteration
        }
        
        // Gradient clipping
        applyGradientClipping()
        
        // Update weights with L2 regularization
        updateWeightsWithRegularization()
    }
    
    // Helper methods for backpropagation
    private func computeOutputGradient(_ output: [Float], _ target: [Float], _ dhnext: [Float]) -> [Float] {
        // Simple mean squared error gradient
        return zip(output, target).enumerated().map { (index, pair) in
            let (pred, actual) = pair
            return 2 * (pred - actual) + dhnext[index]
        }
    }
    
    
    private func applyGradientClipping() {
        // Clip gradients to prevent exploding gradients
        let clipThreshold = gradientClippingThreshold
        
        func clipGradients(_ gradients: inout [Float]) {
            let gradNorm = sqrt(gradients.reduce(0) { $0 + $1 * $1 })
            if gradNorm > clipThreshold {
                let scale = clipThreshold / gradNorm
                gradients = gradients.map { $0 * scale }
            }
        }
        

        
        clipGradients(&gradWf)
        clipGradients(&gradWi)
        clipGradients(&gradWo)
        clipGradients(&gradWv)
        clipGradients(&gradWk)
        clipGradients(&gradWq)
        clipGradients(&gradbf)
        clipGradients(&gradbi)
        clipGradients(&gradbo)
        
        
        
        
        
        
        
    }
    
    private func updateWeightsWithRegularization() {
        // Update weights with L2 regularization
        for i in 0..<Wf.count {
            Wf[i] -= learningRate * (gradWf[i] + l2RegularizationFactor * Wf[i])
        }
        
        
        for i in 0..<Wi.count {
            Wi[i] -= learningRate * (gradWi[i] + l2RegularizationFactor * Wi[i])
        }
        
        for i in 0..<Wo.count {
            Wo[i] -= learningRate * (gradWo[i] + l2RegularizationFactor * Wo[i])
        }
        
        for i in 0..<Wv.count {
            Wv[i] -= learningRate * (gradWv[i] + l2RegularizationFactor * Wv[i])
        }
        
        for i in 0..<Wk.count {
            Wk[i] -= learningRate * (gradWk[i] + l2RegularizationFactor * Wk[i])
        }
        
        for i in 0..<Wq.count {
            Wq[i] -= learningRate * (gradWq[i] + l2RegularizationFactor * Wq[i])
        }
        
        for i in 0..<bf.count {
            bf[i] -= learningRate * (gradbf[i] + l2RegularizationFactor * bf[i])
        }
        
        for i in 0..<bi.count {
            bi[i] -= learningRate * (gradbi[i] + l2RegularizationFactor * bi[i])
        }
        
        for i in 0..<bo.count {
            bo[i] -= learningRate * (gradbo[i] + l2RegularizationFactor * bo[i])
        }
        
        
        
        
    }
    
    // Utility methods
    private func computeExponentialGate(_ W: [Float], _ concat: [Float], _ b: [Float]) -> [Float] {
        return W.indices.map { i in
            exp(W[i] * concat[i % concat.count] + b[i % b.count])
        }
    }
    
    private func computeSigmoidGate(_ W: [Float], _ concat: [Float], _ b: [Float]) -> [Float] {
        return W.indices.map { i in
            1.0 / (1.0 + exp(-W[i] * concat[i % concat.count] - b[i % b.count]))
        }
    }
    
    
    enum LossFunction {
        case meanSquaredError
        case crossEntropy
        case huberloss
        
        func computeGradient(predicted: [Float], target: [Float]) -> [Float] {
            switch self {
            case .meanSquaredError:
                return zip(predicted, target).map { 2 * ($0 - $1) }
            case .crossEntropy:
                return zip(predicted, target).map { pred, actual in
                    let epsilon: Float = 1e-15
                    let clippedPred = max(min(pred, 1 - epsilon), epsilon)
                    return -(actual / clippedPred - (1 - actual) / (1 - clippedPred))
                }
            case .huberloss:
                let delta: Float = 1.0
                return zip(predicted, target).map { pred, actual in
                    let error = pred - actual
                    return abs(error) <= delta ? error : (delta * (error > 0 ? 1 : -1))
                }
            }
        }
    }
    
    // Advanced optimization techniques
    enum OptimizerType {
        case adam
        case rmsprop
        case adagrad
        
        struct Hyperparameters {
            let learningRate: Float
            let beta1: Float = 0.9
            let beta2: Float = 0.999
            let epsilon: Float = 1e-8
        }
    }
    
    // Comprehensive gradient accumulation
    private func accumulateGradients(dh: [Float], input: [Float]) {
        // Detailed gradient computation for weights and biases
        let concat = h + input
        
        // Gradient for forget gate weights
        gradWf = computeGateGradient(dh, Wf, concat, bf)
        
        // Gradient for input gate weights
        gradWi = computeGateGradient(dh, Wi, concat, bi)
        
        // Gradient for output gate weights
        gradWo = computeGateGradient(dh, Wo, concat, bo)
        
        // Gradient for value, key, query weights
        gradWv = computeValueGradient(dh, input)
        gradWk = computeKeyGradient(dh, input)
        gradWq = computeQueryGradient(dh, input)
        
        // Bias gradients
        gradbf = dh
        gradbi = dh
        gradbo = dh
    }
    
    // Advanced Adam optimization method
    private func optimizeWithAdam(
        hyperparameters: OptimizerType.Hyperparameters = .init(learningRate: 0.001)
    ) {
        var mW = [Float](repeating: 0, count: Wf.count)
        var vW = [Float](repeating: 0, count: Wf.count)
        let t = 1
        
        func adamUpdate(_ weights: inout [Float], _ gradients: [Float], _ m: inout [Float], _ v: inout [Float]) {
            for i in 0..<weights.count {
                // Momentum update
                m[i] = hyperparameters.beta1 * m[i] + (1 - hyperparameters.beta1) * gradients[i]
                
                // RMSprop-like update
                v[i] = hyperparameters.beta2 * v[i] + (1 - hyperparameters.beta2) * gradients[i] * gradients[i]
                
                // Bias correction
                let mHat = m[i] / (1 - pow(hyperparameters.beta1, Float(t)))
                let vHat = v[i] / (1 - pow(hyperparameters.beta2, Float(t)))
                
                // Parameter update
                weights[i] -= hyperparameters.learningRate * mHat / (sqrt(vHat) + hyperparameters.epsilon)
            }
        }
        
        // Apply Adam to different weight matrices
        adamUpdate(&Wf, gradWf, &mW, &vW)
        adamUpdate(&Wi, gradWi, &mW, &vW)
        adamUpdate(&Wo, gradWo, &mW, &vW)
        adamUpdate(&Wv, gradWv, &mW, &vW)
        adamUpdate(&Wk, gradWk, &mW, &vW)
        adamUpdate(&Wq, gradWq, &mW, &vW)
    }
    
    // Detailed gradient computation methods
    private func computeGateGradient(_ dh: [Float], _ W: [Float], _ concat: [Float], _ b: [Float]) -> [Float] {
        return W.indices.map { i in
            dh[i % dh.count] * concat[i % concat.count] * (1 - b[i % b.count])
        }
    }
    
    private func computeValueGradient(_ dh: [Float], _ input: [Float]) -> [Float] {
        return input.indices.map { i in
            dh.reduce(0) { $0 + $1 * input[i] }
        }
    }
    
    private func computeKeyGradient(_ dh: [Float], _ input: [Float]) -> [Float] {
        return input.indices.map { i in
            dh.reduce(0) { $0 + $1 * input[i] }
        }
    }
    
    private func computeQueryGradient(_ dh: [Float], _ input: [Float]) -> [Float] {
        return input.indices.map { i in
            dh.reduce(0) { $0 + $1 * input[i] }
        }
    }
    
    // Updated backpropagation method
    func backpropagate(targetSequence: [[Float]], lossType: LossFunction = .meanSquaredError) {
        var dhnext = [Float](repeating: 0, count: hiddenSize)
        
        for (index, targetOutput) in targetSequence.reversed().enumerated() {
            let currentOutput = lastOutputs[lastOutputs.count - 1 - index]
            let currentInput = lastInput
            
            // Compute output error using selected loss function
            let dh = lossType.computeGradient(predicted: currentOutput, target: targetOutput)
            
            // Gradient computation for gates and weights
            accumulateGradients(dh: dh, input: currentInput)
            
            dhnext = dh
        }
        
        // Gradient clipping
        applyGradientClipping()
        
        // Optimize weights using Adam
        optimizeWithAdam()
    }
    
    private func updateMemoryCell(ft: [Float], it: [Float], vt: [Float], kt: [Float]) {
        for i in 0..<memorySize {
            for j in 0..<memorySize {
                // Update memory cell using forget and input gates
                C[i * memorySize + j] = ft[i] * C[i * memorySize + j] + it[i] * vt[i] * kt[j]
            }
        }
    }
    
    // Compute hidden state with normalization
    private func computeHiddenState(ot: [Float], qt: [Float]) -> [Float] {
        // Compute weighted sum of memory cell with query vector
        let Ctq = (0..<memorySize).map { i in
            (0..<memorySize).reduce(0) { result, j in
                result + C[i * memorySize + j] * qt[j]
            }
        }
        
        // Normalization to prevent exploding values
        let norm = max(Ctq.reduce(0, +), 1e-7)
        
        // Apply output gate and normalize
        return Ctq.enumerated().map { (index, value) in
            ot[index] * (value / norm)
        }
    }
}





