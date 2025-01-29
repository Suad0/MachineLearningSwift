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
    
    required init(
        inputSize: Int,
        hiddenSize: Int,
        memorySize: Int,
        learningRate: Float = 0.001,
        l2RegFactor: Float = 1e-5,
        gradClipThreshold: Float = 1.0
    ) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.memorySize = memorySize
        self.learningRate = learningRate
        self.l2RegularizationFactor = l2RegFactor
        self.gradientClippingThreshold = gradClipThreshold
        
        func xavierInitialize(inputSize: Int, outputSize: Int) -> [Float] {
            let limit = sqrt(6.0 / Float(inputSize + outputSize))
            return (0..<inputSize * outputSize).map { _ in Float.random(in: -limit...limit) }
        }
        
        // Initialize weights with correct dimensions
        Wf = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: memorySize)
        Wi = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: memorySize)
        Wo = xavierInitialize(inputSize: inputSize + hiddenSize, outputSize: hiddenSize) // Now hiddenSize
        Wv = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        Wk = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        Wq = xavierInitialize(inputSize: inputSize, outputSize: memorySize)
        
        // Initialize biases with correct sizes
        bf = [Float](repeating: 0, count: memorySize)
        bi = [Float](repeating: 0, count: memorySize)
        bo = [Float](repeating: 0, count: hiddenSize) // Now hiddenSize
        
        // Initialize gradient storage (unchanged)
        gradWf = [Float](repeating: 0, count: Wf.count)
        gradWi = [Float](repeating: 0, count: Wi.count)
        gradWo = [Float](repeating: 0, count: Wo.count)
        gradWv = [Float](repeating: 0, count: Wv.count)
        gradWk = [Float](repeating: 0, count: Wk.count)
        gradWq = [Float](repeating: 0, count: Wq.count)
        gradbf = [Float](repeating: 0, count: bf.count)
        gradbi = [Float](repeating: 0, count: bi.count)
        gradbo = [Float](repeating: 0, count: bo.count)
        
        // Initialize hidden state and memory cell
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
        
        // Corrected gate activations (sigmoid)
        let ft = computeGate(Wf, concat, bf, rowsA: memorySize, activation: { 1.0 / (1.0 + exp(-$0)) })
        let it = computeGate(Wi, concat, bi, rowsA: memorySize, activation: { 1.0 / (1.0 + exp(-$0)) })
        let ot = computeGate(Wo, concat, bo, rowsA: hiddenSize, activation: { 1.0 / (1.0 + exp(-$0)) })
        
        // Bounded value/key/query with tanh
        let vt = tanh(matrixMultiply(Wv, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        let kt = tanh(matrixMultiply(Wk, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        let qt = tanh(matrixMultiply(Wq, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        
        updateMemoryCell(ft: ft, it: it, vt: vt, kt: kt)
        
        // Improved normalization with L2 norm
        h = computeHiddenState(ot: ot, qt: qt)
        return h
    }

    private func computeHiddenState(ot: [Float], qt: [Float]) -> [Float] {
        let Ctq = (0..<memorySize).map { i in
            (0..<memorySize).reduce(0) { result, j in
                result + C[i * memorySize + j] * qt[j]
            }
        }
        let norm = sqrt(Ctq.reduce(0) { $0 + $1 * $1 }) + 1e-7 // L2 norm
        return zip(ot, Ctq).map { (gate, value) in
            gate * (value / norm)
        }
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
    private func computeExponentialGate(_ W: [Float], _ concat: [Float], _ b: [Float], rowsA: Int) -> [Float] {
        return computeGate(W, concat, b, rowsA: rowsA, activation: exp)
    }
    
    private func computeSigmoidGate(_ W: [Float], _ concat: [Float], _ b: [Float], rowsA: Int) -> [Float] {
        return computeGate(W, concat, b, rowsA: rowsA, activation: { 1.0 / (1.0 + exp(-$0)) })
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
        
        // Gradient for memory matrix C
        var dC = [Float](repeating: 0, count: memorySize * memorySize)
        for i in 0..<memorySize {
            for j in 0..<memorySize {
                dC[i * memorySize + j] = dh[i] * C[i * memorySize + j]
            }
        }
        
        // Accumulate gradients for Wv, Wk, Wq based on dC
        // (This is a simplified example; you may need to adjust based on your specific architecture)
        gradWv = matrixMultiply(dC, input, rowsA: memorySize, colsA: memorySize, colsB: inputSize)
        gradWk = matrixMultiply(dC, input, rowsA: memorySize, colsA: memorySize, colsB: inputSize)
        gradWq = matrixMultiply(dC, input, rowsA: memorySize, colsA: memorySize, colsB: inputSize)
    }
    
    // Advanced Adam optimization method
    private func optimizeWithAdam(hyperparameters: OptimizerType.Hyperparameters = .init(learningRate: 0.001)) {
        var mWf = [Float](repeating: 0, count: Wf.count)
        var vWf = [Float](repeating: 0, count: Wf.count)
        var mWi = [Float](repeating: 0, count: Wi.count)
        var vWi = [Float](repeating: 0, count: Wi.count)
        var mWo = [Float](repeating: 0, count: Wo.count)
        var vWo = [Float](repeating: 0, count: Wo.count)
        var mWv = [Float](repeating: 0, count: Wv.count)
        var vWv = [Float](repeating: 0, count: Wv.count)
        var mWk = [Float](repeating: 0, count: Wk.count)
        var vWk = [Float](repeating: 0, count: Wk.count)
        var mWq = [Float](repeating: 0, count: Wq.count)
        var vWq = [Float](repeating: 0, count: Wq.count)
        
        var t = 1
        
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
        adamUpdate(&Wf, gradWf, &mWf, &vWf)
        adamUpdate(&Wi, gradWi, &mWi, &vWi)
        adamUpdate(&Wo, gradWo, &mWo, &vWo)
        adamUpdate(&Wv, gradWv, &mWv, &vWv)
        adamUpdate(&Wk, gradWk, &mWk, &vWk)
        adamUpdate(&Wq, gradWq, &mWq, &vWq)
        
        t += 1
    }
    
    // Detailed gradient computation methods
    private func computeGateGradient(_ dh: [Float], _ W: [Float], _ concat: [Float], _ b: [Float]) -> [Float] {
        let rowsA = W.count / (inputSize + hiddenSize) // Calculate based on weight matrix dimensions
        let gateOutput = computeSigmoidGate(W, concat, b, rowsA: rowsA)
        return zip(gateOutput, dh).map { gate, grad in
            gate * (1 - gate) * grad
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
        // Debug: Print sizes of input arrays
        print("ft size: \(ft.count), it size: \(it.count), vt size: \(vt.count), kt size: \(kt.count)")
        print("C size: \(C.count), memorySize: \(memorySize)")
        
        // Ensure all arrays have the correct size
        guard ft.count == memorySize,
              it.count == memorySize,
              vt.count == memorySize,
              kt.count == memorySize,
              C.count == memorySize * memorySize else {
            fatalError("Array size mismatch in updateMemoryCell")
        }
        
        // Update memory cell using forget and input gates
        for i in 0..<memorySize {
            for j in 0..<memorySize {
                C[i * memorySize + j] = ft[i] * C[i * memorySize + j] + it[i] * vt[i] * kt[j]
            }
        }
    }
    
    
    
    private func matrixMultiply(_ A: [Float], _ B: [Float], rowsA: Int, colsA: Int, colsB: Int) -> [Float] {
        var result = [Float](repeating: 0, count: rowsA * colsB)
        vDSP_mmul(A, 1, B, 1, &result, 1, vDSP_Length(rowsA), vDSP_Length(colsB), vDSP_Length(colsA))
        return result
    }
    
    private func computeGate(
        _ W: [Float],
        _ concat: [Float],
        _ b: [Float],
        rowsA: Int,
        activation: (Float) -> Float
    ) -> [Float] {
        let weightedSum = matrixMultiply(W, concat, rowsA: rowsA, colsA: inputSize + hiddenSize, colsB: 1)
        return zip(weightedSum, b).map { activation($0 + $1) }
    }
     
    
    
    
    
    
}







