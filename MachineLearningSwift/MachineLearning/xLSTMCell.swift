//
//  xLSTMCell.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.01.25.
//




import Foundation
import Accelerate

class xLSTMCell {
    
    public let inputSize: Int
    public let hiddenSize: Int
    public let memorySize: Int
    
    // Weights and biases
    private var Wf, Wi, Wo, Wv, Wk, Wq: [Float]
    private var bf, bi, bo: [Float]
    
    
    private var mWf, vWf, mWi, vWi, mWo, vWo, mWv, vWv, mWk, vWk, mWq, vWq: [Float]
    private var t: Int = 1
    
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
    public var lastOutputs: [[Float]] = []
    
    private var states: [(input: [Float], concat: [Float], ft: [Float], it: [Float], ot: [Float], vt: [Float], kt: [Float], qt: [Float], C: [Float])] = []
    
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
        
        mWf = [Float](repeating: 0, count: Wf.count)
        vWf = [Float](repeating: 0, count: Wf.count)
        mWi = [Float](repeating: 0, count: Wi.count)
        vWi = [Float](repeating: 0, count: Wi.count)
        mWo = [Float](repeating: 0, count: Wo.count)
        vWo = [Float](repeating: 0, count: Wo.count)
        mWv = [Float](repeating: 0, count: Wv.count)
        vWv = [Float](repeating: 0, count: Wv.count)
        mWk = [Float](repeating: 0, count: Wk.count)
        vWk = [Float](repeating: 0, count: Wk.count)
        mWq = [Float](repeating: 0, count: Wq.count)
        vWq = [Float](repeating: 0, count: Wq.count)
        
        
        
    }
    
    // Sequence processing forward pass
    func processSequence(inputs: [[Float]]) -> [[Float]] {
        states = []  // Reset states for each sequence
        var outputs: [[Float]] = []
        for input in inputs {
            let output = forward(input: input)
            outputs.append(output)
        }
        lastOutputs = outputs
        return outputs
    }
    
    // Forward pass with detailed state tracking
    
    private func forward(input: [Float]) -> [Float] {
        precondition(input.count == inputSize, "Input size must be \(inputSize), got \(input.count)")
        precondition(h.count == hiddenSize, "Hidden state size corrupted")
        
        let concat = h + input
        let ft = computeGate(Wf, concat, bf, rowsA: memorySize, activation: { 1.0 / (1.0 + exp(-$0)) })
        let it = computeGate(Wi, concat, bi, rowsA: memorySize, activation: { 1.0 / (1.0 + exp(-$0)) })
        let ot = computeGate(Wo, concat, bo, rowsA: hiddenSize, activation: { 1.0 / (1.0 + exp(-$0)) })
        let vt = tanh(matrixMultiply(Wv, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        let kt = tanh(matrixMultiply(Wk, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        let qt = tanh(matrixMultiply(Wq, input, rowsA: memorySize, colsA: inputSize, colsB: 1))
        
        // Store C before update as C_prev would be the previous C, but we’ll use states[t-1].C later
        updateMemoryCell(ft: ft, it: it, vt: vt, kt: kt)
        h = computeHiddenState(ot: ot, qt: qt)
        
        // Store all necessary states
        states.append((input: input, concat: concat, ft: ft, it: it, ot: ot, vt: vt, kt: kt, qt: qt, C: C))
        return h
    }
    
    private func computeHiddenState(ot: [Float], qt: [Float]) -> [Float] {
        
        guard ot.count == hiddenSize, qt.count == memorySize else {
            fatalError("Dimension mismatch in computeHiddenState")
        }
        
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
    /*
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
     */
    
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
    private func accumulateGradients(
        dh: [Float],
        state: (input: [Float], concat: [Float], ft: [Float], it: [Float], ot: [Float], vt: [Float], kt: [Float], qt: [Float], C: [Float]),
        dCnext: [Float],
        C_prev: [Float]
    ) -> [Float] {
        let input = state.input
        let concat = state.concat
        let ft = state.ft
        let it = state.it
        let ot = state.ot
        let vt = state.vt
        let kt = state.kt
        let qt = state.qt
        let C = state.C  // C_t after update
        
        // Step 1: Compute Ctq and norm (recompute from forward pass)
        let Ctq = (0..<memorySize).map { i in
            (0..<memorySize).reduce(0) { result, j in
                result + C[i * memorySize + j] * qt[j]
            }
        }
        let norm = sqrt(Ctq.reduce(0) { $0 + $1 * $1 }) + 1e-7
        let s = Ctq.map { $0 / norm }  // s = Ctq / norm
        
        // Step 2: Compute ds (gradient w.r.t. s), since h = ot .* s
        let ds = zip(ot, dh).map { $0 * $1 }  // ∂L/∂s = ot .* dh
        
        // Step 3: Compute dCtq (gradient w.r.t. Ctq)
        let Ctq_dot_ds = zip(Ctq, ds).reduce(0) { $0 + $1.0 * $1.1 }
        let dCtq = (0..<memorySize).map { k in
            ds[k] / norm - (Ctq_dot_ds * Ctq[k]) / (norm * norm * norm)
        }
        
        // Step 4: Compute dC (gradient w.r.t. C_t) from Ctq = C @ qt
        var dC = (0..<memorySize).flatMap { i in
            (0..<memorySize).map { j in
                dCtq[i] * qt[j]  // dC[i,j] = dCtq[i] * qt[j]
            }
        }
        
        // Add contribution from dCnext (from the next time step)
        dC = zip(dC, dCnext).map { $0 + $1 }
        
        // Step 5: Compute dot (gradient w.r.t. ot), since h = ot .* s
        let dot = zip(s, dh).map { $0 * $1 }
        let dz_o = zip(dot, ot).map { $0 * $1 * (1 - $1) }  // Sigmoid derivative: ot * (1 - ot)
        let gradWo_update = outerProduct(dz_o, concat)
        gradWo = zip(gradWo, gradWo_update).map { $0 + $1 }
        gradbo = zip(gradbo, dz_o).map { $0 + $1 }
        
        // Step 6: Compute gradients for ft, it, vt, kt from C_t = ft .* C_{t-1} + it .* (vt @ kt.T)
        var dft = [Float](repeating: 0, count: memorySize)
        for i in 0..<memorySize {
            var sum = Float(0)
            for j in 0..<memorySize {
                sum += dC[i * memorySize + j] * C_prev[i * memorySize + j]
            }
            dft[i] = sum
        }
        
        let dz_f = zip(dft, ft).map { $0 * $1 * (1 - $1) }
        gradWf = zip(gradWf, outerProduct(dz_f, concat)).map { $0 + $1 }
        gradbf = zip(gradbf, dz_f).map { $0 + $1 }
        
        let dit = (0..<memorySize).map { i in
            vt[i] * (0..<memorySize).reduce(0) { $0 + dC[i * memorySize + $1] * kt[$1] }
        }
        let dz_i = zip(dit, it).map { $0 * $1 * (1 - $1) }
        gradWi = zip(gradWi, outerProduct(dz_i, concat)).map { $0 + $1 }
        gradbi = zip(gradbi, dz_i).map { $0 + $1 }
        
        let dvt = (0..<memorySize).map { i in
            it[i] * (0..<memorySize).reduce(0) { $0 + dC[i * memorySize + $1] * kt[$1] }
        }
        let dz_v = zip(dvt, vt).map { $0 * (1 - $1 * $1) }  // Tanh derivative: 1 - vt^2
        gradWv = zip(gradWv, outerProduct(dz_v, input)).map { $0 + $1 }
        // Add if gradbv exists: gradbv = zip(gradbv, dz_v).map { $0 + $1 }
        
        let dkt = (0..<memorySize).map { j in
            (0..<memorySize).reduce(0) { $0 + dC[$1 * memorySize + j] * it[$1] * vt[$1] }
        }
        let dz_k = zip(dkt, kt).map { $0 * (1 - $1 * $1) }
        gradWk = zip(gradWk, outerProduct(dz_k, input)).map { $0 + $1 }
        // Add if gradbk exists: gradbk = zip(gradbk, dz_k).map { $0 + $1 }
        
        // Step 7: Compute dqt and update Wq
        let dqt = (0..<memorySize).map { j in
            (0..<memorySize).reduce(0) { $0 + C[$1 * memorySize + j] * dCtq[$1] }
        }
        let dz_q = zip(dqt, qt).map { $0 * (1 - $1 * $1) }
        gradWq = zip(gradWq, outerProduct(dz_q, input)).map { $0 + $1 }
        
        return dC
        
        
    }
    
    private func outerProduct(_ a: [Float], _ b: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: a.count * b.count)
        for i in 0..<a.count {
            for j in 0..<b.count {
                result[i * b.count + j] = a[i] * b[j]
            }
        }
        return result
    }
    
    // Advanced Adam optimization method
    private func optimizeWithAdam(hyperparameters: OptimizerType.Hyperparameters = .init(learningRate: 0.001)) {
        //var mWf = [Float](repeating: 0, count: Wf.count)
        //var vWf = [Float](repeating: 0, count: Wf.count)
        //var mWi = [Float](repeating: 0, count: Wi.count)
        //var vWi = [Float](repeating: 0, count: Wi.count)
        //var mWo = [Float](repeating: 0, count: Wo.count)
        //var vWo = [Float](repeating: 0, count: Wo.count)
        //var mWv = [Float](repeating: 0, count: Wv.count)
        //var vWv = [Float](repeating: 0, count: Wv.count)
        //var mWk = [Float](repeating: 0, count: Wk.count)
        //var vWk = [Float](repeating: 0, count: Wk.count)
        //var mWq = [Float](repeating: 0, count: Wq.count)
        //var vWq = [Float](repeating: 0, count: Wq.count)
        
        //var t = 1
        
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
    func backpropagate(dhSequence: [[Float]]) {
        precondition(dhSequence.count == states.count, "Gradient sequence length must match input sequence length")
        var dhnext = [Float](repeating: 0, count: hiddenSize)
        var dCnext = [Float](repeating: 0, count: memorySize * memorySize)
        
        for (t, externalDh) in dhSequence.reversed().enumerated() {
            let state = states[states.count - 1 - t]
            let dh = zip(externalDh, dhnext).map { $0 + $1 }
            let C_prev = (states.count - 1 - t) > 0 ? states[states.count - 2 - t].C : [Float](repeating: 0, count: memorySize * memorySize)
            let dC = accumulateGradients(dh: dh, state: state, dCnext: dCnext, C_prev: C_prev)
            dhnext = dh
            if (states.count - 1 - t) > 0 {
                dCnext = (0..<memorySize).flatMap { i in
                    (0..<memorySize).map { j in
                        dC[i * memorySize + j] * state.ft[i]
                    }
                }
            } else {
                dCnext = [Float](repeating: 0, count: memorySize * memorySize)
            }
        }
        
        applyGradientClipping()
        optimizeWithAdam()
    }
    
    private func updateMemoryCell(ft: [Float], it: [Float], vt: [Float], kt: [Float]) {
        guard ft.count == memorySize,
              it.count == memorySize,
              vt.count == memorySize,
              kt.count == memorySize,
              C.count == memorySize * memorySize else {
            fatalError("Array size mismatch in updateMemoryCell")
        }
        
        // Create a local mutable copy of C so that we can modify it without conflicting with self.C.
        var newC = C
        
        // Compute vt @ kt.T into temp matrix.
        var temp = [Float](repeating: 0, count: memorySize * memorySize)
        vDSP_mmul(vt, 1, kt, 1, &temp, 1, vDSP_Length(memorySize), 1, vDSP_Length(memorySize))
        
        // Update newC = ft .* newC (row-wise)
        newC.withUnsafeMutableBufferPointer { cBuffer in
            for i in 0..<memorySize {
                let rowStart = i * memorySize
                let rowPtr = cBuffer.baseAddress! + rowStart
                var scalar_ft = ft[i]
                withUnsafePointer(to: &scalar_ft) { scalePtr in
                    vDSP_vsmul(rowPtr, 1, scalePtr, rowPtr, 1, vDSP_Length(memorySize))
                }
            }
        }
        
        // Update temp = it .* (vt @ kt.T) (row-wise)
        temp.withUnsafeMutableBufferPointer { tempBuffer in
            for i in 0..<memorySize {
                let rowStart = i * memorySize
                let rowPtr = tempBuffer.baseAddress! + rowStart
                var scalar_it = it[i]
                withUnsafePointer(to: &scalar_it) { scalePtr in
                    vDSP_vsmul(rowPtr, 1, scalePtr, rowPtr, 1, vDSP_Length(memorySize))
                }
            }
        }
        
        // Create a temporary array to hold the result
        var result = [Float](repeating: 0, count: newC.count)

        newC.withUnsafeBufferPointer { cBuffer in
            temp.withUnsafeBufferPointer { tempBuffer in
                result.withUnsafeMutableBufferPointer { resBuffer in
                    vDSP_vadd(cBuffer.baseAddress!, 1,
                              tempBuffer.baseAddress!, 1,
                              resBuffer.baseAddress!, 1,
                              vDSP_Length(newC.count))
                }
            }
        }

        // Now assign the computed result back to newC.
        newC = result
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


// TODO ONLY FOR TESTING 
extension xLSTMCell {
    func save() -> [String: Any] {
        return [
            "Wf": Wf,
            "Wi": Wi,
            "Wo": Wo,
            "Wv": Wv,
            "Wk": Wk,
            "Wq": Wq,
            "bf": bf,
            "bi": bi,
            "bo": bo
        ]
    }
    
    func load(from data: [String: Any]) {
        Wf = data["Wf"] as! [Float]
        Wi = data["Wi"] as! [Float]
        Wo = data["Wo"] as! [Float]
        Wv = data["Wv"] as! [Float]
        Wk = data["Wk"] as! [Float]
        Wq = data["Wq"] as! [Float]
        bf = data["bf"] as! [Float]
        bi = data["bi"] as! [Float]
        bo = data["bo"] as! [Float]
    }
}









