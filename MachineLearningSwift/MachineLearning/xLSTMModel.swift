//
//  xLSTMModel.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 08.04.25.
//

import Foundation
import Accelerate

class xLSTMModel {
    
    private let cell: xLSTMCell
    private let outputSize: Int
    private var outputWeights: [Float]
    private var outputBias: [Float]
    
    // Optimizer states for output layer (Adam)
    private var mOutputWeights: [Float]
    private var vOutputWeights: [Float]
    private var mOutputBias: [Float]
    private var vOutputBias: [Float]
    private var t: Int = 1
    
    // Hyperparameters
    private let learningRate: Float
    private let adamHyperparams: (beta1: Float, beta2: Float, epsilon: Float, learningRate: Float)
    
    init(
        inputSize: Int,
        hiddenSize: Int,
        memorySize: Int,
        outputSize: Int,
        learningRate: Float = 0.001,
        l2RegFactor: Float = 1e-5,
        gradClipThreshold: Float = 1.0
    ) {
        self.cell = xLSTMCell(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            memorySize: memorySize,
            learningRate: learningRate,
            l2RegFactor: l2RegFactor,
            gradClipThreshold: gradClipThreshold
        )
        self.outputSize = outputSize
        self.learningRate = learningRate
        self.adamHyperparams = (beta1: 0.9, beta2: 0.999, epsilon: 1e-8, learningRate: learningRate)
        
        // Initialize output layer weights and biases
        let outputLayerSize = hiddenSize * outputSize
        self.outputWeights = (0..<outputLayerSize).map { _ in Float.random(in: -0.1...0.1) }
        self.outputBias = [Float](repeating: 0, count: outputSize)
        
        // Initialize Adam optimizer states
        self.mOutputWeights = [Float](repeating: 0, count: outputLayerSize)
        self.vOutputWeights = [Float](repeating: 0, count: outputLayerSize)
        self.mOutputBias = [Float](repeating: 0, count: outputSize)
        self.vOutputBias = [Float](repeating: 0, count: outputSize)
    }
    
    func predict(sequence: [[Float]]) -> [Float] {
        // Process the sequence through the xLSTMCell to get hidden states
        let hiddenStates = cell.processSequence(inputs: sequence)
        // Use the last hidden state for output
        let lastHidden = hiddenStates.last ?? [Float](repeating: 0, count: cell.hiddenSize)
        return computeOutput(hiddenState: lastHidden)
    }
    
    private func computeOutput(hiddenState: [Float]) -> [Float] {
        // Matrix multiplication: outputWeights (outputSize x hiddenSize) * hiddenState (hiddenSize x 1)
        var result = [Float](repeating: 0, count: outputSize)
        vDSP_mmul(
            outputWeights, 1,
            hiddenState, 1,
            &result, 1,
            vDSP_Length(outputSize),
            vDSP_Length(1),
            vDSP_Length(cell.hiddenSize)
        )
        // Add bias
        return zip(result, outputBias).map { $0 + $1 }
    }
    
    func train(sequence: [[Float]], target: [Float], lossType: xLSTMCell.LossFunction = .meanSquaredError) {
        // Forward pass
        let prediction = predict(sequence: sequence)
        
        // Compute loss gradient
        let outputGradient = lossType.computeGradient(predicted: prediction, target: target)
        
        // Get the last hidden state from the cell's lastOutputs
        let lastHidden = cell.lastOutputs.last ?? [Float](repeating: 0, count: cell.hiddenSize)
        
        // Backpropagate through output layer and get gradient for hidden state
        let hiddenGradient = backpropagateOutputLayer(outputGradient: outputGradient, lastHidden: lastHidden)
        
        // Backpropagate through xLSTMCell
        cell.backpropagate(targetSequence: [hiddenGradient], lossType: lossType)
        
        // Increment timestep for Adam
        t += 1
    }
    
    private func backpropagateOutputLayer(outputGradient: [Float], lastHidden: [Float]) -> [Float] {
        // Compute gradients for weights and biases
        var gradOutputWeights = [Float](repeating: 0, count: outputWeights.count)
        var gradOutputBias = [Float](repeating: 0, count: outputBias.count)
        
        for i in 0..<outputSize {
            for j in 0..<cell.hiddenSize {
                gradOutputWeights[i * cell.hiddenSize + j] = outputGradient[i] * lastHidden[j]
            }
            gradOutputBias[i] = outputGradient[i]
        }
        
        // Compute gradient for hidden state
        var hiddenGradient = [Float](repeating: 0, count: cell.hiddenSize)
        for j in 0..<cell.hiddenSize {
            for i in 0..<outputSize {
                hiddenGradient[j] += outputGradient[i] * outputWeights[i * cell.hiddenSize + j]
            }
        }
        
        // Update output layer parameters with Adam
        adamUpdate(&outputWeights, gradOutputWeights, &mOutputWeights, &vOutputWeights)
        adamUpdate(&outputBias, gradOutputBias, &mOutputBias, &vOutputBias)
        
        return hiddenGradient
    }
    
    private func adamUpdate(_ weights: inout [Float], _ gradients: [Float], _ m: inout [Float], _ v: inout [Float]) {
        let (beta1, beta2, epsilon, lr) = adamHyperparams
        for i in 0..<weights.count {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i]
            
            let mHat = m[i] / (1 - pow(beta1, Float(t)))
            let vHat = v[i] / (1 - pow(beta2, Float(t)))
            
            weights[i] -= lr * mHat / (sqrt(vHat) + epsilon)
        }
    }
    
    func save(to url: URL) throws {
            let data: [String: Any] = [
                "inputSize": cell.inputSize,
                "hiddenSize": cell.hiddenSize,
                "memorySize": cell.memorySize,
                "outputSize": outputSize,
                "learningRate": learningRate,
                "outputWeights": outputWeights,
                "outputBias": outputBias,
                "mOutputWeights": mOutputWeights,
                "vOutputWeights": vOutputWeights,
                "mOutputBias": mOutputBias,
                "vOutputBias": vOutputBias,
                "t": t,
                "cell": cell.save()
            ]
            let jsonData = try JSONSerialization.data(withJSONObject: data, options: .prettyPrinted)
            try jsonData.write(to: url)
        }

        static func load(from url: URL) throws -> xLSTMModel {
            let jsonData = try Data(contentsOf: url)
            let data = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
            
            let model = xLSTMModel(
                inputSize: data["inputSize"] as! Int,
                hiddenSize: data["hiddenSize"] as! Int,
                memorySize: data["memorySize"] as! Int,
                outputSize: data["outputSize"] as! Int,
                learningRate: data["learningRate"] as! Float
            )
            model.outputWeights = data["outputWeights"] as! [Float]
            model.outputBias = data["outputBias"] as! [Float]
            model.mOutputWeights = data["mOutputWeights"] as! [Float]
            model.vOutputWeights = data["vOutputWeights"] as! [Float]
            model.mOutputBias = data["mOutputBias"] as! [Float]
            model.vOutputBias = data["vOutputBias"] as! [Float]
            model.t = data["t"] as! Int
            model.cell.load(from: data["cell"] as! [String: Any])
            return model
        }
}




