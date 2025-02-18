//
//  NeuralNetwork.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation

class NeuralNetwork {
    
    private var inputSize: Int
    private var hiddenSize: Int
    private var outputSize: Int
    private var weightsInputHidden: [[Double]]
    private var weightsHiddenOutput: [[Double]]
    private var biasesHidden: [Double]
    private var biasesOutput: [Double]
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        self.weightsInputHidden = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        self.weightsHiddenOutput = Array(repeating: Array(repeating: 0.0, count: outputSize), count: hiddenSize)
        self.biasesHidden = Array(repeating: 0.0, count: hiddenSize)
        self.biasesOutput = Array(repeating: 0.0, count: outputSize)
        
        initializeWeightsAndBiases()
    }
    
    private func initializeWeightsAndBiases() {
        // Initialize weights randomly
        for i in 0..<inputSize {
            for j in 0..<hiddenSize {
                weightsInputHidden[i][j] = Double.random(in: -0.5...0.5)
            }
        }
        for i in 0..<hiddenSize {
            for j in 0..<outputSize {
                weightsHiddenOutput[i][j] = Double.random(in: -0.5...0.5)
            }
        }
        
        // Initialize biases to zeros or small random values
        for i in 0..<hiddenSize {
            biasesHidden[i] = 0.0
        }
        for i in 0..<outputSize {
            biasesOutput[i] = 0.0
        }
    }
    
    private func sigmoid(_ x: [Double]) -> [Double] {
        return x.map { 1 / (1 + exp(-$0)) }
    }
    
    public func predict(_ input: [Double]) -> [Double] {
        // Forward propagation
        let hiddenLayerOutput = sigmoid(calculateHiddenLayerOutput(input))
        return sigmoid(calculateOutput(hiddenLayerOutput))
    }
    
    private func calculateHiddenLayerOutput(_ input: [Double]) -> [Double] {
        var hiddenLayerOutput = [Double](repeating: 0.0, count: hiddenSize)
        for i in 0..<hiddenSize {
            var sum = 0.0
            for j in 0..<inputSize {
                sum += input[j] * weightsInputHidden[j][i]
            }
            hiddenLayerOutput[i] = sum + biasesHidden[i]
        }
        return hiddenLayerOutput
    }
    
    private func calculateOutput(_ hiddenLayerOutput: [Double]) -> [Double] {
        var output = [Double](repeating: 0.0, count: outputSize)
        for i in 0..<outputSize {
            var sum = 0.0
            for j in 0..<hiddenSize {
                sum += hiddenLayerOutput[j] * weightsHiddenOutput[j][i]
            }
            output[i] = sum + biasesOutput[i]
        }
        return output
    }
    
    public func train(_ inputs: [[Double]], _ targets: [[Double]], epochs: Int, learningRate: Double) {
        for _ in 0..<epochs {
            for i in 0..<inputs.count {
                let input = inputs[i]
                let target = targets[i]
                
                // Forward propagation
                let hiddenLayerOutput = sigmoid(calculateHiddenLayerOutput(input))
                let output = sigmoid(calculateOutput(hiddenLayerOutput))
                
                // Backpropagation
                // Calculate output layer error
                var outputError = [Double](repeating: 0.0, count: outputSize)
                for j in 0..<outputSize {
                    outputError[j] = (target[j] - output[j]) * output[j] * (1 - output[j])
                }
                
                // Calculate hidden layer error
                var hiddenError = [Double](repeating: 0.0, count: hiddenSize)
                for j in 0..<hiddenSize {
                    var sum = 0.0
                    for k in 0..<outputSize {
                        sum += outputError[k] * weightsHiddenOutput[j][k]
                    }
                    hiddenError[j] = sum * hiddenLayerOutput[j] * (1 - hiddenLayerOutput[j])
                }
                
                // Update weights and biases
                for j in 0..<hiddenSize {
                    for k in 0..<outputSize {
                        weightsHiddenOutput[j][k] += learningRate * outputError[k] * hiddenLayerOutput[j]
                    }
                }
                for j in 0..<inputSize {
                    for k in 0..<hiddenSize {
                        weightsInputHidden[j][k] += learningRate * hiddenError[k] * input[j]
                    }
                }
                for j in 0..<outputSize {
                    biasesOutput[j] += learningRate * outputError[j]
                }
                for j in 0..<hiddenSize {
                    biasesHidden[j] += learningRate * hiddenError[j]
                }
            }
        }
    }
    
    
}
