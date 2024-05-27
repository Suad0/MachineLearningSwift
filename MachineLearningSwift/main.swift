//
//  main.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation


let inputSize = 2
let hiddenSize = 3
let outputSize = 1

let neuralNetwork = NeuralNetwork(inputSize: inputSize, hiddenSize: hiddenSize, outputSize: outputSize)

let inputs: [[Double]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
let targets: [[Double]] = [[0], [1], [1], [0]]

// Train the neural network
let epochs = 10
let learningRate = 0.1
neuralNetwork.train(inputs, targets, epochs: epochs, learningRate: learningRate)

// Test the neural network
print("Testing neural network predictions:")
for input in inputs {
    let predicted = neuralNetwork.predict(input)
    print("Input: \(input) => Predicted: \(predicted)")
}



