//
//  main.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation

func testNeuralNetwork(inputSize: Int, hiddenSize: Int, outputSize: Int ){
    
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
    
    
}

func test_KNN(){
    let knn = KNearestNeighbors(k: 3)
    
    let trainingData: [[Double]] = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [6.0, 6.0],
        [7.0, 7.0],
        [8.0, 6.0]
    ]
    
    let trainingLabels: [Int] = [0, 0, 0, 1, 1, 1]
    
    knn.fit(data: trainingData, labels: trainingLabels)
    
    let testData: [[Double]] = [
        [1.5, 2.5],
        [6.5, 6.5]
    ]
    
    let predictions = knn.predict(data: testData)
    print(predictions) // Expected output: [0, 1]
}

func testDecisionTree(){
    
    let decisionTree = DecisionTree(maxDepth: 3)
    
    // Example training data
    let X_train: [[Double]] = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9]
    ]
    
    let y_train = [0, 0, 1, 1, 0, 1, 0, 1]
    
    decisionTree.train(X_train: X_train, y_train: y_train)
    
    let X_test: [[Double]] = [
        [1, 2],
        [5, 6],
        [8, 9]
    ]
    
    // Make predictions on test instances
    for instance in X_test {
        let prediction = decisionTree.predict(instance: instance)
        print("Prediction for \(instance): \(prediction)")
    }
    
    
    
}



func vectorStoreTest(){
    
    let docs = [
        "I like apples",
        "I like pears",
        "I like dogs",
        "I like cats"
    ]
    
    var vs = VectorStore(documents: docs)
    
    print(vs.getTopN(query: "I like apples", n: 1))
    print(vs.getTopN(query: "fruit", n: 2))
    
}


func testMetalNeuralNetwork(inputSize: Int, hiddenSize: Int, outputSize: Int) {
    
    
    let neuralNetwork = MetalNeuralNetwork(inputSize: inputSize, hiddenSize: hiddenSize, outputSize: outputSize)
    
    let inputs: [[Float]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    let targets: [[Float]] = [[0.0], [1.0], [1.0], [0.0]]
    
    // Train the neural network
    let epochs = 10
    let learningRate: Float = 0.1
    neuralNetwork.train(inputs, targets, epochs: epochs, learningRate: learningRate)
    
    // Test the neural network
    print("Testing neural network predictions:")
    for input in inputs {
        let predicted = neuralNetwork.predict(input)
        print("Input: \(input) => Predicted: \(predicted)")
    }
}

//testMetalNeuralNetwork(inputSize: 2, hiddenSize: 4, outputSize: 1)

func testLSTMCell() {
    
    // Example of using the LSTMCell
    let inputSize = 10 // For example, 10 features in the input
    let hiddenSize = 20 // Hidden layer size
    let lstm = LSTMCell(inputSize: inputSize, hiddenSize: hiddenSize)
    
    // Assume we have some input vector for a single time step
    let input = (0..<inputSize).map { _ in Float.random(in: -1...1) }
    
    // Forward pass through the LSTM
    let output = lstm.forward(input: input)
    
    print("LSTM output for the current time step:", output)
    
    
    
    
}

//testLSTMCell()

func testMetalLSTMCell() {
    
    let inputSize = 10   // For example, 10 features in the input
    let hiddenSize = 20  // Hidden layer size
    
    let lstm = MetalLSTMCell(inputSize: inputSize, hiddenSize: hiddenSize)
    
    // Assume we have some input vector for a single time step
    let inputArray = (0..<inputSize).map { _ in Float.random(in: -1...1) }
    
    // Convert input array to MTLBuffer
    let inputBuffer = lstm.device.makeBuffer(bytes: inputArray, length: inputSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
    
    // Forward pass through the LSTM
    let outputBuffer = lstm.forward(input: inputBuffer)
    
    // Convert output buffer back to array to print
    let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: hiddenSize)
    let outputArray = Array(UnsafeBufferPointer(start: outputPointer, count: hiddenSize))
    
    // Print LSTM output for the current time step
    print("LSTM output for the current time step:", outputArray)
}

testMetalLSTMCell()











