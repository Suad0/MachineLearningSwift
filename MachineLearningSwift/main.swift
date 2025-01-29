//
//  main.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation
import Metal

import CoreML

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
    
    let inputSize = 10
    let hiddenSize = 20
    let n_timesteps = 3
    
    // Initialize LSTM cell
    let lstmCell = MetalLSTMCell(inputSize: inputSize, hiddenSize: hiddenSize)
    
    // Loop over timesteps
    for timestep in 0..<n_timesteps {
        // Create an input buffer for each timestep
        let inputLength = inputSize * MemoryLayout<Float>.stride
        let inputBuffer = lstmCell.device.makeBuffer(length: inputLength, options: .storageModeShared)!
        
        // Fill with values (here it's filled with 1.0s, but this would be your actual input data)
        let inputPointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: inputSize)
        for i in 0..<inputSize {
            inputPointer[i] = Float(1.0)  // Use real data here
        }
        
        // Perform forward pass
        let hiddenStateBuffer = lstmCell.forward(input: inputBuffer)
        
        // Access hidden state values
        let hiddenStatePointer = hiddenStateBuffer.contents().bindMemory(to: Float.self, capacity: hiddenSize)
        for i in 0..<hiddenSize {
            print("Timestep \(timestep), hidden state at index \(i): \(hiddenStatePointer[i])")
        }
    }
    
}

//testMetalLSTMCell()

func testRestrictedBoltzmannMachine() {
    
    
    // 1. Prepare the data (binary in this case)
    let data: [[Double]] = [
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1]
    ]
    
    let numVisible = 4
    let numHidden = 3
    let rbm = RestrictedBoltzmannMachine(numVisible: numVisible, numHidden: numHidden)
    
    let epochs = 1000
    let learningRate = 0.1
    rbm.train(data: data, epochs: epochs, learningRate: learningRate)
    
    for sample in data {
        let hidden = rbm.sampleHidden(from: sample)
        let reconstructed = rbm.sampleVisible(from: hidden)
        print("Original: \(sample) -> Reconstructed: \(reconstructed)")
    }
    
    
    
}

//testRestrictedBoltzmannMachine()


func trainAndEvaluateRBM(numVisible: Int, numHidden: Int, data: [[Float]], epochs: Int, batchSize: Int, learningRate: Float, k: Int) {
    
    // Initialize the Metal device and command queue
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal is not supported on this device")
    }
    
    // Prepare data into Metal buffers
    func prepareData(device: MTLDevice, data: [[Float]]) -> [MTLBuffer] {
        var buffers = [MTLBuffer]()
        for sample in data {
            let buffer = device.makeBuffer(bytes: sample, length: sample.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
            buffers.append(buffer)
        }
        return buffers
    }
    
    let trainingBuffers = prepareData(device: device, data: data)
    
    // Initialize RBM
    let rbm = MetalRestrictedBoltzmannMachine(numVisible: numVisible, numHidden: numHidden)
    
    // Training Loop
    for epoch in 0..<epochs {
        print("Epoch \(epoch + 1)/\(epochs)")
        var totalReconstructionError: Float = 0.0
        
        for i in stride(from: 0, to: trainingBuffers.count, by: batchSize) {
            // Select a batch from the training data
            let batchEnd = min(i + batchSize, trainingBuffers.count)
            let batch = Array(trainingBuffers[i..<batchEnd])
            
            for sample in batch {
                // Initialize persistent chain (in this case, start with the visible sample)
                var chain = sample
                
                // Perform Contrastive Divergence (CD-k) using persistentCD
                let visible0 = chain
                let hidden0 = rbm.sampleHidden(from: visible0)
                let visibleK = rbm.persistentCD(k: k, chain: &chain)
                let hiddenK = rbm.sampleHidden(from: visibleK)
                
                // Update weights using the CD-k results
                rbm.updateWeights(learningRate: learningRate, visible0: visible0, hidden0: hidden0, visibleK: visibleK, hiddenK: hiddenK)
                
                // Calculate reconstruction error for this sample
                let reconstructionError = rbm.reconstructionError(original: visible0, reconstructed: visibleK)
                totalReconstructionError += reconstructionError
            }
        }
        
        // Average reconstruction error for the epoch
        let avgReconstructionError = totalReconstructionError / Float(trainingBuffers.count)
        print("Average Reconstruction Error in Epoch \(epoch + 1): \(avgReconstructionError)")
    }
    
    // Evaluate the model by reconstructing the first sample in the dataset
    let testSample = trainingBuffers[0] // Take the first sample for testing
    let hiddenSample = rbm.sampleHidden(from: testSample) // Sample hidden units
    let reconstructedSample = rbm.sampleVisible(from: hiddenSample) // Reconstruct visible units
    
    let reconstructionError = rbm.reconstructionError(original: testSample, reconstructed: reconstructedSample)
    print("Reconstruction Error on Test Sample: \(reconstructionError)")
    
    // Optionally, extract the learned weights and biases
    func extractWeights(rbm: MetalRestrictedBoltzmannMachine) -> [[Float]] {
        let weightPointer = rbm.weightsBuffer.contents().bindMemory(to: Float.self, capacity: rbm.numVisible * rbm.numHidden)
        var weights = [[Float]](repeating: [Float](repeating: 0.0, count: rbm.numHidden), count: rbm.numVisible)
        
        for i in 0..<rbm.numVisible {
            for j in 0..<rbm.numHidden {
                let index = i * rbm.numHidden + j
                weights[i][j] = weightPointer[index]
            }
        }
        
        return weights
    }
    
    let learnedWeights = extractWeights(rbm: rbm)
    print("Learned Weights: \(learnedWeights)")
}

let numVisible = 784  // Number of visible units (e.g., for 28x28 images)
let numHidden = 128   // Number of hidden units
let epochs = 10       // Number of training epochs
let batchSize = 10    // Batch size for training
let learningRate: Float = 0.01  // Learning rate
let k = 1             // Number of Gibbs sampling steps in Contrastive Divergence


// Generate random data for testing (e.g., 100 samples, each with 784 features)
let numberOfSamples = 100
var data: [[Float]] = []

for _ in 0..<numberOfSamples {
    let sample = (0..<numVisible).map { _ in Float.random(in: 0.0...1.0) } // Random values between 0 and 1
    data.append(sample)
}




//trainAndEvaluateRBM(numVisible: numVisible, numHidden: numHidden, data: data, epochs: epochs, batchSize: batchSize, learningRate: learningRate, k: k)

func testQLearning() {
    let numStates = 5
    let numActions = 3
    let learningRate: Float = 0.1
    let discountFactor: Float = 0.9
    let explorationRate: Float = 0.1
    
    let qTable = MetalQLearning(
        numStates: numStates,
        numActions: numActions,
        learningRate: learningRate,
        discountFactor: discountFactor,
        explorationRate: explorationRate
    )
    
    let testEpisodes = 10
    
    for episode in 1...testEpisodes {
        print("Episode \(episode):")
        
        var currentState = Int.random(in: 0..<numStates)
        
        for _ in 0..<10 {
            let action = Int.random(in: 0..<numActions)
            let reward: Float = Float.random(in: 0..<10)
            let nextState = Int.random(in: 0..<numStates)
            
            // Use the corrected maxQValue function to get the max Q for the next state
            let maxNextQ = qTable.maxQValue(forState: nextState)
            
            // Update Q-value for (currentState, action)
            let qTablePointer = qTable.qTableBuffer.contents().bindMemory(to: Float.self, capacity: numStates * numActions)
            let oldQ = qTablePointer[currentState * numActions + action]
            let newQ = oldQ + learningRate * (reward + discountFactor * maxNextQ - oldQ)
            qTablePointer[currentState * numActions + action] = newQ
            
            print("State: \(currentState), Action: \(action), Reward: \(reward), Next State: \(nextState), New Q: \(newQ)")
            
            currentState = nextState
        }
    }
    
    // Print final Q-table values for inspection
    let finalQTablePointer = qTable.qTableBuffer.contents().bindMemory(to: Float.self, capacity: numStates * numActions)
    print("\nFinal Q-Table:")
    for state in 0..<numStates {
        for action in 0..<numActions {
            print("Q(\(state), \(action)) = \(finalQTablePointer[state * numActions + action])")
        }
    }
}

//testQLearning()



/*
 func chatBotMain() {
 do {
 let chatbot = try Chatbot()
 if let tag = chatbot.predictTag(for: "what is project managment") {
 print("Predicted Tag: \(tag)")
 } else {
 print("No tag matched.")
 }
 } catch {
 print("Error initializing Chatbot: \(error)")
 }
 }
 */

func chatBotMain() {
    
    do {
        let chatbot = try Chatbot()
        chatbot.chat()
    } catch {
        print("[ERROR] Failed to initialize chatbot: \(error)")
    }
}

/*
 
 func chatBotLSTM() {
 do {
 let coreMLModel = try chatbot_model(configuration: MLModelConfiguration())
 if let chatbot = try? ChatBotLSTM(coreMLModel: coreMLModel, lstmInputSize: 100, lstmHiddenSize: 734) {
 print("LSTM Chatbot initialized successfully.")
 chatbot.chat()
 } else {
 print("Failed to initialize ChatBotLSTM.")
 }
 } catch {
 print("Error initializing CoreML model: \(error)")
 }
 }
 
 */


func chatBotLSTM() {
    do {
        
        let coreMLModel = try chatbot_model(configuration: MLModelConfiguration())
        
        if let chatbot = ChatBotLSTM(coreMLModel: coreMLModel,
                                     lstmInputSize: 100,
                                     lstmHiddenSize: 734) {
            print("LSTM Chatbot initialized successfully.")
            chatbot.chat()
        } else {
            print("Failed to initialize ChatBotLSTM.")
        }
    } catch {
        print("Error initializing CoreML model: \(error)")
    }
}



//chatBotLSTM()





//chatBotMain()

func testTalkingChatBot() {
    do {
        // Initialize the CoreML model
        let coreMLModel = try chatbot_model(configuration: MLModelConfiguration())
        
        // Create an instance of ChatBotLSTMTalking
        if let talkingChatBot = ChatBotLSTMTalking(coreMLModel: coreMLModel,
                                                   lstmInputSize: 100,
                                                   lstmHiddenSize: 734) {
            print("Talking Chatbot initialized successfully.")
            
            // Start the chatbot
            talkingChatBot.chat()
            
            // Keep the program running to listen for input
            RunLoop.main.run()
        } else {
            print("Failed to initialize ChatBotLSTMTalking.")
        }
    } catch {
        print("Error initializing CoreML model: \(error)")
    }
}


func testTensorANDAutoGrad(){
    do {
        // Matrix Multiplication
        let a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        let b = Tensor([[2.0, 0.0], [1.0, 3.0]])
        
        let c = try a.matmul(b)
        print("Matrix Multiplication Result:")
        print(c.value)
        
        // Trigonometric Operations
        let x = Tensor([[0.0, Float.pi/2], [Float.pi, 0.0]])
        let sinX = x.sin()
        let cosX = x.cos()
        
        print("\nTrigonometric Operations:")
        print("Sin(x):", sinX.value)
        print("Cos(x):", cosX.value)
        
        // Logarithmic Operations
        let y = Tensor([[1.0, 2.0], [3.0, 4.0]])
        let logY = try y.log()
        
        print("\nLogarithmic Operations:")
        print("Log(y):", logY.value)
        
    } catch let error as TensorError {
        switch error {
        case .dimensionMismatch(let message):
            print("Dimension Error: \(message)")
        case .invalidOperation(let message):
            print("Operation Error: \(message)")
        }
    } catch {
        print(" An unexpected error occurred: \(error)")
    }
    
    let x = Tensor(3.0)
    let y = Tensor(2.0)
    
    let z = x * y + y
    z.backward()
    
    print("Scalar Tensor Example:")
    print("Value of z: \(z.value)")
    print("Gradient of x: \(x.grad)")
    print("Gradient of y: \(y.grad)")
    
    // Multi-dimensional tensor
    let a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = Tensor([[2.0, 3.0], [4.0, 5.0]])
    
    let c = a * b + b
    c.backward()
    
    print("\nMulti-dimensional Tensor Example:")
    print("Value of c: \(c.value)")
    print("Gradient of a: \(a.grad)")
    print("Gradient of b: \(b.grad)")
}

//testTensorANDAutoGrad()

func advancedTensorTest() {
    // Activation Functions
    let x = Tensor([[1.0, -2.0, 3.0]])
    let reluResult = x.relu()
    let sigmoidResult = x.sigmoid()
    let tanhResult = x.tanh()
    
    print("ReLU Result: \(reluResult.value)")
    print("Sigmoid Result: \(sigmoidResult.value)")
    print("Tanh Result: \(tanhResult.value)")
    
    // Statistical Operations
    let stats = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    let mean = stats.mean()
    let std = stats.std()
    
    print("Mean: \(mean.value)")
    print("Standard Deviation: \(std.value)")
    
    // Layer Initialization
    let (weights, bias) = x.createNeuralNetworkLayer(inputSize: 10, outputSize: 5)
    print("Weights Shape: \(weights.value.count)x\(weights.value[0].count)")
    print("Bias Shape: \(bias.value.count)x\(bias.value[0].count)")
}

//advancedTensorTest()


func testTransformer() {
    do {
        // Create transformer configuration
        let config = try Transformer.TransformerConfiguration(
            inputSize: 512,
            hiddenSize: 512,
            outputSize: 512,
            numHeads: 8,
            ffHiddenSize: 2048
        )
        
        // Initialize transformer
        let transformer = try Transformer(config: config)
        
        // Test positional encoding
        let positionalEncodings = [
            try transformer.positionalEncoding(sequenceLength: 10, embeddingSize: 512, type: .sinusoidal),
            try transformer.positionalEncoding(sequenceLength: 10, embeddingSize: 512, type: .learned),
            try transformer.positionalEncoding(sequenceLength: 10, embeddingSize: 512, type: .hybrid)
        ]
        
        print("Positional Encodings Generated: \(positionalEncodings.count)")
        
        // Create sample input matrices
        let inputSequence = (0..<10).map { _ in
            (0..<512).map { _ in Double.random(in: -1...1) }
        }
        
        // Test multi-head attention
        let attentionOutput = try transformer.multiHeadAttention(
            query: inputSequence,
            key: inputSequence,
            value: inputSequence
        )
        
        print("Multi-Head Attention Output Size: \(attentionOutput.count)x\(attentionOutput[0].count)")
        
        // Test encoder layer
        let encoderOutput = try transformer.encoderLayer(input: inputSequence)
        
        print("Encoder Layer Output Size: \(encoderOutput.count)x\(encoderOutput[0].count)")
        
        print("Transformer Test Completed Successfully!")
    } catch {
        print("Transformer Test Failed: \(error)")
    }
}

//testTransformer()


func visualizeTransformerBehavior() {
    do {
        // Create more detailed configuration
        let config = try Transformer.TransformerConfiguration(
            inputSize: 512,
            hiddenSize: 512,
            outputSize: 512,
            numHeads: 8,
            ffHiddenSize: 2048,
            dropout: 0.1
        )
        
        let transformer = try Transformer(config: config)
        
        // Generate a sample input sequence
        let inputSequence = (0..<10).map { pos in
            (0..<512).map { dim in
                sin(Double(pos) * 0.1 + Double(dim) * 0.01)
            }
        }
        
        // Visualization of different transformer components
        struct TransformerComponentAnalysis {
            let name: String
            let inputShape: String
            let outputShape: String
            let statisticalProperties: (mean: Double, variance: Double, min: Double, max: Double)
        }
        
        var analyses: [TransformerComponentAnalysis] = []
        
        // Positional Encoding Analysis
        let sinusoidalEncoding = try transformer.positionalEncoding(
            sequenceLength: inputSequence.count,
            embeddingSize: inputSequence[0].count,
            type: .sinusoidal
        )
        analyses.append(TransformerComponentAnalysis(
            name: "Positional Encoding",
            inputShape: "\(inputSequence.count)x\(inputSequence[0].count)",
            outputShape: "\(sinusoidalEncoding.count)x\(sinusoidalEncoding[0].count)",
            statisticalProperties: computeMatrixStatistics(sinusoidalEncoding)
        ))
        
        // Multi-Head Attention Analysis
        let attentionOutput = try transformer.multiHeadAttention(
            query: inputSequence,
            key: inputSequence,
            value: inputSequence
        )
        analyses.append(TransformerComponentAnalysis(
            name: "Multi-Head Attention",
            inputShape: "\(inputSequence.count)x\(inputSequence[0].count)",
            outputShape: "\(attentionOutput.count)x\(attentionOutput[0].count)",
            statisticalProperties: computeMatrixStatistics(attentionOutput)
        ))
        
        // Encoder Layer Analysis
        let encoderOutput = try transformer.encoderLayer(input: inputSequence)
        analyses.append(TransformerComponentAnalysis(
            name: "Encoder Layer",
            inputShape: "\(inputSequence.count)x\(inputSequence[0].count)",
            outputShape: "\(encoderOutput.count)x\(encoderOutput[0].count)",
            statisticalProperties: computeMatrixStatistics(encoderOutput)
        ))
        
        // Print Visualization Results
        print("🔍 Transformer Component Analysis:")
        for analysis in analyses {
            print("\n\(analysis.name):")
            print("  Input Shape: \(analysis.inputShape)")
            print("  Output Shape: \(analysis.outputShape)")
            print("  Statistical Properties:")
            print("    Mean: \(String(format: "%.4f", analysis.statisticalProperties.mean))")
            print("    Variance: \(String(format: "%.4f", analysis.statisticalProperties.variance))")
            print("    Min: \(String(format: "%.4f", analysis.statisticalProperties.min))")
            print("    Max: \(String(format: "%.4f", analysis.statisticalProperties.max))")
        }
    } catch {
        print("Visualization Error: \(error)")
    }
}

// Helper function to compute matrix statistics
func computeMatrixStatistics(_ matrix: [[Double]]) -> (mean: Double, variance: Double, min: Double, max: Double) {
    let flattenedMatrix = matrix.flatMap { $0 }
    
    let mean = flattenedMatrix.reduce(0, +) / Double(flattenedMatrix.count)
    let variance = flattenedMatrix.map { pow($0 - mean, 2) }.reduce(0, +) / Double(flattenedMatrix.count)
    let min = flattenedMatrix.min() ?? 0
    let max = flattenedMatrix.max() ?? 0
    
    return (mean, variance, min, max)
}

//visualizeTransformerBehavior()


public func usageXLSTM() {
    // Initialize the LSTM cell with specific sizes
    let lstmCell = xLSTMCell(inputSize: 10, hiddenSize: 20, memorySize: 30)
    
    // Define a sequence of inputs
    let inputs: [[Float]] = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ]
    
    // Process the input sequence and get the outputs
    let outputs = lstmCell.processSequence(inputs: inputs)
    
    // Print the outputs for inspection
    print("Outputs after processing the sequence:")
    for (index, output) in outputs.enumerated() {
        print("Time step \(index): \(output)")
    }
    
    // Define a target sequence for backpropagation
    let targetSequence: [[Float]] = [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    ]
    
    // Perform backpropagation to update the weights
    lstmCell.backpropagate(targetSequence: targetSequence)
    
    // Print the updated weights (optional, for debugging)
    //print("\nUpdated weights after backpropagation:")
    //print("Wf: \(lstmCell.Wf)")
    //print("Wi: \(lstmCell.Wi)")
    //print("Wo: \(lstmCell.Wo)")
    //print("Wv: \(lstmCell.Wv)")
    //print("Wk: \(lstmCell.Wk)")
    //print("Wq: \(lstmCell.Wq)")
    
    // Optionally, process the sequence again to see the effect of weight updates
    let updatedOutputs = lstmCell.processSequence(inputs: inputs)
    print("\nOutputs after backpropagation and weight updates:")
    for (index, output) in updatedOutputs.enumerated() {
        print("Time step \(index): \(output)")
    }
}


usageXLSTM()

