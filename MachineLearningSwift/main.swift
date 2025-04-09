//
//  main.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation
import Metal
import CoreML
import Numerics



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






func computeMatrixStatistics(_ matrix: [[Double]]) -> (mean: Double, variance: Double, min: Double, max: Double) {
    let flattenedMatrix = matrix.flatMap { $0 }
    
    let mean = flattenedMatrix.reduce(0, +) / Double(flattenedMatrix.count)
    let variance = flattenedMatrix.map { pow($0 - mean, 2) }.reduce(0, +) / Double(flattenedMatrix.count)
    let min = flattenedMatrix.min() ?? 0
    let max = flattenedMatrix.max() ?? 0
    
    return (mean, variance, min, max)
}



public func usageXLSTM() {
    let lstmCell = xLSTMCell(inputSize: 10, hiddenSize: 20, memorySize: 20)
    
    let inputs: [[Float]] = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ]
    
    let outputs = lstmCell.processSequence(inputs: inputs)
    
    print("Outputs after processing the sequence:")
    for (index, output) in outputs.enumerated() {
        print("Time step \(index): \(output)")
    }
    
    let targetSequence: [[Float]] = [
        [Float](repeating: 0.0, count: 20),
        [Float](repeating: 0.0, count: 20)
    ]
    
    lstmCell.backpropagate(dhSequence: targetSequence)
    
    let updatedOutputs = lstmCell.processSequence(inputs: inputs)
    print("\nOutputs after backpropagation and weight updates:")
    for (index, output) in updatedOutputs.enumerated() {
        print("Time step \(index): \(output)")
    }
}


func testTransformer() {
    do {
        let config = try Transformer.TransformerConfiguration(
            inputSize: 4,
            hiddenSize: 8,
            outputSize: 4,
            numHeads: 2,
            ffHiddenSize: 16,
            dropout: 0.1,
            learningRate: 0.001
        )
        
        let transformer = try Transformer(config: config)
        transformer.setTrainingMode(true)
        
        // Test input: batch of 2 sequences, each with 3 tokens of size 4
        let input: [[Double]] = [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
            [0.1, 0.2, 0.3, 0.4]
        ]
        
        let encoded = try transformer.encoderLayer(input: input)
        print("Encoder output shape: [\(encoded.count), \(encoded[0].count)]")
        print("First token output: \(encoded[0])")
        
        assert(encoded.count == input.count)
        assert(encoded[0].count == config.hiddenSize)
        print("All tests passed!")
    } catch {
        print("Test failed with error: \(error)")
    }
}

//testTransformer()








func testDifferentialEquations() {
    print("\n=== ODE SOLVER TEST SUITE ===\n")
    
    // Test configuration
    let testSteps = 5
    let testTolerance = 1e-6
    let testGamma = 1.0
    let testN = 100
    
    // Test 1: Euler Method (dy/dx = -2xy)
    let eulerResult = ODESolver.eulerMethod(x0: 0.0, y0: 1.0, h: 0.1, steps: testSteps) { x, y in
        -2 * x * y
    }
    
    // Test 2: Runge-Kutta 4th Order
    let rk4Result = ODESolver.rungeKutta4(x0: 0.0, y0: 1.0, h: 0.1, steps: testSteps) { x, y in
        -2 * x * y
    }
    
    // Test 3: Adaptive RKF45
    let rkf45Result = ODESolver.rkf45(x0: 0.0, y0: 1.0, h: 0.1, tolerance: testTolerance, maxSteps: 10) { x, y in
        -2 * x * y
    }
    
    // Test 4: Simpson's Rule (∫x² dx from 0 to 1)
    do {
        let integralResult = try ODESolver.simpsonsRule(a: 0.0, b: 1.0, n: testN) { x in
            x * x
        }
        print("\nNumerical Integration Test:")
        print(String(format: "∫x² dx from 0 to 1 = %.6f (Expected ≈0.333333)", integralResult))
    } catch {
        print("Integration Error: \(error)")
    }
    
    // Test 5: Inverse Laplace Transform (L⁻¹{1/(s²+1)} = sin(t))
    do {
        let t = Double.pi/2 // Should get ≈1.0
        let iltResult = try ODESolver.inverseLaplace(t: t, gamma: testGamma, n: testN) { s in
            1 / (s * s + 1)
        }
        print("\nInverse Laplace Test:")
        print(String(format: "L⁻¹{1/(s²+1)} at t=π/2 = %.6f (Expected ≈1.000000)", iltResult))
    } catch {
        print("Laplace Transform Error: \(error)")
    }
    
    print("\n=== TEST COMPLETE ===\n")
}

// testDifferentialEquations()

func testAnalysis() {
    let x = MathExpression.variable("x")
    let y = MathExpression.variable("y")
    
    // Differentiate x^3
    let expr1 = MathExpression.power(base: x, exponent: .constant(3))
    let derivative1 = derivative(of: expr1, withRespectTo: "x")
    print("d/dx \(expr1) = \(derivative1)")  // d/dx ((x)^3.00) = (3.00 * (x)^2.00 * 1.00)
    
    // Integrate x^2
    let integral1 = integral(of: expr1, withRespectTo: "x")
    print("∫\(expr1) dx = \(integral1)")  // ∫((x)^3.00) dx = ((x)^4.00 / 4.00)
    
    // Differentiate sin(x^2)
    let expr2 = MathExpression.sin(.power(base: x, exponent: .constant(2)))
    let derivative2 = derivative(of: expr2, withRespectTo: "x")
    print("d/dx \(expr2) = \(derivative2)")  // d/dx sin((x)^2.00) = (cos((x)^2.00) * (2.00 * (x)^1.00 * 1.00))
}

//testAnalysis()


func demonstrateStatisticalCalculator() {
    // Sample datasets
    let studentScores: [Double] = [85, 92, 78, 95, 88, 83, 90, 87, 91, 84]
    let studentWeights: [Double] = [1.2, 1.0, 0.8, 1.1, 1.0, 0.9, 1.0, 1.0, 1.1, 0.9]
    
    let examScores1: [Double] = [72, 85, 90, 88, 83, 87, 89, 92]
    let examScores2: [Double] = [75, 88, 92, 85, 85, 89, 91, 94]
    
    print("🔍 Statistical Analysis Demo")
    print("============================")
    
    do {
        // Basic Statistics
        print("\n📊 Basic Statistics for Student Scores:")
        print("---------------------------------------")
        print("Mean Score: \(try StatisticalCalculator.mean(studentScores))")
        print("Median Score: \(try StatisticalCalculator.median(studentScores))")
        print("Standard Deviation: \(try StatisticalCalculator.standardDeviation(studentScores))")
        
        // Weighted Statistics
        print("\n⚖️ Weighted Statistics:")
        print("----------------------")
        print("Weighted Mean Score: \(try StatisticalCalculator.weightedMean(studentScores, weights: studentWeights))")
        
        // Comprehensive Analysis
        print("\n📈 Comprehensive Analysis:")
        print("-------------------------")
        let stats = try StatisticalCalculator.descriptiveStatistics(studentScores)
        print("Minimum: \(stats["minimum"] ?? 0)")
        print("Maximum: \(stats["maximum"] ?? 0)")
        print("Range: \(stats["range"] ?? 0)")
        print("Skewness: \(stats["skewness"] ?? 0)")
        print("Kurtosis: \(stats["kurtosis"] ?? 0)")
        
        // Correlation Analysis
        print("\n🔗 Correlation Analysis between Exam Scores:")
        print("------------------------------------------")
        let correlation = try StatisticalCalculator.correlation(examScores1, examScores2)
        print("Correlation Coefficient: \(correlation)")
        
        // Linear Regression
        print("\n📉 Linear Regression Analysis:")
        print("-----------------------------")
        let regression = try StatisticalCalculator.linearRegression(examScores1, examScores2)
        print("Slope: \(regression.slope)")
        print("Intercept: \(regression.intercept)")
        
        // Probability Distributions
        print("\n🎲 Probability Distributions:")
        print("--------------------------")
        print("Normal PDF at x=85, μ=87, σ=5: \(try StatisticalCalculator.normalPDF(x: 85, mu: 87, sigma: 5))")
        print("Normal CDF at x=85, μ=87, σ=5: \(StatisticalCalculator.normalCDF(x: 85, mu: 87, sigma: 5))")
        
        // Hypothesis Testing
        print("\n🧪 Hypothesis Testing:")
        print("--------------------")
        let zTestResult = try StatisticalCalculator.zTest(
            sampleMean: try StatisticalCalculator.mean(studentScores),
            populationMean: 85,
            sigma: 5,
            n: studentScores.count
        )
        print("Z-Test Score: \(zTestResult.zScore)")
        print("P-Value: \(zTestResult.pValue)")
        
        // Using Array Extensions
        print("\n🔄 Using Array Extensions:")
        print("-------------------------")
        print("Direct Mean Calculation: \(try studentScores.mean())")
        print("Direct Standard Deviation: \(try studentScores.standardDeviation())")
        
    } catch StatisticalCalculator.StatisticalError.insufficientData {
        print("Error: Insufficient data for calculation")
    } catch StatisticalCalculator.StatisticalError.invalidParameters {
        print("Error: Invalid parameters provided")
    } catch {
        print("An unexpected error occurred: \(error)")
    }
}

//demonstrateStatisticalCalculator()

func testMarkovChain() {
    
    let markovChain = MarkovChain()
    
    // Training data
    let trainingSequence = ["A", "B", "C", "A", "B", "D", "A", "B", "A", "B"]
    
    // Train the model
    markovChain.train(on: trainingSequence)
    
    // Generate a sequence
    let generatedSequence = markovChain.generateSequence(length: 10)
    print("Generated sequence: \(generatedSequence)")
}

//testMarkovChain()

func testTensorExtension() {
    // Tensor Description Test
    let tensor = Tensor([[1.0, 2.0], [3.0, 4.0]])
    print("Tensor Description Test:")
    print(tensor.tensorDescription())
    print("\n")
    
    // Parallel Matrix Multiplication Tests
    print("Parallel Matrix Multiplication Tests:")
    let tensor1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    let tensor2 = Tensor([[5.0, 6.0], [7.0, 8.0]])
    
    do {
        let result = try tensor1.parallelMatmul(tensor2)
        print("Matrix Multiplication Result:")
        print(result.value)
    } catch {
        print("Matrix multiplication failed: \(error)")
    }
    print("\n")
    
    // Safe Division Test
    print("Safe Division Test:")
    let dividendTensor = Tensor([[10.0, 20.0], [30.0, 40.0]])
    let divisorTensor = Tensor([[2.0, 4.0], [5.0, 8.0]])
    
    do {
        let dividedTensor = try dividendTensor.divide(by: divisorTensor)
        print("Division Result:")
        print(dividedTensor.value)
    } catch {
        print("Division failed: \(error)")
    }
    print("\n")
    
    // JSON Serialization and Reconstruction Test
    print("JSON Serialization Test:")
    let jsonRepresentation = tensor.toJSON()
    print("JSON Representation:")
    print(jsonRepresentation)
    
    if let reconstructedTensor = Tensor.fromJSON(jsonRepresentation) {
        print("\nReconstructed Tensor:")
        print(reconstructedTensor.tensorDescription())
    }
}

//testTensorExtension()


func xLSTMModel_test() {
    let model = xLSTMModel(
        inputSize: 5,
        hiddenSize: 10,
        memorySize: 10,
        outputSize: 1
    )

    let sequence: [[Float]] = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    let target: [Float] = [0.8]

    model.train(sequence: sequence, target: target)

    let prediction = model.predict(sequence: sequence)
    print("Prediction: \(prediction)")

    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("model.json")
    try? model.save(to: url)
    
    
    
    
    
    /*

    if let loadedModel = try? xLSTMModel.load(from: url) {
        let newPrediction = loadedModel.predict(sequence: sequence)
        print("Loaded model prediction: \(newPrediction)")
    }
     
     */
    
    
}

xLSTMModel_test()
