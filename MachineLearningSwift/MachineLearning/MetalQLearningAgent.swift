//
//  QLearningAgent.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 01.11.24.
//

import Foundation
import Metal
import MetalPerformanceShaders

class MetalQLearning {
    
    var numStates: Int
    var numActions: Int
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    var qTableBuffer: MTLBuffer
    
    var learningRate: Float
    var discountFactor: Float
    var explorationRate: Float
    
    init(numStates: Int, numActions: Int, learningRate: Float, discountFactor: Float, explorationRate: Float) {
        self.numStates = numStates
        self.numActions = numActions
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Initialize Q-table with random values
        qTableBuffer = device.makeBuffer(length: numStates * numActions * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let qTablePointer = qTableBuffer.contents().bindMemory(to: Float.self, capacity: numStates * numActions)
        
        for i in 0..<(numStates * numActions) {
            qTablePointer[i] = Float.random(in: -1...1)
        }
    }
    
    // Step to select action using epsilon-greedy policy
    func selectAction(forState state: Int) -> Int {
        if Float.random(in: 0...1) < explorationRate {
            return Int.random(in: 0..<numActions) // Explore: choose random action
        } else {
            // Exploit: choose best action for the current state
            return maxAction(forState: state)
        }
    }
    
    // Find the action with the highest Q-value for a given state
    private func maxAction(forState state: Int) -> Int {
        let qTablePointer = qTableBuffer.contents().bindMemory(to: Float.self, capacity: numStates * numActions)
        let startIndex = state * numActions
        var bestAction = 0
        var maxQ = qTablePointer[startIndex]
        
        for action in 1..<numActions {
            let qValue = qTablePointer[startIndex + action]
            if qValue > maxQ {
                maxQ = qValue
                bestAction = action
            }
        }
        return bestAction
    }
    
    // Update Q-table based on the Q-learning formula
    func updateQValue(currentState: Int, action: Int, reward: Float, nextState: Int) {
        let qTablePointer = qTableBuffer.contents().bindMemory(to: Float.self, capacity: numStates * numActions)
        
        let startIndex = nextState * numActions
        let endIndex = startIndex + numActions
        let maxNextQ = (startIndex..<endIndex).map { qTablePointer[$0] }.max() ?? 0.0
        let oldQ = qTablePointer[currentState * numActions + action]
        
        // Apply Q-learning update formula
        let newQ = oldQ + learningRate * (reward + discountFactor * maxNextQ - oldQ)
        qTablePointer[currentState * numActions + action] = newQ
    }
    
    
    /*
     
     
     // Metal-accelerated max Q-value calculation for the next state
     func maxQValue(forState state: Int) -> Float {
     let commandBuffer = commandQueue.makeCommandBuffer()!
     
     // Update to use the correct number of rows and columns for this function
     let inputMatrixDescriptor = MPSMatrixDescriptor(rows: numActions, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
     let resultIndexMatrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: 1, rowBytes: MemoryLayout<Int32>.stride, dataType: .int32)
     let resultValueMatrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
     
     // Create the input matrix and result matrices
     let inputMatrix = MPSMatrix(buffer: qTableBuffer, descriptor: inputMatrixDescriptor)
     let resultIndexMatrix = MPSMatrix(buffer: device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared)!, descriptor: resultIndexMatrixDescriptor)
     let resultValueMatrix = MPSMatrix(buffer: device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!, descriptor: resultValueMatrixDescriptor)
     
     // Set up MPSMatrixFindTopK kernel
     let maxQValueKernel = MPSMatrixFindTopK(device: device, numberOfTopKValues: 1)
     
     // Encode the command to find the top 1 value
     maxQValueKernel.encode(commandBuffer: commandBuffer, inputMatrix: inputMatrix, resultIndexMatrix: resultIndexMatrix, resultValueMatrix: resultValueMatrix)
     
     commandBuffer.commit()
     commandBuffer.waitUntilCompleted()
     
     // Get the maximum Q-value for the given state
     let resultPointer = resultValueMatrix.data.contents().bindMemory(to: Float.self, capacity: 1)
     return resultPointer[0]
     }
     
     */
    
    func maxQValue(forState state: Int) -> Float {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Calculate the offset for the current state's Q-values
        let stateOffset = state * numActions * MemoryLayout<Float>.stride
        
        // Create a temporary buffer for the current state's Q-values
        let stateBuffer = device.makeBuffer(length: numActions * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Copy the Q-values for the current state to the temporary buffer
        memcpy(stateBuffer.contents(),
               qTableBuffer.contents().advanced(by: stateOffset),
               numActions * MemoryLayout<Float>.stride)
        
        // Create matrix descriptors
        let inputMatrixDescriptor = MPSMatrixDescriptor(rows: 1,           // One row
                                                        columns: numActions, // Number of actions as columns
                                                        rowBytes: numActions * MemoryLayout<Float>.stride,
                                                        dataType: .float32)
        
        let resultIndexMatrixDescriptor = MPSMatrixDescriptor(rows: 1,     // One row
                                                              columns: 1,    // One column (top-1)
                                                              rowBytes: MemoryLayout<Int32>.stride,
                                                              dataType: .int32)
        
        let resultValueMatrixDescriptor = MPSMatrixDescriptor(rows: 1,     // One row
                                                              columns: 1,    // One column (top-1)
                                                              rowBytes: MemoryLayout<Float>.stride,
                                                              dataType: .float32)
        
        // Create matrices
        let inputMatrix = MPSMatrix(buffer: stateBuffer,
                                    descriptor: inputMatrixDescriptor)
        
        let resultIndexMatrix = MPSMatrix(buffer: device.makeBuffer(length: MemoryLayout<Int32>.stride,
                                                                    options: .storageModeShared)!,
                                          descriptor: resultIndexMatrixDescriptor)
        
        let resultValueMatrix = MPSMatrix(buffer: device.makeBuffer(length: MemoryLayout<Float>.stride,
                                                                    options: .storageModeShared)!,
                                          descriptor: resultValueMatrixDescriptor)
        
        // Create and configure the FindTopK kernel
        let maxQValueKernel = MPSMatrixFindTopK(device: device, numberOfTopKValues: 1)
        
        // Encode the command
        maxQValueKernel.encode(commandBuffer: commandBuffer,
                               inputMatrix: inputMatrix,
                               resultIndexMatrix: resultIndexMatrix,
                               resultValueMatrix: resultValueMatrix)
        
        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Get the maximum Q-value
        let resultPointer = resultValueMatrix.data.contents().bindMemory(to: Float.self, capacity: 1)
        return resultPointer[0]
    }
    
    
    
    
}



/*
 class Environment {
 
 let numStates: Int
 let numActions: Int
 let rewards: [[Double]]
 
 init(numStates: Int, numActions: Int, rewards: [[Double]]) {
 self.numStates = numStates
 self.numActions = numActions
 self.rewards = rewards
 }
 
 
 func getNextState(currentState: Int, action: Int) -> Int {
 return (currentState + action) % numStates
 }
 
 func getReward(currentState: Int, action: Int) -> Double {
 return rewards[currentState][action]
 }
 }
 
 */

