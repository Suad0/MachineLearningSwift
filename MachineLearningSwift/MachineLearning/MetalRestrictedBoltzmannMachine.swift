//
//  MetalRestrictedBoltzmannMachine.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 15.10.24.
//

import Foundation

import Metal
import MetalPerformanceShaders

class MetalRestrictedBoltzmannMachine {
    
    var numVisible: Int
    var numHidden: Int
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    var weightsBuffer: MTLBuffer
    var hiddenBiasBuffer: MTLBuffer
    var visibleBiasBuffer: MTLBuffer
    
    init(numVisible: Int, numHidden: Int) {
        self.numVisible = numVisible
        self.numHidden = numHidden
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Initialize weights and biases using Xavier initialization
        let weightCount = numVisible * numHidden
        weightsBuffer = device.makeBuffer(length: weightCount * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Xavier initialization for weights
        let limit = sqrt(6.0 / Double(numVisible + numHidden))
        let weightPointer = weightsBuffer.contents().bindMemory(to: Float.self, capacity: weightCount)
        for i in 0..<weightCount {
            weightPointer[i] = Float.random(in: -Float(limit)...Float(limit))
        }
        
        // Initialize hidden and visible biases to zero
        hiddenBiasBuffer = device.makeBuffer(length: numHidden * MemoryLayout<Float>.stride, options: .storageModeShared)!
        visibleBiasBuffer = device.makeBuffer(length: numVisible * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        let hiddenBiasPointer = hiddenBiasBuffer.contents().bindMemory(to: Float.self, capacity: numHidden)
        let visibleBiasPointer = visibleBiasBuffer.contents().bindMemory(to: Float.self, capacity: numVisible)
        
        for i in 0..<numHidden { hiddenBiasPointer[i] = 0.0 }
        for i in 0..<numVisible { visibleBiasPointer[i] = 0.0 }
    }
    
    
    
    func persistentCD(k: Int, chain: inout MTLBuffer) -> MTLBuffer {
        for _ in 0..<k {
            // Sample hidden from visible
            let hidden = sampleHidden(from: chain)
            // Sample visible from hidden
            chain = sampleVisible(from: hidden)
        }
        return chain
    }
    
    func sampleHidden(from visible: MTLBuffer) -> MTLBuffer {
        let hiddenBuffer = device.makeBuffer(length: numHidden * MemoryLayout<Float>.stride, options: .storageModeShared)!
        matrixMultiply(weightBuffer: weightsBuffer, inputBuffer: visible, resultBuffer: hiddenBuffer, rows: numHidden, columns: numVisible)
        return sigmoid(inputBuffer: hiddenBuffer, length: numHidden)
    }
    
    func sampleVisible(from hidden: MTLBuffer) -> MTLBuffer {
        let visibleBuffer = device.makeBuffer(length: numVisible * MemoryLayout<Float>.stride, options: .storageModeShared)!
        matrixMultiply(weightBuffer: weightsBuffer, inputBuffer: hidden, resultBuffer: visibleBuffer, rows: numVisible, columns: numHidden)
        return sigmoid(inputBuffer: visibleBuffer, length: numVisible)
    }
    
    
    func energy(visible: MTLBuffer, hidden: MTLBuffer) -> Float {
        var energy: Float = 0.0
        
        // Compute - hᵀ W v
        let hiddenBuffer = device.makeBuffer(length: numHidden * MemoryLayout<Float>.stride, options: .storageModeShared)!
        matrixMultiply(weightBuffer: weightsBuffer, inputBuffer: visible, resultBuffer: hiddenBuffer, rows: numHidden, columns: numVisible)
        
        let hiddenPointer = hidden.contents().bindMemory(to: Float.self, capacity: numHidden)
        let dotHiddenWeightVisible = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: numHidden)
        
        for i in 0..<numHidden {
            energy -= hiddenPointer[i] * dotHiddenWeightVisible[i]
        }
        
        // Compute - bᵀ v (visible bias term)
        let visiblePointer = visible.contents().bindMemory(to: Float.self, capacity: numVisible)
        let visibleBiasPointer = visibleBiasBuffer.contents().bindMemory(to: Float.self, capacity: numVisible)
        
        for i in 0..<numVisible {
            energy -= visiblePointer[i] * visibleBiasPointer[i]
        }
        
        // Compute - cᵀ h (hidden bias term)
        let hiddenBiasPointer = hiddenBiasBuffer.contents().bindMemory(to: Float.self, capacity: numHidden)
        
        for i in 0..<numHidden {
            energy -= hiddenPointer[i] * hiddenBiasPointer[i]
        }
        
        return energy
    }
    
    
    
    func updateWeights(learningRate: Float, visible0: MTLBuffer, hidden0: MTLBuffer, visibleK: MTLBuffer, hiddenK: MTLBuffer) {
        let weightPointer = weightsBuffer.contents().bindMemory(to: Float.self, capacity: numVisible * numHidden)
        let visible0Pointer = visible0.contents().bindMemory(to: Float.self, capacity: numVisible)
        let hidden0Pointer = hidden0.contents().bindMemory(to: Float.self, capacity: numHidden)
        let visibleKPointer = visibleK.contents().bindMemory(to: Float.self, capacity: numVisible)
        let hiddenKPointer = hiddenK.contents().bindMemory(to: Float.self, capacity: numHidden)
        
        // Update weights
        for i in 0..<numVisible {
            for j in 0..<numHidden {
                let idx = i * numHidden + j
                weightPointer[idx] += learningRate * (visible0Pointer[i] * hidden0Pointer[j] - visibleKPointer[i] * hiddenKPointer[j])
            }
        }
        
        // Update visible bias
        let visibleBiasPointer = visibleBiasBuffer.contents().bindMemory(to: Float.self, capacity: numVisible)
        for i in 0..<numVisible {
            visibleBiasPointer[i] += learningRate * (visible0Pointer[i] - visibleKPointer[i])
        }
        
        // Update hidden bias
        let hiddenBiasPointer = hiddenBiasBuffer.contents().bindMemory(to: Float.self, capacity: numHidden)
        for i in 0..<numHidden {
            hiddenBiasPointer[i] += learningRate * (hidden0Pointer[i] - hiddenKPointer[i])
        }
    }
    
    
    func reconstructionError(original: MTLBuffer, reconstructed: MTLBuffer) -> Float {
        let originalPointer = original.contents().bindMemory(to: Float.self, capacity: numVisible)
        let reconstructedPointer = reconstructed.contents().bindMemory(to: Float.self, capacity: numVisible)
        
        var mse: Float = 0.0
        for i in 0..<numVisible {
            let diff = originalPointer[i] - reconstructedPointer[i]
            mse += diff * diff
        }
        
        return mse / Float(numVisible)
    }
    
    
    func train(learningRate: Float, k: Int, epochs: Int, batchSize: Int, data: [MTLBuffer]) {
        // Initialize chain buffer (e.g., as the first batch of data)
        var chain = data[0]
        
        for epoch in 0..<epochs {
            for batchIndex in stride(from: 0, to: data.count, by: batchSize) {
                let visible0 = data[batchIndex]
                let hidden0 = sampleHidden(from: visible0)
                
                // Perform persistent CD
                let visibleK = persistentCD(k: k, chain: &chain)
                let hiddenK = sampleHidden(from: visibleK)
                
                // Update weights and biases
                updateWeights(learningRate: learningRate, visible0: visible0, hidden0: hidden0, visibleK: visibleK, hiddenK: hiddenK)
            }
            
            // (Optional) Track reconstruction error or energy over epochs
            let reconstruction = sampleVisible(from: sampleHidden(from: chain))
            let error = reconstructionError(original: chain, reconstructed: reconstruction)
            print("Epoch \(epoch), Reconstruction Error: \(error)")
        }
    }
    
    
    
    
    func matrixMultiply(weightBuffer: MTLBuffer, inputBuffer: MTLBuffer, resultBuffer: MTLBuffer, rows: Int, columns: Int) {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let matrixMultiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: rows,
            resultColumns: 1,
            interiorColumns: columns,
            alpha: 1.0,
            beta: 0.0
        )
        
        let inputDescriptor = MPSMatrixDescriptor(rows: columns, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let resultDescriptor = MPSMatrixDescriptor(rows: rows, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        
        matrixMultiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: MPSMatrix(buffer: weightBuffer, descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * MemoryLayout<Float>.stride, dataType: .float32)),
            rightMatrix: MPSMatrix(buffer: inputBuffer, descriptor: inputDescriptor),
            resultMatrix: MPSMatrix(buffer: resultBuffer, descriptor: resultDescriptor)
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    
    func sigmoid(inputBuffer: MTLBuffer, length: Int) -> MTLBuffer {
        let resultBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let inputPointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: length)
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: length)
        
        for i in 0..<length {
            resultPointer[i] = 1.0 / (1.0 + exp(-inputPointer[i]))
        }
        
        return resultBuffer
    }
    
}
