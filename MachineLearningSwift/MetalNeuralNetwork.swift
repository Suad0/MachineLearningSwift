//
//  MPSNeuralNetwork.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 17.06.24.
//

import Foundation

import Metal
import MetalPerformanceShaders


// Custom split function for MPSMatrix

/*
 
 extension MPSMatrix {
 
 func split(into chunkSize: Int, buffer: MTLBuffer) -> [[MPSMatrix]] {
 let numChunks = (rows * columns * rowBytes) / chunkSize
 return (0..<numChunks).map { index -> [MPSMatrix] in
 let startIndex = index * chunkSize
 let endIndex = min(startIndex + chunkSize, rows * columns * rowBytes)
 
 let chunkRows = endIndex / rowBytes - startIndex / rowBytes
 let chunkColumns = min(chunkSize / rowBytes, columns)
 
 let chunkDescriptor = MPSMatrixDescriptor(rows: chunkRows, columns: chunkColumns, rowBytes: rowBytes, dataType: dataType)
 let chunkMatrix = MPSMatrix(buffer: buffer, offset: startIndex, descriptor: chunkDescriptor)
 return [chunkMatrix]
 }
 }
 }
 
 */

class MetalNeuralNetwork {
    
    private var inputSize: Int
    private var hiddenSize: Int
    private var outputSize: Int
    private var device: MTLDevice
    private var commandQueue: MTLCommandQueue
    
    private var weightsInputHidden: MTLBuffer
    private var weightsHiddenOutput: MTLBuffer
    private var biasesHidden: MTLBuffer
    private var biasesOutput: MTLBuffer
    
    
    
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        self.weightsInputHidden = device.makeBuffer(length: inputSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        self.weightsHiddenOutput = device.makeBuffer(length: hiddenSize * outputSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        self.biasesHidden = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        self.biasesOutput = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        initializeWeightsAndBiases()
    }
    
    
    
    
    
    private func initializeWeightsAndBiases() {
        // Initialize weights randomly
        let weightsIH = weightsInputHidden.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<inputSize {
            for j in 0..<hiddenSize {
                weightsIH[i * hiddenSize + j] = Float.random(in: -0.5...0.5)
            }
        }
        
        let weightsHO = weightsHiddenOutput.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<hiddenSize {
            for j in 0..<outputSize {
                weightsHO[i * outputSize + j] = Float.random(in: -0.5...0.5)
            }
        }
        
        // Initialize biases to zeros or small random values
        let biasesH = biasesHidden.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<hiddenSize {
            biasesH[i] = 0.0
        }
        
        let biasesO = biasesOutput.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<outputSize {
            biasesO[i] = 0.0
        }
    }
    
    private func sigmoid(_ x: [Float]) -> [Float] {
        return x.map { 1 / (1 + exp(-$0)) }
    }
    
    public func predict(_ input: [Float]) -> [Float] {
        // Convert input to MTLBuffer
        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Forward propagation using MPS
        let hiddenLayerOutput = calculateHiddenLayerOutput(inputBuffer)
        let output = calculateOutput(hiddenLayerOutput)
        
        // Copy result from MTLBuffer to [Float]
        let resultPointer = output.contents().assumingMemoryBound(to: Float.self)
        let result = Array(UnsafeBufferPointer(start: resultPointer, count: outputSize))
        
        return sigmoid(result)
    }
    
    
    
    
    
    
    private func calculateHiddenLayerOutput(_ inputBuffer: MTLBuffer) -> MTLBuffer {
        
        
        print("calculateHiddenLayerOutput")
        
        // Define required sizes
        let requiredInputBufferSize = inputSize * MemoryLayout<Float>.stride
        let hiddenLayerOutputSize = hiddenSize * MemoryLayout<Float>.stride
        
        print(inputBuffer.length)
        
        // Ensure input buffer size is correct
        assert(inputBuffer.length >= requiredInputBufferSize, "inputBuffer is too small for the matrix size.")
        
        // Create the output buffer
        let hiddenLayerOutputBuffer = device.makeBuffer(length: hiddenLayerOutputSize, options: .storageModeShared)!
        
        // Matrix descriptors
        let inputDescriptor = MPSMatrixDescriptor(rows: inputSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let weightsDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: inputSize, rowBytes: inputSize * MemoryLayout<Float>.stride, dataType: .float32)
        let outputDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        
        // Chunk parameters
        let chunkSize = min(inputSize, 8) // Example chunk size, adjust as needed
        let numChunks = (inputSize + chunkSize - 1) / chunkSize
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        for chunkIndex in 0..<numChunks {
            let startRow = chunkIndex * chunkSize
            let numRows = min(chunkSize, inputSize - startRow)
            
            // Debug prints to monitor chunk sizes
            print("Processing chunk \(chunkIndex): startRow=\(startRow), numRows=\(numRows)")
            
            let chunkInputDescriptor = MPSMatrixDescriptor(rows: numRows, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
            let chunkWeightsDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: numRows, rowBytes: numRows * MemoryLayout<Float>.stride, dataType: .float32)
            let chunkOutputDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
            
            // Use offsets for chunking
            let chunkInputBuffer = device.makeBuffer(length: numRows * MemoryLayout<Float>.stride, options: .storageModeShared)!
            let chunkInputMatrix = MPSMatrix(buffer: chunkInputBuffer, descriptor: chunkInputDescriptor)
            let chunkWeightsMatrix = MPSMatrix(buffer: weightsInputHidden, descriptor: chunkWeightsDescriptor)
            let chunkOutputMatrix = MPSMatrix(buffer: hiddenLayerOutputBuffer, descriptor: chunkOutputDescriptor)
            
            // Assert chunk sizes are correct
            assert(chunkInputBuffer.length >= numRows * MemoryLayout<Float>.stride, "Chunk input buffer size mismatch.")
            
            let multiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: hiddenSize, resultColumns: 1, interiorColumns: numRows, alpha: 1.0, beta: 0.0)
            
            multiplication.encode(commandBuffer: commandBuffer, leftMatrix: chunkWeightsMatrix, rightMatrix: chunkInputMatrix, resultMatrix: chunkOutputMatrix)
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return hiddenLayerOutputBuffer
    }
    
    
    
    
    
    private func calculateOutput(_ hiddenLayerOutputBuffer: MTLBuffer) -> MTLBuffer {
        
        
        
        print("calculateOutput")

        // Define required sizes
        let outputBufferSize = outputSize * MemoryLayout<Float>.stride

        // Ensure the hidden layer output buffer is large enough
        assert(hiddenLayerOutputBuffer.length >= hiddenSize * MemoryLayout<Float>.stride, "hiddenLayerOutputBuffer size is too small.")

        // Create the output buffer
        let outputBuffer = device.makeBuffer(length: outputBufferSize, options: .storageModeShared)!

        // Matrix descriptors
        let hiddenDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let weightsDescriptor = MPSMatrixDescriptor(rows: outputSize, columns: hiddenSize, rowBytes: hiddenSize * MemoryLayout<Float>.stride, dataType: .float32)
        let outputDescriptor = MPSMatrixDescriptor(rows: outputSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)

        // Determine chunk size based on buffer capacity
        let maxBufferBytes = outputBufferSize // Size of the output buffer
        let maxChunkSize = min(maxBufferBytes / MemoryLayout<Float>.stride, hiddenSize) // Adjust chunk size
        let numChunks = (hiddenSize + maxChunkSize - 1) / maxChunkSize

        let commandBuffer = commandQueue.makeCommandBuffer()!

        for chunkIndex in 0..<numChunks {
            let startRow = chunkIndex * maxChunkSize
            let rowsInChunk = min(maxChunkSize, hiddenSize - startRow)

            // Debug prints to monitor chunk sizes
            print("Processing chunk \(chunkIndex): startRow=\(startRow), rowsInChunk=\(rowsInChunk)")

            let chunkHiddenDescriptor = MPSMatrixDescriptor(rows: rowsInChunk, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
            let chunkWeightsDescriptor = MPSMatrixDescriptor(rows: outputSize, columns: rowsInChunk, rowBytes: rowsInChunk * MemoryLayout<Float>.stride, dataType: .float32)
            let chunkOutputDescriptor = MPSMatrixDescriptor(rows: outputSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)

            // Use offsets for chunking
            let chunkHiddenMatrix = MPSMatrix(buffer: hiddenLayerOutputBuffer, descriptor: chunkHiddenDescriptor)
            let chunkWeightsMatrix = MPSMatrix(buffer: weightsHiddenOutput, descriptor: chunkWeightsDescriptor)
            let chunkOutputMatrix = MPSMatrix(buffer: outputBuffer, descriptor: chunkOutputDescriptor)

            let multiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: outputSize, resultColumns: 1, interiorColumns: rowsInChunk, alpha: 1.0, beta: 0.0)

            multiplication.encode(commandBuffer: commandBuffer, leftMatrix: chunkWeightsMatrix, rightMatrix: chunkHiddenMatrix, resultMatrix: chunkOutputMatrix)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return outputBuffer
    }
    
    
    
    
    
    
    
    
    
    
    
    
    public func train(_ inputs: [[Float]], _ targets: [[Float]], epochs: Int, learningRate: Float) {
        
        print("train")
        
        
        
        for _ in 0..<epochs {
            for i in 0..<inputs.count {
                let input = inputs[i]
                let target = targets[i]
                
                // Forward propagation
                let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
                let hiddenLayerOutputBuffer = calculateHiddenLayerOutput(inputBuffer)
                let outputBuffer = calculateOutput(hiddenLayerOutputBuffer)
                
                // Copy result from MTLBuffer to [Float]
                let outputPointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
                let output = Array(UnsafeBufferPointer(start: outputPointer, count: outputSize))
                
                // Backpropagation
                // Calculate output layer error
                var outputError = [Float](repeating: 0.0, count: outputSize)
                for j in 0..<outputSize {
                    outputError[j] = (target[j] - output[j]) * output[j] * (1 - output[j])
                }
                
                // Calculate hidden layer error
                var hiddenError = [Float](repeating: 0.0, count: hiddenSize)
                for j in 0..<hiddenSize {
                    var sum = Float(0.0)
                    for k in 0..<outputSize {
                        sum += outputError[k] * (weightsHiddenOutput.contents().assumingMemoryBound(to: Float.self) + j * outputSize + k).pointee
                    }
                    hiddenError[j] = sum * (hiddenLayerOutputBuffer.contents().assumingMemoryBound(to: Float.self) + j).pointee * (1 - (hiddenLayerOutputBuffer.contents().assumingMemoryBound(to: Float.self) + j).pointee)
                }
                
                // Update weights and biases
                for j in 0..<hiddenSize {
                    for k in 0..<outputSize {
                        (weightsHiddenOutput.contents().assumingMemoryBound(to: Float.self) + j * outputSize + k).pointee += learningRate * outputError[k] * (hiddenLayerOutputBuffer.contents().assumingMemoryBound(to: Float.self) + j).pointee
                    }
                }
                for j in 0..<inputSize {
                    for k in 0..<hiddenSize {
                        (weightsInputHidden.contents().assumingMemoryBound(to: Float.self) + j * hiddenSize + k).pointee += learningRate * hiddenError[k] * input[j]
                    }
                }
                for j in 0..<outputSize {
                    (biasesOutput.contents().assumingMemoryBound(to: Float.self) + j).pointee += learningRate * outputError[j]
                }
                for j in 0..<hiddenSize {
                    (biasesHidden.contents().assumingMemoryBound(to: Float.self) + j).pointee += learningRate * hiddenError[j]
                }
            }
        }
    }
}

