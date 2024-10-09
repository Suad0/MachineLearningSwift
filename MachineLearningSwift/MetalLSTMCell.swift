//
//  MetalLSTMCell.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 08.10.24.
//

import Foundation
import Metal
import MetalPerformanceShaders

class MetalLSTMCell {
    
    private var inputSize: Int
    private var hiddenSize: Int
    public var device: MTLDevice
    private var commandQueue: MTLCommandQueue
    
    // Buffers for LSTM weights
    private var Wf: MTLBuffer
    private var Wi: MTLBuffer
    private var Wo: MTLBuffer
    private var Wc: MTLBuffer
    private var Uf: MTLBuffer
    private var Ui: MTLBuffer
    private var Uo: MTLBuffer
    private var Uc: MTLBuffer
    private var bf: MTLBuffer
    private var bi: MTLBuffer
    private var bo: MTLBuffer
    private var bc: MTLBuffer
    
    private var h_t: MTLBuffer // Hidden state
    private var c_t: MTLBuffer // Cell state
    
    init(inputSize: Int, hiddenSize: Int) {
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Allocate space for LSTM weights and states
        Wf = device.makeBuffer(length: inputSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Wi = device.makeBuffer(length: inputSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Wo = device.makeBuffer(length: inputSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Wc = device.makeBuffer(length: inputSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        Uf = device.makeBuffer(length: hiddenSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Ui = device.makeBuffer(length: hiddenSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Uo = device.makeBuffer(length: hiddenSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        Uc = device.makeBuffer(length: hiddenSize * hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        bf = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        bi = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        bo = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        bc = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Hidden and cell states
        h_t = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        c_t = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
    }
    
    // Custom element-wise addition
    private func elementWiseAdd(bufferA: MTLBuffer, bufferB: MTLBuffer, resultBuffer: MTLBuffer, length: Int) {
        let aPointer = bufferA.contents().bindMemory(to: Float.self, capacity: length)
        let bPointer = bufferB.contents().bindMemory(to: Float.self, capacity: length)
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: length)
        
        for i in 0..<length {
            resultPointer[i] = aPointer[i] + bPointer[i]
        }
    }
    
    // Custom element-wise multiplication
    private func elementWiseMultiply(bufferA: MTLBuffer, bufferB: MTLBuffer, resultBuffer: MTLBuffer, length: Int) {
        let aPointer = bufferA.contents().bindMemory(to: Float.self, capacity: length)
        let bPointer = bufferB.contents().bindMemory(to: Float.self, capacity: length)
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: length)
        
        for i in 0..<length {
            resultPointer[i] = aPointer[i] * bPointer[i]
        }
    }
    
    // Matrix multiplication for each gate using MPS
    private func matrixMultiply(weightBuffer: MTLBuffer, inputBuffer: MTLBuffer, resultBuffer: MTLBuffer, rows: Int, columns: Int) {
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
    
    func forward(input: MTLBuffer) -> MTLBuffer {
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Buffers for gates
        let forgetGateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let inputGateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let outputGateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let cellGateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Perform matrix multiplications
        matrixMultiply(weightBuffer: Wf, inputBuffer: input, resultBuffer: forgetGateBuffer, rows: hiddenSize, columns: inputSize)
        matrixMultiply(weightBuffer: Wi, inputBuffer: input, resultBuffer: inputGateBuffer, rows: hiddenSize, columns: inputSize)
        matrixMultiply(weightBuffer: Wo, inputBuffer: input, resultBuffer: outputGateBuffer, rows: hiddenSize, columns: inputSize)
        matrixMultiply(weightBuffer: Wc, inputBuffer: input, resultBuffer: cellGateBuffer, rows: hiddenSize, columns: inputSize)
        
        // Apply activations
        let forgetGateSigmoidBuffer = sigmoid(inputBuffer: forgetGateBuffer, length: hiddenSize)
        let inputGateSigmoidBuffer = sigmoid(inputBuffer: inputGateBuffer, length: hiddenSize)
        let outputGateSigmoidBuffer = sigmoid(inputBuffer: outputGateBuffer, length: hiddenSize)
        let cellGateTanhBuffer = tanh(inputBuffer: cellGateBuffer, length: hiddenSize)
        
        // Element-wise operations
        let updatedCellStateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        elementWiseMultiply(bufferA: forgetGateSigmoidBuffer, bufferB: c_t, resultBuffer: updatedCellStateBuffer, length: hiddenSize)
        elementWiseMultiply(bufferA: inputGateSigmoidBuffer, bufferB: cellGateTanhBuffer, resultBuffer: updatedCellStateBuffer, length: hiddenSize)
        
        // Update hidden state
        let updatedHiddenStateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        elementWiseMultiply(bufferA: outputGateSigmoidBuffer, bufferB: updatedCellStateBuffer, resultBuffer: updatedHiddenStateBuffer, length: hiddenSize)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Update internal state for the next timestep
        h_t = updatedHiddenStateBuffer
        c_t = updatedCellStateBuffer
        
        return updatedHiddenStateBuffer
    }
    
    
    
    private func sigmoid(inputBuffer: MTLBuffer, length: Int) -> MTLBuffer {
        // Create a buffer to store the output
        let outputBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // MPS command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Define a neuron descriptor for sigmoid
        //let neuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .sigmoid)
        
        // Create the MPSMatrixNeuron
        let sigmoidNeuron = MPSMatrixNeuron(device: device)
        
        // Define matrix descriptors
        let inputDescriptor = MPSMatrixDescriptor(rows: length, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let outputDescriptor = inputDescriptor
        
        // Create MPSMatrix objects
        let inputMatrix = MPSMatrix(buffer: inputBuffer, descriptor: inputDescriptor)
        let outputMatrix = MPSMatrix(buffer: outputBuffer, descriptor: outputDescriptor)
        
        // Apply sigmoid function
        sigmoidNeuron.encode(commandBuffer: commandBuffer, inputMatrix: inputMatrix, biasVector: nil, resultMatrix: outputMatrix)
        
        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Return the buffer with sigmoid results
        return outputBuffer
    }
    
    
    
    
    
    
    
    private func tanh(inputBuffer: MTLBuffer, length: Int) -> MTLBuffer {
        // Create a buffer to store the output
        let outputBuffer = device.makeBuffer(length: length * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // MPS command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Define a neuron descriptor for tanh
        //let neuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .tanH)
        
        // Create the MPSMatrixNeuron
        let tanhNeuron = MPSMatrixNeuron(device: device)
        
        // Define matrix descriptors
        let inputDescriptor = MPSMatrixDescriptor(rows: length, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let outputDescriptor = inputDescriptor
        
        // Create MPSMatrix objects
        let inputMatrix = MPSMatrix(buffer: inputBuffer, descriptor: inputDescriptor)
        let outputMatrix = MPSMatrix(buffer: outputBuffer, descriptor: outputDescriptor)
        
        // Apply tanh function
        tanhNeuron.encode(commandBuffer: commandBuffer, inputMatrix: inputMatrix, biasVector: nil, resultMatrix: outputMatrix)
        
        // Commit and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Return the buffer with tanh results
        return outputBuffer
    }
    
    
    
    
    
    
    
}



