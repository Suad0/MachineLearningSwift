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
    
    private func sigmoid(inputBuffer: MTLBuffer, length: Int) -> MTLBuffer {
        // Create a buffer to store the output
        let outputBuffer = device.makeBuffer(length: length, options: .storageModeShared)!
        
        // MPS command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Define a MPSNeuronSigmoid operation
        let neuronDescriptor = MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .sigmoid, a: 1.0, b: 1.0)
        
        let sigmoidNeuron = MPSMatrixNeuron(device: device)
        
        // Define matrix descriptors
        let inputDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let outputDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        
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
    
    
    
    func forward(input: MTLBuffer) -> MTLBuffer {
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Define matrix descriptors for input and hidden layers
        let inputDescriptor = MPSMatrixDescriptor(rows: inputSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        let hiddenDescriptor = MPSMatrixDescriptor(rows: hiddenSize, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
        
        // Matrix multiplications for forget gate
        let forgetGateBuffer = device.makeBuffer(length: hiddenSize * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        let forgetMultiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: hiddenSize,
            resultColumns: 1,
            interiorColumns: inputSize,
            alpha: 1.0,
            beta: 0.0
        )
        
        // Encode matrix multiplication for forget gate
        forgetMultiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: MPSMatrix(buffer: Wf, descriptor: MPSMatrixDescriptor(rows: hiddenSize, columns: inputSize, rowBytes: inputSize * MemoryLayout<Float>.stride, dataType: .float32)),
            rightMatrix: MPSMatrix(buffer: input, descriptor: inputDescriptor),
            resultMatrix: MPSMatrix(buffer: forgetGateBuffer, descriptor: hiddenDescriptor)
        )
        
        // Apply sigmoid to forget gate buffer
        let forgetGateSigmoidBuffer = sigmoid(inputBuffer: forgetGateBuffer  , length: hiddenSize * MemoryLayout<Float>.stride)
        
        // Repeat similar steps for input, output, and cell gates (Wi, Wo, Wc, etc.)
        
        // Update hidden state h_t and cell state c_t
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return h_t  // Assuming h_t is the updated hidden state buffer
    }
    
}
