//
//  GPT.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 16.01.25.
//

import Foundation

// MARK: - Constants
struct TransformerConfig {
    static let validation = true
    static let encFileSize = 722883
    static let dataSizeMax = 1000000
    static let safetensorFileSize = 548105171
    static let safetensorJsonSize = 14283
    static let vocabSize = 50257
    static let sequenceLength = 1024
    static let modelDim = 768
    static let keyDim = 64
    static let numHeads = 12
    static let numLayers = 12
    static let rsqrtKeyDim: Float = 0.125
}

// MARK: - Decoder Item
struct DecoderItem {
    var offset: UInt32
    var size: UInt32
}

// MARK: - Decoder
struct Decoder {
    var items: [DecoderItem]
    var raw: Data
    
    init() {
        self.items = Array(repeating: DecoderItem(offset: 0, size: 0), count: TransformerConfig.vocabSize)
        self.raw = Data(count: TransformerConfig.encFileSize - TransformerConfig.vocabSize * MemoryLayout<DecoderItem>.size)
    }
}

// MARK: - Parameters
class Parameters {
    class Embedding {
        var weight: [Float] = []
    }
    
    class LayerNormalization {
        var bias: [Float] = []
        var weight: [Float] = []
    }
    
    class Attention {
        class Dense {
            var bias: [Float] = []
            var weight: [[Float]] = []
        }
        var cAttn = Dense()
        var cProj = Dense()
    }
    
    class FeedForward {
        class Dense {
            var bias: [Float] = []
            var weight: [[Float]] = []
        }
        var cFc = Dense()
        var cProj = Dense()
    }
    
    class Layer {
        var ln1 = LayerNormalization()
        var attention = Attention()
        var ln2 = LayerNormalization()
        var feedForward = FeedForward()
    }
    
    var wte = Embedding()
    var wpe = Embedding()
    var layers: [Layer] = Array(repeating: Layer(), count: TransformerConfig.numLayers)
    var lnF = LayerNormalization()
}

// MARK: - Transformer Operations
class TransformerOperations {
    var decoder = Decoder()
    var parameters = Parameters()
    
    func forward(input: [UInt16]) -> [[Float]] {
        // Example implementation of embedding layer forward pass
        var embeddings: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: TransformerConfig.modelDim),
            count: input.count
        )
        
        for (i, token) in input.enumerated() {
            let wteOffset = Int(token) * TransformerConfig.modelDim
            let wpeOffset = i * TransformerConfig.modelDim
            let wte = Array(parameters.wte.weight[wteOffset..<wteOffset + TransformerConfig.modelDim])
            let wpe = Array(parameters.wpe.weight[wpeOffset..<wpeOffset + TransformerConfig.modelDim])
            
            embeddings[i] = zip(wte, wpe).map { $0 + $1 }
        }
        
        return embeddings
    }
}
