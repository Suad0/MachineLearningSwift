//
//  ChatBotLSTM.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 21.11.24.
//

import Foundation
import CoreML

class ChatBotLSTM {
    
    private let coreMLModel: chatbot_model
    private let lstmCell: xLSTMCell
    private var tokenizer: Tokenizer
    
        private static let jsonFilePath = "/Users/suad/PycharmProjects/chatbot_flask/data/intents_small.json"
        
        init?(coreMLModel: chatbot_model, lstmInputSize: Int, lstmHiddenSize: Int) {
            self.coreMLModel = coreMLModel
            self.lstmCell = xLSTMCell(inputSize: lstmInputSize, hiddenSize: lstmHiddenSize, memorySize: 734)
            
            guard let tokenizer = Tokenizer(fromJSONFile: Self.jsonFilePath) else {
                return nil
            }
            self.tokenizer = tokenizer
        }
        
    func preprocessInput(_ input: String) -> MLMultiArray? {
        guard let tokenizedInput = tokenizer.tokenize(input) else {
            return nil
        }
        
        // Ensure the input vector matches the model's input size
        let expectedSize = 734
        let paddedInput = tokenizedInput + [Float](repeating: 0, count: max(0, expectedSize - tokenizedInput.count))
        let truncatedInput = paddedInput.prefix(expectedSize)
        
        do {
            let multiArray = try MLMultiArray(shape: [1, 1, NSNumber(value: expectedSize)], dataType: .float32)
            for (index, value) in truncatedInput.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }
            return multiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }

    
    func getResponse(for input: String) -> String {
        guard let processedInput = preprocessInput(input) else {
            return "Sorry, I couldn't process your input."
        }
        
        do {
            // Use CoreML model to get prediction
            let output = try coreMLModel.prediction(input_text: processedInput)
            
            // Convert output to response
            // Use .first or safe indexing for the shaped array
            guard let predictedTag = output.reduce_argmax_0ShapedArray.scalars.first else {
                return "I couldn't understand that."
            }
            
            return tokenizer.getResponseForTag(Int(predictedTag))
        } catch {
            return "An error occurred: \(error.localizedDescription)"
        }
    }
    
    func chat() {
        print("Welcome to the LSTM-powered Chatbot!")
        
        while true {
            print("You: ", terminator: "")
            guard let input = readLine(), !input.isEmpty else {
                continue
            }
            
            if input.lowercased() == "quit" {
                break
            }
            
            let response = getResponse(for: input)
            print("Bot: \(response)")
        }
    }
}
