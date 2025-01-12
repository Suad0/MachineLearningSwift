//
//  ChatBotLSTMTalking.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 23.12.24.
//

import Foundation
import CoreML
import Speech
import AVFoundation

class ChatBotLSTMTalking: NSObject {
    
    private let coreMLModel: chatbot_model
    private let lstmCell: LSTMCell
    private var tokenizer: Tokenizer
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    
    private static let jsonFilePath = "/Users/suad/PycharmProjects/chatbot_flask/data/intents_small.json"
    
    init?(coreMLModel: chatbot_model, lstmInputSize: Int, lstmHiddenSize: Int) {
        self.coreMLModel = coreMLModel
        self.lstmCell = LSTMCell(inputSize: lstmInputSize, hiddenSize: lstmHiddenSize)
        
        guard let tokenizer = Tokenizer(fromJSONFile: Self.jsonFilePath) else {
            return nil
        }
        self.tokenizer = tokenizer
    }
    
    func preprocessInput(_ input: String) -> MLMultiArray? {
        guard let tokenizedInput = tokenizer.tokenize(input) else {
            return nil
        }
        
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
            let output = try coreMLModel.prediction(input_text: processedInput)
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
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                switch status {
                case .authorized:
                    print("Speech recognition authorized.")
                    self.startListening()
                case .denied:
                    print("Speech recognition authorization denied.")
                case .restricted:
                    print("Speech recognition restricted on this device.")
                case .notDetermined:
                    print("Speech recognition authorization not determined.")
                @unknown default:
                    print("Unknown authorization status.")
                }
            }
        }
    }


    
    // MARK: - Speech-to-Text (STT)
    func startListening() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                if status == .authorized {
                    self.listenForInput()
                } else {
                    print("Speech recognition not authorized.")
                }
            }
        }
    }
    
    private func listenForInput() {
        do {
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else {
                print("Unable to create a recognition request.")
                return
            }
            
            let inputNode = audioEngine.inputNode
            recognitionRequest.shouldReportPartialResults = true
            
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { result, error in
                if let result = result {
                    let recognizedText = result.bestTranscription.formattedString
                    print("You: \(recognizedText)")
                    
                    // Process chatbot response
                    let response = self.getResponse(for: recognizedText)
                    print("Bot: \(response)")
                    self.speakText(response)
                }
                
                if error != nil || result?.isFinal == true {
                    self.audioEngine.stop()
                    inputNode.removeTap(onBus: 0)
                    self.recognitionRequest = nil
                    self.recognitionTask = nil
                }
            }
            
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, when in
                self.recognitionRequest?.append(buffer)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            print("Listening for your question...")
            
        } catch {
            print("Error starting the audio engine: \(error.localizedDescription)")
        }
    }


    
    // MARK: - Text-to-Speech (TTS)
    func speakText(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        synthesizer.speak(utterance)
    }
    
    
    
}
