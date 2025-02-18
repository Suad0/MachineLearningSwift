//
//  Tokenizer.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 06.12.24.
//

import Foundation

class Tokenizer {
    
    private var wordIndex: [String: Int] = [:]
    private var tagIndex: [String: Int] = [:]
    private var intents: [Intent] = []
    
    struct Intent: Decodable {
        let tag: String
        let patterns: [String]
        let responses: [String]
    }
    
    init?(fromJSONFile path: String) {
        print("Attempting to load JSON file from path: \(path)")
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            print("File loaded successfully.")
            
            // Decode the top-level dictionary and then the intents array
            let topLevel = try JSONDecoder().decode([String: [Intent]].self, from: data)
            guard let intents = topLevel["intents"] else {
                print("No 'intents' key found in the JSON file.")
                return nil
            }
            
            self.intents = intents
            buildVocabulary()
        } catch {
            print("Error loading JSON file: \(error)")
            return nil
        }
    }


    
    private func buildVocabulary() {
        // Build word and tag indices
        var words = Set<String>()
        var tags = Set<String>()
        
        for intent in intents {
            tags.insert(intent.tag)
            for pattern in intent.patterns {
                words.formUnion(pattern.lowercased().components(separatedBy: .whitespacesAndNewlines))
            }
        }
        
        wordIndex = Dictionary(uniqueKeysWithValues: words.enumerated().map { ($0.element, $0.offset) })
        tagIndex = Dictionary(uniqueKeysWithValues: tags.enumerated().map { ($0.element, $0.offset) })
    }
    
    func tokenize(_ input: String) -> [Float]? {
        let words = input.lowercased().components(separatedBy: .whitespacesAndNewlines)
        var tokenVector = [Float](repeating: 0, count: wordIndex.count)
        
        for word in words {
            if let index = wordIndex[word] {
                tokenVector[index] = 1.0
            }
        }
        
        return tokenVector
    }
    
    func getResponseForTag(_ tagIndex: Int) -> String {
        guard tagIndex < intents.count else {
            return "I'm not sure how to respond."
        }
        
        let intent = intents[tagIndex]
        return intent.responses.randomElement() ?? "Hello!"
    }
    
    
}
