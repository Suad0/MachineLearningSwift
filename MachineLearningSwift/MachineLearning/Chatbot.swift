//
//  Chatbot.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 16.11.24.
//


/**
 
 Key Concepts and Techniques:
 
 Tokenization: Breaking down sentences into individual words.
 Bag of Words: Representing text data by creating a vector that marks the presence (or absence) of each unique word in the input.
 Padding and Trimming: Ensuring that input vectors have a consistent size.
 MLMultiArray: A data structure used by Core ML to represent multi-dimensional arrays, which is required for predictions.
 Core ML Model: A pre-trained machine learning model that classifies user input into one of the predefined categories (tags).
 
 What Happens in Practice:
 When a user inputs a message, the program tokenizes it, converts the tokens into a bag-of-words vector, and ensures the vector has the correct size.
 This vector is then fed into a Core ML model, which classifies the input into one of the defined categories (tags).
 Finally, the predicted tag is returned and printed out.
 Summary:
 Goal: The chatbot predicts the appropriate tag for a user's input (e.g., greeting, project management question, etc.).
 Process: Tokenizing input, converting it to a vector, padding/trimming it, feeding it into a Core ML model, and interpreting the output.
 
 
 */




import Foundation
import CoreML

class Chatbot {
    
    private var allWords: [String]
    public var tags: [String]
    private var responses: [String: [String]]
    private var mlModel: chatbot_model?
    private var synonymDictionary: [String: [String]]

    private static let jsonFilePath = "/Users/suad/PycharmProjects/chatbot_flask/data/intents_small.json"

    init() throws {
        let (loadedWords, loadedTags, loadedResponses) = try Chatbot.loadIntents()
        self.allWords = loadedWords
        self.tags = loadedTags
        self.responses = loadedResponses
        self.synonymDictionary = Chatbot.loadSynonyms()

        self.mlModel = try? chatbot_model()
        if mlModel == nil {
            throw NSError(domain: "ChatbotInitializationError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to load Core ML model."])
        }

        print("[DEBUG] Chatbot initialized with \(allWords.count) words and \(tags.count) tags.")
    }

    func predictTag(for input: String) -> String? {
        let preprocessedInput = preprocessSentence(input)
        let bowVector = bagOfWords(preprocessedInput, allWords: allWords)

        // chatbot_model input_text: 1 × 1 × 734
        let requiredSize = 734
        var paddedBowVector = bowVector
        if bowVector.count < requiredSize {
            paddedBowVector += Array(repeating: Float(0), count: requiredSize - bowVector.count)
        } else if bowVector.count > requiredSize {
            paddedBowVector = Array(bowVector.prefix(requiredSize))
        }

        guard let mlInput = try? MLMultiArray(shape: [1, 1, requiredSize] as [NSNumber], dataType: .float32) else {
            print("[ERROR] Failed to create MLMultiArray.")
            return nil
        }

        for (index, value) in paddedBowVector.enumerated() {
            mlInput[index] = NSNumber(value: value)
        }

        do {
            let predictionOutput = try mlModel?.prediction(input_text: mlInput)
            guard let rawOutput = predictionOutput?.reduce_argmax_0 else {
                print("[ERROR] Failed to retrieve prediction output.")
                return nil
            }

            let outputArray = mlMultiArrayToArray(rawOutput)
            guard let predictedIndex = outputArray.first else {
                print("[ERROR] Failed to extract predicted index.")
                return nil
            }

            let predictedTag = tags[Int(predictedIndex)]
            print("[DEBUG] Predicted tag: \(predictedTag)")
            return predictedTag
        } catch {
            print("[ERROR] Prediction error: \(error)")
            return nil
        }
    }


    func getResponse(for tag: String) -> String? {
        if let responsesForTag = responses[tag], let randomResponse = responsesForTag.randomElement() {
            return randomResponse
        }
        return "I'm sorry, I don't understand."
    }

    func chat() {
        print("Chatbot is ready! Type 'exit' to quit.")
        while true {
            print("You: ", terminator: "")
            guard let input = readLine(), input.lowercased() != "exit" else {
                print("Exiting the chat. Goodbye!")
                break
            }
            if let tag = predictTag(for: input) {
                if let response = getResponse(for: tag) {
                    print("Chatbot: \(response)")
                }
            } else {
                print("Chatbot: I didn't understand that.")
            }
        }
    }

    static func loadIntents() throws -> ([String], [String], [String: [String]]) {
        let filePath = Chatbot.jsonFilePath
        print("[DEBUG] Loading intents from \(filePath)")
        let fileURL = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: fileURL)

        let intentData = try JSONDecoder().decode(IntentData.self, from: data)
        print("[DEBUG] JSON decoded successfully with \(intentData.intents.count) intents.")

        var responses = [String: [String]]()
        for intent in intentData.intents {
            responses[intent.tag] = intent.responses
        }

        var allWordsSet = Set<String>()
        for intent in intentData.intents {
            for pattern in intent.patterns {
                let words = self.tokenize(sentence: pattern)
                allWordsSet.formUnion(words)
            }
        }
        let allWords = Array(allWordsSet).sorted()
        let tags = intentData.intents.map { $0.tag }
        return (allWords, tags, responses)
    }

    static func loadSynonyms() -> [String: [String]] {
        return [
            "hello": ["hi", "hey", "greetings"],
            "bye": ["goodbye", "farewell", "see you"],
            "help": ["assist", "support", "aid"]
        ]
    }

    func preprocessSentence(_ sentence: String) -> [String] {
        let tokenized = Chatbot.tokenize(sentence: sentence)
        return expandWithSynonymsAndStem(tokenized)
    }

    static func tokenize(sentence: String) -> [String] {
        let pattern = "\\w+"
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            print("[ERROR] Failed to create regular expression for tokenizing.")
            return []
        }
        let range = NSRange(location: 0, length: sentence.utf16.count)
        return regex.matches(in: sentence, range: range).map {
            String(sentence[Range($0.range, in: sentence)!]).lowercased()
        }
    }

    private func expandWithSynonymsAndStem(_ tokenizedSentence: [String]) -> [String] {
        var expanded = [String]()
        for word in tokenizedSentence {
            expanded.append(stem(word))
            if let synonyms = synonymDictionary[word] {
                expanded.append(contentsOf: synonyms.map { stem($0) })
            }
        }
        return expanded
    }

    private func stem(_ word: String) -> String {
        return word.lowercased() // Replace with actual stemming logic if needed
    }

    private func bagOfWords(_ tokenizedSentence: [String], allWords: [String]) -> [Float] {
        var bag = [Float](repeating: 0.0, count: allWords.count)
        for word in tokenizedSentence {
            if let index = allWords.firstIndex(of: word) {
                bag[index] = 1.0
            }
        }
        return bag
    }

    func mlMultiArrayToArray(_ multiArray: MLMultiArray) -> [Int] {
        let count = multiArray.count
        var array = [Int](repeating: 0, count: count)
        for i in 0..<count {
            array[i] = Int(truncating: multiArray[i])
        }
        return array
    }
}

// JSON parsing structures
struct Intent: Codable {
    let tag: String
    let patterns: [String]
    let responses: [String]
}

struct IntentData: Codable {
    let intents: [Intent]
}
