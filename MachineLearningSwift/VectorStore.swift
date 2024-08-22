//
//  VectorStore.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 22.08.24.
//

import Foundation

struct VectorStore {
    
    var documents: [String]
    var vectors: [String: [Double]]

    init(documents: [String]) {
        self.documents = documents
        self.vectors = [:]
        self.build()
    }

    mutating func build() {
        for doc in documents {
            let vector = simpleEncode(doc: doc)
            vectors[doc] = vector
        }
    }

    func simpleEncode(doc: String) -> [Double] {
        // Simple encoding: counts of each character as a feature
        var vector = [Double](repeating: 0.0, count: 26)
        for ch in doc.lowercased().filter({ $0.isLetter }) {
            let idx = Int(ch.asciiValue! - Character("a").asciiValue!)
            vector[idx] += 1.0
        }
        return vector
    }

    func cosineSimilarity(u: [Double], v: [Double]) -> Double {
        let dotProduct = zip(u, v).map(*).reduce(0, +)
        let normU = sqrt(u.map { $0 * $0 }.reduce(0, +))
        let normV = sqrt(v.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (normU * normV)
    }

    func getTopN(query: String, n: Int) -> [(String, Double)] {
        let embeddedQuery = simpleEncode(doc: query)
        var scores: [(String, Double)] = vectors.map { (doc, vec) in
            let similarity = cosineSimilarity(u: embeddedQuery, v: vec)
            return (doc, similarity)
        }
        
        scores.sort { $0.1 > $1.1 }
        return Array(scores.prefix(n))
    }
}



