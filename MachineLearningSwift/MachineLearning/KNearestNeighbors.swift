//
//  KNearestNeighbors.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation

class KNearestNeighbors {
    private var trainingData: [[Double]] = []
    private var trainingLabels: [Int] = []
    private var k: Int
    
    init(k: Int) {
        self.k = k
    }
    
    // Fit the model with training data and labels
    func fit(data: [[Double]], labels: [Int]) {
        self.trainingData = data
        self.trainingLabels = labels
    }
    
    // Predict the label for a new data point
    func predict(data: [[Double]]) -> [Int] {
        return data.map { predictSingle($0) }
    }
    
    private func predictSingle(_ point: [Double]) -> Int {
        let distances = trainingData.map { euclideanDistance($0, point) }
        let sortedIndices = distances.enumerated().sorted(by: { $0.element < $1.element }).map { $0.offset }
        let kNearestLabels = sortedIndices.prefix(k).map { trainingLabels[$0] }
        
        return mostCommonLabel(kNearestLabels)
    }
    
    private func euclideanDistance(_ a: [Double], _ b: [Double]) -> Double {
        return sqrt(zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    private func mostCommonLabel(_ labels: [Int]) -> Int {
        var frequency: [Int: Int] = [:]
        for label in labels {
            frequency[label, default: 0] += 1
        }
        return frequency.max(by: { $0.value < $1.value })?.key ?? 0
    }
}

