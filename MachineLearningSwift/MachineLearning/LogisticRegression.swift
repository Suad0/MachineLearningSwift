//
//  LogisticRegression.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 27.05.24.
//

import Foundation

class LogisticRegression {
    
    private var coefficients: [Double] = []
    
    init(){}
    
    func train(X_train: [[Double]], y_train: [Double], learningRate: Double, iterations: Int) {
        let numFeatures = X_train[0].count
        coefficients = Array(repeating: 0.0, count: numFeatures + 1)
        
        
        let augmentedX_train = X_train.map { [1.0] + $0 }
        
        // Gradient Descent
        for _ in 0..<iterations {
            let predictions = predictProbability(X: augmentedX_train)
            
            let errors = zip(predictions, y_train).map { $0 - $1 }
            
            for j in 0..<numFeatures + 1 { // Include bias term
                var gradient = 0.0
                for i in 0..<X_train.count {
                    gradient += errors[i] * augmentedX_train[i][j]
                }
                coefficients[j] -= learningRate * gradient / Double(X_train.count)
            }
        }
    }
    
    private func sigmoid(z: Double) -> Double {
        return 1 / (1 + exp(-z))
    }
    
    private func predictProbability(X: [[Double]]) -> [Double] {
        return X.map { row in
            let linearCombination = zip(coefficients, row).reduce(0.0) { $0 + $1.0 * $1.1 }
            return sigmoid(z: linearCombination)
        }
    }
    
    func predict(X_test: [[Double]]) -> [Double] {
        let augmentedX_test = X_test.map { [1.0] + $0 }
        return predictProbability(X: augmentedX_test)
    }
    
    
}
