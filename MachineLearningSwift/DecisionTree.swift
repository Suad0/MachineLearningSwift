//
//  DecisionTree.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 29.05.24.
//

import Foundation

class DecisionTree {
    
    private var root: Node?
    private var maxDepth: Int
    
    init(maxDepth: Int) {
        self.maxDepth = maxDepth
    }
    
    func train(X_train: [[Double]], y_train: [Int]) {
        root = buildTree(X_train: X_train, y_train: y_train, depth: 0)
    }
    
    func predict(instance: [Double]) -> Int {
        return classify(instance: instance, node: root!)
    }
    
    private func buildTree(X_train: [[Double]], y_train: [Int], depth: Int) -> Node {
        if depth >= maxDepth || allSame(array: y_train) {
            return Node(classLabel: mostCommonClass(array: y_train))
        }
        
        let bestSplit = findBestSplit(X_train: X_train, y_train: y_train)
        
        if bestSplit == nil {
            return Node(classLabel: mostCommonClass(array: y_train))
        }
        
        let leftChild = buildTree(X_train: bestSplit!.leftX, y_train: bestSplit!.leftY, depth: depth + 1)
        let rightChild = buildTree(X_train: bestSplit!.rightX, y_train: bestSplit!.rightY, depth: depth + 1)
        
        return Node(featureIndex: bestSplit!.featureIndex, threshold: bestSplit!.threshold, leftChild: leftChild, rightChild: rightChild)
    }
    
    private func allSame(array: [Int]) -> Bool {
        let first = array[0]
        for i in 1..<array.count {
            if array[i] != first {
                return false
            }
        }
        return true
    }
    
    private func mostCommonClass(array: [Int]) -> Int {
        var classCounts = [Int: Int]()
        for value in array {
            classCounts[value] = (classCounts[value] ?? 0) + 1
        }
        var mostCommonClass = -1
        var maxCount = Int.min
        for (key, value) in classCounts {
            if value > maxCount {
                maxCount = value
                mostCommonClass = key
            }
        }
        return mostCommonClass
    }
    
    private func findBestSplit(X_train: [[Double]], y_train: [Int]) -> Split? {
        let numFeatures = X_train[0].count
        let numRows = X_train.count
        var bestGini = Double.greatestFiniteMagnitude
        var bestFeatureIndex = -1
        var bestThreshold = 0.0
        
        var leftY = [Int](repeating: 0, count: numRows)
        var leftX = [[Double]](repeating: [Double](repeating: 0.0, count: numFeatures), count: numRows)
        var leftIndex = 0
        
        var rightY = [Int](repeating: 0, count: numRows)
        var rightX = [[Double]](repeating: [Double](repeating: 0.0, count: numFeatures), count: numRows)
        var rightIndex = 0
        
        for featureIndex in 0..<numFeatures {
            for rowIndex in 0..<numRows {
                let threshold = X_train[rowIndex][featureIndex]
                
                leftIndex = 0
                rightIndex = 0
                
                for i in 0..<numRows {
                    if X_train[i][featureIndex] <= threshold {
                        leftY[leftIndex] = y_train[i]
                        leftX[leftIndex] = X_train[i]
                        leftIndex += 1
                    } else {
                        rightY[rightIndex] = y_train[i]
                        rightX[rightIndex] = X_train[i]
                        rightIndex += 1
                    }
                }
                
                let gini = calculateGini(groups: leftY, rightY)
                if gini < bestGini {
                    bestGini = gini
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold
                }
            }
        }
        
        if bestFeatureIndex == -1 {
            return nil
        }
        
        return Split(featureIndex: bestFeatureIndex, threshold: bestThreshold, leftX: leftX, leftY: leftY, rightX: rightX, rightY: rightY)
    }
    
    private func calculateGini(groups: [Int]...) -> Double {
        var totalInstances = 0
        for group in groups {
            totalInstances += group.count
        }
        
        var gini = 0.0
        for group in groups {
            let groupSize = Double(group.count)
            if groupSize == 0 {
                continue
            }
            var score = 0.0
            for classLabel in group {
                let p = Double(countOccurrences(value: classLabel, array: group)) / groupSize
                score += p * p
            }
            gini += (1.0 - score) * (groupSize / Double(totalInstances))
        }
        return gini
    }
    
    private func countOccurrences(value: Int, array: [Int]) -> Int {
        var count = 0
        for v in array {
            if v == value {
                count += 1
            }
        }
        return count
    }
    
    private func classify(instance: [Double], node: Node) -> Int {
        if node.isLeaf() {
            return node.classLabel
        }
        
        if instance[node.featureIndex!] <= node.threshold! {
            return classify(instance: instance, node: node.leftChild!)
        } else {
            return classify(instance: instance, node: node.rightChild!)
        }
    }
    
    class Node {
        var featureIndex: Int?
        var threshold: Double?
        var classLabel: Int
        var leftChild: Node?
        var rightChild: Node?
        
        init(classLabel: Int) {
            self.classLabel = classLabel
        }
        
        init(featureIndex: Int, threshold: Double, leftChild: Node, rightChild: Node) {
            self.featureIndex = featureIndex
            self.threshold = threshold
            self.leftChild = leftChild
            self.rightChild = rightChild
            self.classLabel = -1 // Dummy initialization, not used for non-leaf nodes in classification
        }
        
        func isLeaf() -> Bool {
            return leftChild == nil && rightChild == nil
        }
    }
    
    struct Split {
        var featureIndex: Int
        var threshold: Double
        var leftX: [[Double]]
        var leftY: [Int]
        var rightX: [[Double]]
        var rightY: [Int]
    }
}
