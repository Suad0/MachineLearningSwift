//
//  MarkovChain.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 24.02.25.
//

import Foundation

class MarkovChain {
    // Stores transitions as [Current State: (Total Count, [Next State: Transition Count])]
    private var transitions: [String: (total: Int, nextStates: [String: Int])] = [:]
    
    /// Trains the Markov chain on a given sequence of states
    /// - Parameter states: The training sequence of states
    func train(on states: [String]) {
        guard states.count >= 2 else { return }
        
        for i in 0..<(states.count - 1) {
            let currentState = states[i]
            let nextState = states[i+1]
            
            // Get or create transition data for current state
            var transitionData = transitions[currentState] ?? (total: 0, nextStates: [:])
            
            // Update transition counts
            transitionData.total += 1
            transitionData.nextStates[nextState] = (transitionData.nextStates[nextState] ?? 0) + 1
            
            // Save updated data back to transitions
            transitions[currentState] = transitionData
        }
    }
    
    /// Generates the next state based on current state's transition probabilities
    /// - Parameter current: The current state
    /// - Returns: A randomly selected next state or nil if no transitions
    func generateNextState(after current: String) -> String? {
        guard let transition = transitions[current], transition.total > 0 else {
            return nil
        }
        
        // Generate random number within total transitions range
        let randomValue = Int.random(in: 0..<transition.total)
        var cumulative = 0
        
        // Iterate through possible next states to find the selected one
        for (state, count) in transition.nextStates {
            cumulative += count
            if randomValue < cumulative {
                return state
            }
        }
        
        return nil // Should never reach here if counts are consistent
    }
    
    /// Generates a sequence of states
    /// - Parameters:
    ///   - start: Optional starting state (random if nil)
    ///   - length: Desired length of the generated sequence
    /// - Returns: Generated sequence of states
    func generateSequence(startingWith start: String? = nil, length: Int) -> [String] {
        var sequence: [String] = []
        
        // Determine starting state
        var current = start ?? transitions.keys.randomElement()
        guard let startingState = current else {
            return sequence // Empty if no training data
        }
        
        sequence.append(startingState)
        
        // Generate subsequent states
        for _ in 1..<length {
            guard let nextState = generateNextState(after: current!) else {
                break // Stop if no more transitions
            }
            sequence.append(nextState)
            current = nextState
        }
        
        return sequence
    }
}

