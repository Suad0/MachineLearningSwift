//
//  RestrictedBoltzmannMachine.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 14.10.24.
//



/**
 
 
 A Boltzmann Machine (BM) is a type of stochastic (probabilistic) neural network that is used for learning and representing complex patterns in data. It's named after the Boltzmann distribution from statistical physics, as it models probability distributions over its inputs. Boltzmann Machines are particularly useful in tasks involving unsupervised learning, such as feature extraction or data generation. Let me break it down step by step:
 
 Key Concepts of a Boltzmann Machine:
 Neurons:
 
 A Boltzmann Machine consists of neurons (or units) that are either in an active state (on) or an inactive state (off). These neurons are typically binary (0 or 1).
 Neurons are divided into two layers:
 Visible units: These are the input nodes where data is provided.
 Hidden units: These are internal nodes that are not directly observed, and their purpose is to learn and capture underlying patterns in the data.
 Connections:
 
 Neurons in a Boltzmann Machine are connected to each other through weighted edges. Each connection has an associated weight that determines the strength of interaction between two neurons.
 Undirected connections: The connections are symmetric, meaning if neuron A influences neuron B, neuron B can also influence neuron A.
 Energy-Based Model:
 
 Boltzmann Machines are energy-based models. The idea is to define a system that has an "energy" associated with each configuration of neuron states.
 The goal during training is to lower the system’s energy, which means finding the most likely configuration of neurons that represents the input data. This is akin to how physical systems tend to move toward states of lower energy (like a ball rolling down to the lowest point in a valley).
 The energy function of the system depends on the states of the neurons and the weights of the connections.
 Probabilistic Model:
 
 The probability of a particular configuration (set of states) of the neurons is governed by the Boltzmann distribution:
 𝑃(state)=𝑒^−Energy(state)/𝑍
 
 ​
 
 where:

 e^−Energy(state)
 is a measure of how "low" the energy of that state is.
 
 Z is a normalization constant (called the partition function) that ensures the probabilities add up to 1.
 Neurons are updated in a probabilistic manner based on the energies of their neighboring neurons.
 Training a Boltzmann Machine:
 The goal of training is to adjust the weights between the neurons so that the model learns a probability distribution that reflects the input data.
 
 Positive Phase:
 
 The machine is shown input data, and the visible neurons are set to correspond to the input. The hidden neurons are then updated based on this input, and the system tries to lower its energy to fit the data pattern.
 Negative Phase:
 
 After updating the hidden neurons, the machine tries to "reconstruct" the visible layer using the hidden neurons. This reconstructed data should ideally be close to the original input data if the machine is well-trained.
 Contrastive Divergence (CD):
 
 A popular training method is Contrastive Divergence, which compares the visible neurons after one reconstruction step (negative phase) with the original input data (positive phase) and updates the weights based on their difference.
 Types of Boltzmann Machines:
 Standard Boltzmann Machine:
 
 In the original Boltzmann Machine, every neuron is connected to every other neuron, making it a fully connected network. This is very powerful but computationally expensive to train.
 Restricted Boltzmann Machine (RBM):
 
 In a Restricted Boltzmann Machine (RBM), which is the version you are implementing, the connections are limited: only visible neurons and hidden neurons are connected, but there are no connections within the same layer (i.e., no visible-to-visible or hidden-to-hidden connections). This restriction makes the RBM much easier and faster to train.
 RBMs are commonly used for feature learning, dimensionality reduction, and as building blocks for more complex models like Deep Belief Networks (DBNs).
 Applications of Boltzmann Machines:
 Dimensionality Reduction: RBMs are often used to reduce the complexity of large datasets by learning a lower-dimensional representation of the input.
 Recommendation Systems: RBMs can be used in collaborative filtering techniques, like in Netflix or Amazon, to recommend products based on learned user preferences.
 Generative Models: Once trained, Boltzmann Machines can generate new data samples similar to the input data they were trained on.
 Feature Extraction: RBMs are used to automatically discover useful features in the input data.
 Challenges:
 Training complexity: Boltzmann Machines, especially standard ones, can be slow to train due to the probabilistic nature of the neuron updates and the complexity of calculating energy and gradients.
 Sampling: Efficient sampling methods, like Gibbs sampling or Persistent Contrastive Divergence (PCD), are needed to make training feasible.
 In summary, Boltzmann Machines are powerful probabilistic models that learn by lowering the energy of the system to capture patterns in the data. RBMs, a simplified version, are commonly used in practice due to their tractability and efficiency, particularly for tasks like feature learning and data generation.
  
 */



import Foundation

class RestrictedBoltzmannMachine {
    var numVisible: Int
    var numHidden: Int
    var weights: [[Double]]
    var hiddenBias: [Double]
    var visibleBias: [Double]
    
    // Improved Initialization (Xavier)
    init(numVisible: Int, numHidden: Int) {
        self.numVisible = numVisible
        self.numHidden = numHidden
        let limit = sqrt(6.0 / Double(numVisible + numHidden))  // Xavier Initialization limit
        
        // Xavier Initialization for weights
        self.weights = (0..<numVisible).map { _ in (0..<numHidden).map { _ in Double.random(in: -limit...limit) } }
        
        // Initialize biases to zero
        self.hiddenBias = (0..<numHidden).map { _ in 0.0 }
        self.visibleBias = (0..<numVisible).map { _ in 0.0 }
    }
    
    // Improved Sigmoid function with numerical stability
    func sigmoid(_ x: Double) -> Double {
        let stableX = max(min(x, 15.0), -15.0)  // Clamping to avoid overflow in exp()
        return 1.0 / (1.0 + exp(-stableX))
    }
    
    // Sample hidden layer from visible layer
    func sampleHidden(from visible: [Double]) -> [Double] {
        var hidden = [Double](repeating: 0.0, count: numHidden)
        for j in 0..<numHidden {
            var activation = hiddenBias[j]
            for i in 0..<numVisible {
                activation += visible[i] * weights[i][j]
            }
            hidden[j] = sigmoid(activation)
        }
        return hidden
    }
    
    // Sample visible layer from hidden layer
    func sampleVisible(from hidden: [Double]) -> [Double] {
        var visible = [Double](repeating: 0.0, count: numVisible)
        for i in 0..<numVisible {
            var activation = visibleBias[i]
            for j in 0..<numHidden {
                activation += hidden[j] * weights[i][j]
            }
            visible[i] = sigmoid(activation)
        }
        return visible
    }
    
    // Persistent Contrastive Divergence (PCD) sampling method
    func persistentCD(k: Int, chain: inout [Double]) -> [Double] {
        for _ in 0..<k {
            let hidden = sampleHidden(from: chain)
            chain = sampleVisible(from: hidden)
        }
        return chain
    }
    
    // Training with Persistent Contrastive Divergence
    func train(data: [[Double]], epochs: Int, learningRate: Double, k: Int = 1) {
        var persistentChain = data.randomElement() ?? [Double](repeating: 0.0, count: numVisible)
        
        for _ in 0..<epochs {
            for sample in data {
                // Positive phase: sample hidden from data
                let positiveHidden = sampleHidden(from: sample)
                
                // Negative phase: PCD sampling
                let reconstructedVisible = persistentCD(k: k, chain: &persistentChain)
                let negativeHidden = sampleHidden(from: reconstructedVisible)
                
                // Update weights and biases
                for i in 0..<numVisible {
                    for j in 0..<numHidden {
                        // Update weights
                        weights[i][j] += learningRate * (sample[i] * positiveHidden[j] - reconstructedVisible[i] * negativeHidden[j])
                    }
                    // Update visible bias
                    visibleBias[i] += learningRate * (sample[i] - reconstructedVisible[i])
                }
                for j in 0..<numHidden {
                    // Update hidden bias
                    hiddenBias[j] += learningRate * (positiveHidden[j] - negativeHidden[j])
                }
            }
        }
    }
}
