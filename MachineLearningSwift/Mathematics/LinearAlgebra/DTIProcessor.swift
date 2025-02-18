//
//  DTIProcessor.swift
//  MachineLearningSwift
//
//  Created by Suad Demiri on 14.02.25.
//

import Foundation

/*

// MARK: - Data Structure for DWI JSON Input
struct DWIData: Codable {
    let dwi: [[[Float]]]
    let bvals: [Float]
    let bvecs: [[Float]]
}

// MARK: - DTI Processing Class
class DTIProcessor {
    var dwiData: DWIData
    var tensorMap: [[[Float]]]
    
    init(jsonPath: String) {
        guard let data = DTIProcessor.loadDWIData(from: jsonPath) else {
            fatalError("Failed to load DWI data from JSON.")
        }
        self.dwiData = data
        self.tensorMap = []
    }
    
    // Load JSON file
    static func loadDWIData(from jsonPath: String) -> DWIData? {
        guard let jsonData = try? Data(contentsOf: URL(fileURLWithPath: jsonPath)) else {
            print("Error loading JSON file.")
            return nil
        }
        let decoder = JSONDecoder()
        return try? decoder.decode(DWIData.self, from: jsonData)
    }
    
    // Normalize DWI data
    func normalizeDWI() -> [[[Float]]] {
        let maxValue = dwiData.dwi.flatMap { $0 }.flatMap { $0 }.max() ?? 1.0
        return dwiData.dwi.map { $0.map { $0.map { $0 / maxValue } } }
    }
    
    // Compute Diffusion Tensor using Least Squares
    func computeDiffusionTensor() {
        let rows = dwiData.dwi.count
        let cols = dwiData.dwi[0].count
        tensorMap = [[[Float]]](repeating: [[Float]](repeating: [0, 0, 0, 0, 0, 0], count: cols), count: rows)
        
        for i in 0..<rows {
            for j in 0..<cols {
                var X: [[Float]] = []
                var Y: [Float] = []
                
                for k in 1..<dwiData.bvals.count { // Skip b0
                    let b = dwiData.bvals[k]
                    let g = dwiData.bvecs[k]
                    let signal = dwiData.dwi[i][j][k]
                    let S0 = dwiData.dwi[i][j][0]
                    
                    if S0 > 0 && signal > 0 {
                        let logRatio = -log(signal / S0)
                        let row = [b * g[0] * g[0], b * g[1] * g[1], b * g[2] * g[2],
                                   2 * b * g[0] * g[1], 2 * b * g[0] * g[2], 2 * b * g[1] * g[2]]
                        
                        X.append(row)
                        Y.append(logRatio)
                    }
                }
                
                if !X.isEmpty {
                    let tensor = leastSquaresSolve(X, Y)
                    tensorMap[i][j] = tensor
                }
            }
        }
    }
    
    // Least squares solver for DTI tensor estimation
    private func leastSquaresSolve(_ X: [[Float]], _ Y: [Float]) -> [Float] {
        let X_tensor = Tensor(X)
        let Y_tensor = Tensor([Y.map { [$0] }])
        
        do {
            let Xt = try X_tensor.transpose()
            let XtX = try Xt.matmul(X_tensor)
            let XtY = try Xt.matmul(Y_tensor)
            let D = try XtX.inverse().matmul(XtY)
            
            return D.value.flatMap { $0 }
        } catch {
            print("Error solving least squares:", error)
            return [0, 0, 0, 0, 0, 0]
        }
    }
    
    // Compute Fractional Anisotropy (FA)
    func computeFA() -> [[Float]] {
        return tensorMap.map { row in
            row.map { computeFA(for: $0) }
        }
    }
    
    private func computeFA(for tensor: [Float]) -> Float {
        let λ1 = tensor[0]
        let λ2 = tensor[1]
        let λ3 = tensor[2]
        
        let meanDiffusivity = (λ1 + λ2 + λ3) / 3.0
        let numerator = sqrt(3.0 * ((λ1 - meanDiffusivity) * (λ1 - meanDiffusivity) +
                                    (λ2 - meanDiffusivity) * (λ2 - meanDiffusivity) +
                                    (λ3 - meanDiffusivity) * (λ3 - meanDiffusivity)))
        
        let denominator = sqrt(λ1 * λ1 + λ2 * λ2 + λ3 * λ3)
        return denominator > 0 ? numerator / denominator : 0
    }
    
    // Compute Mean Diffusivity (MD)
    func computeMD() -> [[Float]] {
        return tensorMap.map { row in
            row.map { computeMD(for: $0) }
        }
    }
    
    private func computeMD(for tensor: [Float]) -> Float {
        return (tensor[0] + tensor[1] + tensor[2]) / 3.0
    }
    
    // Save results as JSON
    func saveResults(to path: String) {
        let results: [String: [[Float]]] = [
            "fa_map": computeFA(),
            "md_map": computeMD()
        ]
        
        guard let jsonData = try? JSONEncoder().encode(results) else {
            print("Error encoding JSON.")
            return
        }
        
        do {
            try jsonData.write(to: URL(fileURLWithPath: path))
            print("Results saved successfully at \(path)")
        } catch {
            print("Error saving file: \(error)")
        }
    }
}


/*
 Add Tractography using Neural Networks 🚀
 ✅ Visualize DTI Metrics as Heatmaps
 ✅ Enhance Preprocessing with Noise Reduction

 Would you like fiber tractography or a visualization tool next? 🎨
 */

*/
