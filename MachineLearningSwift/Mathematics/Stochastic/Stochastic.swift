import Foundation

// Using a struct is appropriate here because:
// 1. The calculator is stateless - all methods are static
// 2. Value semantics make sense for mathematical operations
// 3. No inheritance is needed
// 4. It's more performant than classes for this use case

struct StatisticalCalculator {
    
    // MARK: - Error Handling
    enum StatisticalError: Error {
        case insufficientData
        case invalidParameters
        case divisionByZero
        case computationError
    }
    
    // MARK: - Data Validation
    private static func validateData(_ data: [Double]) throws {
        guard !data.isEmpty else {
            throw StatisticalError.insufficientData
        }
    }
    
    // MARK: - Basic Statistical Functions
    static func mean(_ data: [Double]) throws -> Double {
        try validateData(data)
        return data.reduce(0, +) / Double(data.count)
    }
    
    static func weightedMean(_ data: [Double], weights: [Double]) throws -> Double {
        guard data.count == weights.count else {
            throw StatisticalError.invalidParameters
        }
        try validateData(data)
        let weightSum = weights.reduce(0, +)
        guard weightSum != 0 else {
            throw StatisticalError.divisionByZero
        }
        return zip(data, weights).map(*).reduce(0, +) / weightSum
    }
    
    static func median(_ data: [Double]) throws -> Double {
        try validateData(data)
        let sortedData = data.sorted()
        let n = sortedData.count
        return n % 2 == 0 ?
        (sortedData[n/2 - 1] + sortedData[n/2]) / 2 :
        sortedData[n/2]
    }
    
    static func mode(_ data: [Double]) throws -> [Double] {
        try validateData(data)
        let frequencyDict = Dictionary(grouping: data) { $0 }.mapValues { $0.count }
        guard let maxFrequency = frequencyDict.values.max() else {
            throw StatisticalError.computationError
        }
        return frequencyDict.filter { $0.value == maxFrequency }.map { $0.key }.sorted()
    }
    
    static func variance(_ data: [Double], isSample: Bool = true) throws -> Double {
        try validateData(data)
        guard data.count > 1 else {
            throw StatisticalError.insufficientData
        }
        let m = try mean(data)
        let count = Double(data.count - (isSample ? 1 : 0))
        return data.reduce(0) { $0 + pow($1 - m, 2) } / count
    }
    
    static func standardDeviation(_ data: [Double], isSample: Bool = true) throws -> Double {
        return sqrt(try variance(data, isSample: isSample))
    }
    
    static func skewness(_ data: [Double]) throws -> Double {
        try validateData(data)
        let m = try mean(data)
        let std = try standardDeviation(data)
        let n = Double(data.count)
        let sum = data.reduce(0) { $0 + pow(($1 - m) / std, 3) }
        return (n / ((n - 1) * (n - 2))) * sum
    }
    
    static func kurtosis(_ data: [Double]) throws -> Double {
        try validateData(data)
        let m = try mean(data)
        let std = try standardDeviation(data)
        let n = Double(data.count)
        let sum = data.reduce(0) { $0 + pow(($1 - m) / std, 4) }
        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum - (3 * pow(n - 1, 2)) / ((n - 2) * (n - 3))
    }
    
    // MARK: - Correlation and Regression
    static func correlation(_ x: [Double], _ y: [Double]) throws -> Double {
        guard x.count == y.count else {
            throw StatisticalError.invalidParameters
        }
        try validateData(x)
        try validateData(y)
        
        let meanX = try mean(x)
        let meanY = try mean(y)
        let stdX = try standardDeviation(x)
        let stdY = try standardDeviation(y)
        
        let numerator = zip(x, y).reduce(0) { $0 + ($1.0 - meanX) * ($1.1 - meanY) }
        let denominator = Double(x.count - 1) * stdX * stdY
        return numerator / denominator
    }
    
    static func linearRegression(_ x: [Double], _ y: [Double]) throws -> (slope: Double, intercept: Double) {
        guard x.count == y.count else {
            throw StatisticalError.invalidParameters
        }
        try validateData(x)
        try validateData(y)
        
        let meanX = try mean(x)
        let meanY = try mean(y)
        
        let numerator = zip(x, y).reduce(0) { $0 + ($1.0 - meanX) * ($1.1 - meanY) }
        let denominator = x.reduce(0) { $0 + pow($1 - meanX, 2) }
        
        guard denominator != 0 else {
            throw StatisticalError.divisionByZero
        }
        
        let slope = numerator / denominator
        let intercept = meanY - slope * meanX
        
        return (slope, intercept)
    }
    
    // MARK: - Hypothesis Testing
    static func zTest(sampleMean: Double, populationMean: Double,
                      sigma: Double, n: Int) throws -> (zScore: Double, pValue: Double) {
        guard n > 0, sigma > 0 else {
            throw StatisticalError.invalidParameters
        }
        
        let zScore = (sampleMean - populationMean) / (sigma / sqrt(Double(n)))
        let pValue = 2 * (1 - normalCDF(x: abs(zScore)))
        
        return (zScore, pValue)
    }
    
    static func tTest(sample1: [Double], sample2: [Double]) throws -> (tScore: Double, pValue: Double) {
        try validateData(sample1)
        try validateData(sample2)
        
        let n1 = Double(sample1.count)
        let n2 = Double(sample2.count)
        let mean1 = try mean(sample1)
        let mean2 = try mean(sample2)
        let var1 = try variance(sample1)
        let var2 = try variance(sample2)
        
        let pooledVariance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        let standardError = sqrt(pooledVariance * (1/n1 + 1/n2))
        
        let tScore = (mean1 - mean2) / standardError
        // Note: p-value calculation would require t-distribution implementation
        // This is a simplified version using normal distribution
        let pValue = 2 * (1 - normalCDF(x: abs(tScore)))
        
        return (tScore, pValue)
    }
    
    // MARK: - Probability Distributions
    static func normalPDF(x: Double, mu: Double = 0, sigma: Double = 1) throws -> Double {
        guard sigma > 0 else {
            throw StatisticalError.invalidParameters
        }
        let exponent = -pow(x - mu, 2) / (2 * pow(sigma, 2))
        return (1 / (sigma * sqrt(2 * .pi))) * exp(exponent)
    }
    
    static func normalCDF(x: Double, mu: Double = 0, sigma: Double = 1) -> Double {
        return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    }
    
    static func binomialPMF(k: Int, n: Int, p: Double) throws -> Double {
        guard n > 0, p >= 0, p <= 1, k >= 0, k <= n else {
            throw StatisticalError.invalidParameters
        }
        
        let combinations = try combinations(n: n, k: k)
        return Double(combinations) * pow(p, Double(k)) * pow(1 - p, Double(n - k))
    }
    
    // MARK: - Descriptive Statistics
    static func descriptiveStatistics(_ data: [Double]) throws -> [String: Double] {
        try validateData(data)
        
        return [
            "count": Double(data.count),
            "mean": try mean(data),
            "median": try median(data),
            "variance": try variance(data),
            "standardDeviation": try standardDeviation(data),
            "skewness": try skewness(data),
            "kurtosis": try kurtosis(data),
            "minimum": data.min() ?? 0,
            "maximum": data.max() ?? 0,
            "range": (data.max() ?? 0) - (data.min() ?? 0)
        ]
    }
    
    // MARK: - Helper Functions
    static func combinations(n: Int, k: Int) throws -> Int {
        guard n >= 0, k >= 0, k <= n else {
            throw StatisticalError.invalidParameters
        }
        return permutations(n: n, k: k) / factorial(k)
    }
    
    private static func permutations(n: Int, k: Int) -> Int {
        return (n - k + 1...n).reduce(1, *)
    }
    
    private static func factorial(_ n: Int) -> Int {
        return (1...max(n, 1)).reduce(1, *)
    }
}

// MARK: - Extensions for Convenient Usage
extension Array where Element == Double {
    func mean() throws -> Double {
        return try StatisticalCalculator.mean(self)
    }
    
    func median() throws -> Double {
        return try StatisticalCalculator.median(self)
    }
    
    func standardDeviation(isSample: Bool = true) throws -> Double {
        return try StatisticalCalculator.standardDeviation(self, isSample: isSample)
    }
    
    func descriptiveStatistics() throws -> [String: Double] {
        return try StatisticalCalculator.descriptiveStatistics(self)
    }
}
