// Implementation of robustness metrics from the ANP paper

class RobustnessMetrics {
  constructor() {
    this.tf = null; // Will be set to window.tf or require('@tensorflow/tfjs')
  }

  setTensorFlow(tf) {
    this.tf = tf;
  }

  /**
   * Calculate Mean Corruption Error (mCE)
   * Lower is better
   */
  async calculateMCE(model, corruptedData, baselineErrors) {
    const errors = [];
    
    for (const corruption of corruptedData) {
      const predictions = model.predict(corruption.inputs);
      const accuracy = await this.tf.metrics.categoricalAccuracy(
        corruption.labels, 
        predictions
      ).mean().data();
      
      const error = 1 - accuracy;
      const ce = error / baselineErrors[corruption.type];
      errors.push(ce);
    }
    
    return errors.reduce((a, b) => a + b, 0) / errors.length;
  }

  /**
   * Calculate Relative mCE
   * Measures gap between mCE and clean data error
   */
  async calculateRelativeMCE(model, cleanData, corruptedData, baselineErrors) {
    const cleanPredictions = model.predict(cleanData.inputs);
    const cleanAccuracy = await this.tf.metrics.categoricalAccuracy(
      cleanData.labels,
      cleanPredictions
    ).mean().data();
    const cleanError = 1 - cleanAccuracy;
    
    const mce = await this.calculateMCE(model, corruptedData, baselineErrors);
    
    return mce - cleanError;
  }

  /**
   * Calculate mean Flip Rate (mFR) for noise sequences
   * Lower is better
   */
  async calculateMFR(model, noiseSequences, baselineFlipRates) {
    const flipRates = [];
    
    for (const sequence of noiseSequences) {
      let flips = 0;
      let prevPrediction = null;
      
      for (const frame of sequence.frames) {
        const prediction = model.predict(frame);
        const predClass = await this.tf.argMax(prediction, -1).data();
        
        if (prevPrediction !== null && predClass[0] !== prevPrediction) {
          flips++;
        }
        prevPrediction = predClass[0];
      }
      
      const flipProbability = flips / (sequence.frames.length - 1);
      const flipRate = flipProbability / baselineFlipRates[sequence.type];
      flipRates.push(flipRate);
    }
    
    return flipRates.reduce((a, b) => a + b, 0) / flipRates.length;
  }

  /**
   * Calculate Empirical Boundary Distance
   * Higher is better - measures minimum distance to decision boundary
   */
  async calculateEmpiricalBoundaryDistance(model, inputs, labels, numDirections = 100) {
    const distances = [];
    
    for (let i = 0; i < inputs.shape[0]; i++) {
      const input = inputs.slice([i, 0, 0, 0], [1, -1, -1, -1]);
      const label = labels.slice([i, 0], [1, -1]);
      
      const minDistance = await this.findMinimumBoundaryDistance(
        model, input, label, numDirections
      );
      distances.push(minDistance);
    }
    
    return distances.reduce((a, b) => a + b, 0) / distances.length;
  }

  /**
   * Find minimum distance to decision boundary for a single input
   */
  async findMinimumBoundaryDistance(model, input, label, numDirections) {
    let minDistance = Infinity;
    const originalPred = await this.tf.argMax(model.predict(input), -1).data();
    
    for (let d = 0; d < numDirections; d++) {
      // Generate random orthogonal direction
      const direction = this.tf.randomNormal(input.shape);
      const normalizedDir = this.tf.div(direction, this.tf.norm(direction));
      
      // Binary search for boundary
      let low = 0, high = 1.0;
      const tolerance = 0.001;
      
      while (high - low > tolerance) {
        const mid = (low + high) / 2;
        const perturbedInput = this.tf.add(
          input,
          this.tf.mul(normalizedDir, mid)
        );
        
        const pred = await this.tf.argMax(model.predict(perturbedInput), -1).data();
        
        if (pred[0] === originalPred[0]) {
          low = mid;
        } else {
          high = mid;
        }
      }
      
      minDistance = Math.min(minDistance, high);
    }
    
    return minDistance;
  }

  /**
   * Calculate ε-Empirical Noise Insensitivity
   * Lower is better - measures model's sensitivity to noise
   */
  async calculateNoiseInsensitivity(model, cleanInputs, labels, epsilon, noiseTypes) {
    const insensitivities = [];
    
    for (const noiseType of noiseTypes) {
      let totalDiff = 0;
      let count = 0;
      
      for (let i = 0; i < cleanInputs.shape[0]; i++) {
        const cleanInput = cleanInputs.slice([i, 0, 0, 0], [1, -1, -1, -1]);
        const label = labels.slice([i, 0], [1, -1]);
        
        // Generate noisy samples
        const noisyInputs = await this.generateNoisySamples(
          cleanInput, epsilon, noiseType, 10
        );
        
        // Calculate loss differences
        const cleanLoss = await this.calculateLoss(model, cleanInput, label);
        
        for (const noisyInput of noisyInputs) {
          const noisyLoss = await this.calculateLoss(model, noisyInput, label);
          const lossDiff = Math.abs(cleanLoss - noisyLoss);
          const inputDiff = await this.tf.norm(
            this.tf.sub(noisyInput, cleanInput), 'euclidean'
          ).data();
          
          totalDiff += lossDiff / inputDiff[0];
          count++;
        }
      }
      
      insensitivities.push(totalDiff / count);
    }
    
    return insensitivities.reduce((a, b) => a + b, 0) / insensitivities.length;
  }

  /**
   * Generate noisy samples for insensitivity calculation
   */
  async generateNoisySamples(input, epsilon, noiseType, numSamples) {
    const samples = [];
    
    for (let i = 0; i < numSamples; i++) {
      let noisyInput;
      
      switch (noiseType) {
        case 'gaussian':
          const gaussianNoise = this.tf.randomNormal(input.shape, 0, epsilon);
          noisyInput = this.tf.add(input, gaussianNoise);
          break;
          
        case 'uniform':
          const uniformNoise = this.tf.randomUniform(input.shape, -epsilon, epsilon);
          noisyInput = this.tf.add(input, uniformNoise);
          break;
          
        case 'adversarial':
          // Use FGSM for adversarial noise
          // This would need the attack methods class
          noisyInput = input; // Placeholder
          break;
          
        default:
          noisyInput = input;
      }
      
      // Ensure noisy input stays within epsilon ball
      const diff = this.tf.sub(noisyInput, input);
      const clippedDiff = this.tf.clipByValue(diff, -epsilon, epsilon);
      noisyInput = this.tf.add(input, clippedDiff);
      
      // Clip to valid range
      noisyInput = this.tf.clipByValue(noisyInput, 0, 1);
      samples.push(noisyInput);
    }
    
    return samples;
  }

  /**
   * Calculate loss for a single input
   */
  async calculateLoss(model, input, label) {
    return this.tf.tidy(() => {
      const prediction = model.predict(input);
      const loss = this.tf.losses.softmaxCrossEntropy(label, prediction);
      return loss.dataSync()[0];
    });
  }

  /**
   * Calculate comprehensive robustness report
   */
  async generateRobustnessReport(model, testData, options = {}) {
    const report = {
      timestamp: new Date().toISOString(),
      modelInfo: {
        layers: model.layers.length,
        parameters: this.countParameters(model)
      },
      metrics: {}
    };

    // Clean accuracy
    const cleanPredictions = model.predict(testData.clean.inputs);
    report.metrics.cleanAccuracy = await this.tf.metrics.categoricalAccuracy(
      testData.clean.labels,
      cleanPredictions
    ).mean().data();

    // Adversarial robustness
    if (testData.adversarial) {
      report.metrics.adversarial = {};
      for (const [attackName, attackData] of Object.entries(testData.adversarial)) {
        const advPredictions = model.predict(attackData.inputs);
        report.metrics.adversarial[attackName] = {
          accuracy: await this.tf.metrics.categoricalAccuracy(
            attackData.labels,
            advPredictions
          ).mean().data(),
          epsilon: attackData.epsilon
        };
      }
    }

    // Corruption robustness
    if (testData.corrupted && options.baselineErrors) {
      report.metrics.mce = await this.calculateMCE(
        model,
        testData.corrupted,
        options.baselineErrors
      );
      
      report.metrics.relativeMce = await this.calculateRelativeMCE(
        model,
        testData.clean,
        testData.corrupted,
        options.baselineErrors
      );
    }

    // Noise sequences
    if (testData.sequences && options.baselineFlipRates) {
      report.metrics.mfr = await this.calculateMFR(
        model,
        testData.sequences,
        options.baselineFlipRates
      );
    }

    // Structural robustness
    if (options.calculateStructural) {
      report.metrics.boundaryDistance = await this.calculateEmpiricalBoundaryDistance(
        model,
        testData.clean.inputs.slice([0, 0, 0, 0], [100, -1, -1, -1]), // First 100 samples
        testData.clean.labels.slice([0, 0], [100, -1]),
        options.numDirections || 50
      );

      report.metrics.noiseInsensitivity = await this.calculateNoiseInsensitivity(
        model,
        testData.clean.inputs.slice([0, 0, 0, 0], [50, -1, -1, -1]), // First 50 samples
        testData.clean.labels.slice([0, 0], [50, -1]),
        options.epsilon || 0.1,
        options.noiseTypes || ['gaussian', 'uniform']
      );
    }

    return report;
  }

  /**
   * Count total parameters in model
   */
  countParameters(model) {
    let total = 0;
    model.weights.forEach(w => {
      const shape = w.shape;
      const params = shape.reduce((a, b) => a * b, 1);
      total += params;
    });
    return total;
  }

  /**
   * Compare two models' robustness
   */
  compareModels(report1, report2) {
    const comparison = {
      model1Better: [],
      model2Better: [],
      similar: []
    };

    const compareMetric = (metric, lowerBetter = false) => {
      const val1 = this.getNestedValue(report1.metrics, metric);
      const val2 = this.getNestedValue(report2.metrics, metric);
      
      if (val1 === undefined || val2 === undefined) return;
      
      const diff = Math.abs(val1 - val2);
      const threshold = 0.01; // 1% difference threshold
      
      if (diff < threshold) {
        comparison.similar.push({ metric, val1, val2 });
      } else if ((val1 < val2 && lowerBetter) || (val1 > val2 && !lowerBetter)) {
        comparison.model1Better.push({ metric, val1, val2, improvement: diff });
      } else {
        comparison.model2Better.push({ metric, val1, val2, improvement: diff });
      }
    };

    // Compare metrics
    compareMetric('cleanAccuracy', false);
    compareMetric('mce', true);
    compareMetric('relativeMce', true);
    compareMetric('mfr', true);
    compareMetric('boundaryDistance', false);
    compareMetric('noiseInsensitivity', true);

    return comparison;
  }

  /**
   * Helper to get nested object value by path
   */
  getNestedValue(obj, path) {
    const keys = path.split('.');
    let value = obj;
    for (const key of keys) {
      if (value && value[key] !== undefined) {
        value = value[key];
      } else {
        return undefined;
      }
    }
    return value;
  }
}

module.exports = RobustnessMetrics;