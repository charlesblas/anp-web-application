// Adversarial Noise Propagation (ANP) Algorithm Implementation
// Based on the paper: "Training Robust Deep Neural Networks via Adversarial Noise Propagation"

class ANPTrainer {
  constructor(config = {}) {
    this.config = {
      epsilon: config.epsilon || 0.3,  // Noise magnitude constraint
      eta: config.eta || 0.1,          // Previous noise contribution coefficient
      k: config.k || 3,                // Number of gradient descent steps
      norm: config.norm || 'inf',      // Norm type ('inf' or '2')
      topLayers: config.topLayers || 4, // Number of top layers to perturb
      ...config
    };
    
    this.noiseRegisters = new Map(); // Store noise for each layer
  }

  /**
   * Initialize noise registers for all layers
   */
  initializeNoiseRegisters(model) {
    const layers = this.getTrainableLayers(model);
    layers.forEach(layer => {
      this.noiseRegisters.set(layer.name, {
        noise: null,
        gradients: []
      });
    });
  }

  /**
   * Get trainable layers from model (only top-k layers as per paper)
   */
  getTrainableLayers(model) {
    const allLayers = model.layers.filter(layer => 
      layer.trainable && layer.weights.length > 0
    );
    
    // According to the paper, shallow layers are more critical
    // So we take the first topLayers layers
    return allLayers.slice(0, this.config.topLayers);
  }

  /**
   * Compute adversarial gradients for each layer
   */
  async computeAdversarialGradients(model, inputs, labels, lossFunction) {
    const tf = window.tf || require('@tensorflow/tfjs');
    
    return tf.tidy(() => {
      const layerGradients = new Map();
      
      // Forward pass with current noise
      const predictions = this.forwardPassWithNoise(model, inputs);
      const loss = lossFunction(labels, predictions);
      
      // Compute gradients with respect to each layer's activations
      const grads = tf.grads((inputs) => {
        const preds = this.forwardPassWithNoise(model, inputs);
        return lossFunction(labels, preds);
      });
      
      const gradFunction = grads(inputs);
      
      // Store gradients for each layer
      const trainableLayers = this.getTrainableLayers(model);
      trainableLayers.forEach(layer => {
        const layerOutput = layer.apply(inputs);
        const gradient = tf.grad(x => lossFunction(labels, model.apply(x)))(layerOutput);
        layerGradients.set(layer.name, gradient);
      });
      
      return layerGradients;
    });
  }

  /**
   * Update noise registers using gradient descent
   */
  updateNoiseRegisters(layerGradients, step) {
    const tf = window.tf || require('@tensorflow/tfjs');
    
    layerGradients.forEach((gradient, layerName) => {
      const register = this.noiseRegisters.get(layerName);
      
      tf.tidy(() => {
        // Normalize gradient based on norm type
        let normalizedGrad;
        if (this.config.norm === 'inf') {
          const maxVal = tf.max(tf.abs(gradient));
          normalizedGrad = tf.div(gradient, maxVal);
        } else if (this.config.norm === '2') {
          const l2Norm = tf.norm(gradient);
          normalizedGrad = tf.div(gradient, l2Norm);
        }
        
        // Update noise: r_{m,t+1} = (1-η)r_{m,t} + (ε/k)g_{m,t}/||g_{m,t}||_p
        if (register.noise === null) {
          register.noise = tf.mul(normalizedGrad, this.config.epsilon / this.config.k);
        } else {
          const prevContribution = tf.mul(register.noise, 1 - this.config.eta);
          const gradContribution = tf.mul(normalizedGrad, this.config.epsilon / this.config.k);
          register.noise = tf.add(prevContribution, gradContribution);
        }
      });
    });
  }

  /**
   * Forward pass with adversarial noise injection
   */
  forwardPassWithNoise(model, inputs) {
    const tf = window.tf || require('@tensorflow/tfjs');
    
    return tf.tidy(() => {
      let activations = inputs;
      
      for (const layer of model.layers) {
        // Apply layer transformation
        activations = layer.apply(activations);
        
        // Add noise if this layer has a noise register
        if (this.noiseRegisters.has(layer.name)) {
          const register = this.noiseRegisters.get(layer.name);
          if (register.noise !== null) {
            activations = tf.add(activations, register.noise);
          }
        }
      }
      
      return activations;
    });
  }

  /**
   * ANP training step
   */
  async trainStep(model, inputs, labels, optimizer, lossFunction) {
    const tf = window.tf || require('@tensorflow/tfjs');
    
    // Initialize noise registers if not done
    if (this.noiseRegisters.size === 0) {
      this.initializeNoiseRegisters(model);
    }
    
    let totalLoss = 0;
    
    // K-step gradient descent for noise computation
    for (let step = 0; step < this.config.k; step++) {
      // Compute adversarial gradients
      const layerGradients = await this.computeAdversarialGradients(
        model, inputs, labels, lossFunction
      );
      
      // Update noise registers
      this.updateNoiseRegisters(layerGradients, step);
      
      // Perform standard training with noise injection
      const loss = await tf.tidy(() => {
        const f = () => tf.tidy(() => {
          const preds = this.forwardPassWithNoise(model, inputs);
          return lossFunction(labels, preds);
        });
        
        const { value, grads } = tf.variableGrads(f);
        optimizer.applyGradients(grads);
        return value;
      });
      
      totalLoss += await loss.data();
    }
    
    return totalLoss / this.config.k;
  }

  /**
   * Evaluate model robustness
   */
  async evaluateRobustness(model, testData, attackMethods) {
    const results = {
      clean: await this.evaluateClean(model, testData),
      adversarial: {},
      corruption: {}
    };
    
    // Evaluate against different attack methods
    for (const attack of attackMethods) {
      results.adversarial[attack.name] = await this.evaluateAdversarial(
        model, testData, attack
      );
    }
    
    return results;
  }

  /**
   * Evaluate on clean data
   */
  async evaluateClean(model, testData) {
    const tf = window.tf || require('@tensorflow/tfjs');
    const predictions = model.predict(testData.inputs);
    const accuracy = tf.metrics.categoricalAccuracy(testData.labels, predictions);
    return {
      accuracy: await accuracy.mean().data()
    };
  }

  /**
   * Evaluate against adversarial attacks
   */
  async evaluateAdversarial(model, testData, attack) {
    // Implementation depends on specific attack method
    // This is a placeholder for attack-specific evaluation
    return {
      accuracy: 0,
      epsilon: attack.epsilon
    };
  }

  /**
   * Clean up resources
   */
  dispose() {
    const tf = window.tf || require('@tensorflow/tfjs');
    this.noiseRegisters.forEach(register => {
      if (register.noise !== null) {
        register.noise.dispose();
      }
    });
    this.noiseRegisters.clear();
  }
}

module.exports = ANPTrainer;