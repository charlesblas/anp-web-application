// Implementation of various adversarial attack methods

class AdversarialAttacks {
  constructor() {
    this.tf = null; // Will be set to window.tf or require('@tensorflow/tfjs')
  }

  setTensorFlow(tf) {
    this.tf = tf;
  }

  /**
   * Fast Gradient Sign Method (FGSM)
   */
  async fgsm(model, inputs, labels, epsilon = 0.3) {
    return this.tf.tidy(() => {
      const inputVariable = this.tf.variable(inputs);
      
      const f = () => this.tf.tidy(() => {
        const predictions = model.predict(inputVariable);
        return this.tf.losses.softmaxCrossEntropy(labels, predictions);
      });
      
      const { value: loss, grads } = this.tf.variableGrads(f);
      const gradient = grads[inputVariable.name];
      
      // Create adversarial perturbation
      const perturbation = this.tf.mul(this.tf.sign(gradient), epsilon);
      const adversarialInputs = this.tf.add(inputs, perturbation);
      
      // Clip to valid range [0, 1]
      return this.tf.clipByValue(adversarialInputs, 0, 1);
    });
  }

  /**
   * Basic Iterative Method (BIM) / Iterative FGSM
   */
  async bim(model, inputs, labels, epsilon = 0.3, alpha = 0.01, iterations = 10) {
    let adversarialInputs = this.tf.clone(inputs);
    
    for (let i = 0; i < iterations; i++) {
      adversarialInputs = await this.tf.tidy(() => {
        const inputVariable = this.tf.variable(adversarialInputs);
        
        const f = () => this.tf.tidy(() => {
          const predictions = model.predict(inputVariable);
          return this.tf.losses.softmaxCrossEntropy(labels, predictions);
        });
        
        const { value: loss, grads } = this.tf.variableGrads(f);
        const gradient = grads[inputVariable.name];
        
        // Update with small step
        const update = this.tf.mul(this.tf.sign(gradient), alpha);
        const newInputs = this.tf.add(adversarialInputs, update);
        
        // Project back to epsilon ball
        const diff = this.tf.sub(newInputs, inputs);
        const clippedDiff = this.tf.clipByValue(diff, -epsilon, epsilon);
        const projectedInputs = this.tf.add(inputs, clippedDiff);
        
        // Clip to valid range
        return this.tf.clipByValue(projectedInputs, 0, 1);
      });
    }
    
    return adversarialInputs;
  }

  /**
   * Projected Gradient Descent (PGD)
   */
  async pgd(model, inputs, labels, epsilon = 0.3, alpha = 0.01, iterations = 40) {
    // Initialize with random noise
    const noise = this.tf.randomUniform(inputs.shape, -epsilon, epsilon);
    let adversarialInputs = this.tf.add(inputs, noise);
    adversarialInputs = this.tf.clipByValue(adversarialInputs, 0, 1);
    
    for (let i = 0; i < iterations; i++) {
      adversarialInputs = await this.tf.tidy(() => {
        const inputVariable = this.tf.variable(adversarialInputs);
        
        const f = () => this.tf.tidy(() => {
          const predictions = model.predict(inputVariable);
          return this.tf.losses.softmaxCrossEntropy(labels, predictions);
        });
        
        const { value: loss, grads } = this.tf.variableGrads(f);
        const gradient = grads[inputVariable.name];
        
        // Update with gradient ascent
        const update = this.tf.mul(this.tf.sign(gradient), alpha);
        const newInputs = this.tf.add(adversarialInputs, update);
        
        // Project back to epsilon ball
        const diff = this.tf.sub(newInputs, inputs);
        const clippedDiff = this.tf.clipByValue(diff, -epsilon, epsilon);
        const projectedInputs = this.tf.add(inputs, clippedDiff);
        
        // Clip to valid range
        return this.tf.clipByValue(projectedInputs, 0, 1);
      });
    }
    
    return adversarialInputs;
  }

  /**
   * Momentum Iterative FGSM (MI-FGSM)
   */
  async mifgsm(model, inputs, labels, epsilon = 0.3, alpha = 0.01, iterations = 10, decay = 1.0) {
    let adversarialInputs = this.tf.clone(inputs);
    let momentum = this.tf.zeros(inputs.shape);
    
    for (let i = 0; i < iterations; i++) {
      const result = await this.tf.tidy(() => {
        const inputVariable = this.tf.variable(adversarialInputs);
        
        const f = () => this.tf.tidy(() => {
          const predictions = model.predict(inputVariable);
          return this.tf.losses.softmaxCrossEntropy(labels, predictions);
        });
        
        const { value: loss, grads } = this.tf.variableGrads(f);
        const gradient = grads[inputVariable.name];
        
        // Update momentum
        const normGrad = this.tf.div(gradient, this.tf.norm(gradient, 1));
        momentum = this.tf.add(
          this.tf.mul(momentum, decay),
          normGrad
        );
        
        // Update with momentum
        const update = this.tf.mul(this.tf.sign(momentum), alpha);
        const newInputs = this.tf.add(adversarialInputs, update);
        
        // Project back to epsilon ball
        const diff = this.tf.sub(newInputs, inputs);
        const clippedDiff = this.tf.clipByValue(diff, -epsilon, epsilon);
        const projectedInputs = this.tf.add(inputs, clippedDiff);
        
        // Clip to valid range
        const clippedInputs = this.tf.clipByValue(projectedInputs, 0, 1);
        
        return {
          inputs: clippedInputs,
          momentum: momentum
        };
      });
      
      adversarialInputs = result.inputs;
      momentum.dispose();
      momentum = result.momentum;
    }
    
    momentum.dispose();
    return adversarialInputs;
  }

  /**
   * Add common corruptions (Gaussian noise, blur, etc.)
   */
  async addGaussianNoise(inputs, stddev = 0.1) {
    return this.tf.tidy(() => {
      const noise = this.tf.randomNormal(inputs.shape, 0, stddev);
      const corrupted = this.tf.add(inputs, noise);
      return this.tf.clipByValue(corrupted, 0, 1);
    });
  }

  /**
   * Add salt and pepper noise
   */
  async addSaltPepperNoise(inputs, amount = 0.05) {
    return this.tf.tidy(() => {
      const mask = this.tf.randomUniform(inputs.shape);
      const saltMask = this.tf.less(mask, amount / 2);
      const pepperMask = this.tf.greater(mask, 1 - amount / 2);
      
      let corrupted = this.tf.where(saltMask, this.tf.ones(inputs.shape), inputs);
      corrupted = this.tf.where(pepperMask, this.tf.zeros(inputs.shape), corrupted);
      
      return corrupted;
    });
  }

  /**
   * Apply Gaussian blur
   */
  async applyGaussianBlur(inputs, kernelSize = 5, sigma = 1.0) {
    return this.tf.tidy(() => {
      // Create Gaussian kernel
      const kernel = this.createGaussianKernel(kernelSize, sigma);
      
      // Apply convolution for blur
      // Assuming inputs shape is [batch, height, width, channels]
      const blurred = this.tf.conv2d(
        inputs,
        kernel,
        1, // strides
        'same' // padding
      );
      
      return blurred;
    });
  }

  /**
   * Create Gaussian kernel for blur
   */
  createGaussianKernel(size, sigma) {
    const kernel = [];
    const mean = size / 2;
    let sum = 0;
    
    for (let x = 0; x < size; x++) {
      kernel[x] = [];
      for (let y = 0; y < size; y++) {
        const exponent = -((x - mean) ** 2 + (y - mean) ** 2) / (2 * sigma ** 2);
        kernel[x][y] = Math.exp(exponent);
        sum += kernel[x][y];
      }
    }
    
    // Normalize kernel
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        kernel[x][y] /= sum;
      }
    }
    
    // Convert to tensor and reshape for conv2d
    // Shape: [height, width, in_channels, out_channels]
    return this.tf.tensor4d(kernel, [size, size, 1, 1]);
  }

  /**
   * Evaluate attack success rate
   */
  async evaluateAttack(model, originalInputs, adversarialInputs, labels) {
    return this.tf.tidy(() => {
      const originalPreds = model.predict(originalInputs);
      const adversarialPreds = model.predict(adversarialInputs);
      
      const originalClasses = this.tf.argMax(originalPreds, -1);
      const adversarialClasses = this.tf.argMax(adversarialPreds, -1);
      const trueClasses = this.tf.argMax(labels, -1);
      
      // Attack is successful if prediction changed and is incorrect
      const changed = this.tf.notEqual(originalClasses, adversarialClasses);
      const incorrect = this.tf.notEqual(adversarialClasses, trueClasses);
      const successful = this.tf.logicalAnd(changed, incorrect);
      
      const successRate = this.tf.mean(this.tf.cast(successful, 'float32'));
      
      return {
        successRate: successRate,
        originalAccuracy: this.tf.metrics.categoricalAccuracy(labels, originalPreds).mean(),
        adversarialAccuracy: this.tf.metrics.categoricalAccuracy(labels, adversarialPreds).mean()
      };
    });
  }
}

module.exports = AdversarialAttacks;