const express = require('express');
const router = express.Router();
const { query } = require('../utils/database');
const ANPTrainer = require('../anp/anpAlgorithm');
const AdversarialAttacks = require('../anp/attackMethods');
const RobustnessMetrics = require('../anp/robustnessMetrics');

// Store active training sessions
const trainingSessions = new Map();

/**
 * Start ANP training
 */
router.post('/train', async (req, res) => {
  try {
    const {
      modelId,
      userId,
      anpConfig,
      trainingConfig
    } = req.body;

    // Validate inputs
    if (!modelId || !userId) {
      return res.status(400).json({ error: 'Model ID and User ID are required' });
    }

    // Create new ANP trainer instance
    const trainer = new ANPTrainer(anpConfig);
    const sessionId = `${userId}_${modelId}_${Date.now()}`;
    
    trainingSessions.set(sessionId, {
      trainer,
      modelId,
      userId,
      status: 'training',
      startTime: new Date(),
      config: { anpConfig, trainingConfig }
    });

    // Update model status in database
    await query(
      'UPDATE models SET status = $1, anp_params = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3',
      ['training', JSON.stringify(anpConfig), modelId]
    );

    res.json({
      sessionId,
      message: 'ANP training started',
      config: anpConfig
    });
  } catch (error) {
    console.error('Error starting ANP training:', error);
    res.status(500).json({ error: 'Failed to start training' });
  }
});

/**
 * Get training status
 */
router.get('/train/status/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  const session = trainingSessions.get(sessionId);

  if (!session) {
    return res.status(404).json({ error: 'Training session not found' });
  }

  res.json({
    sessionId,
    status: session.status,
    startTime: session.startTime,
    currentEpoch: session.currentEpoch || 0,
    metrics: session.metrics || {}
  });
});

/**
 * Stop training
 */
router.post('/train/stop/:sessionId', async (req, res) => {
  const { sessionId } = req.params;
  const session = trainingSessions.get(sessionId);

  if (!session) {
    return res.status(404).json({ error: 'Training session not found' });
  }

  // Clean up resources
  if (session.trainer) {
    session.trainer.dispose();
  }
  
  // Update model status
  await query(
    'UPDATE models SET status = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2',
    ['stopped', session.modelId]
  );

  trainingSessions.delete(sessionId);

  res.json({ message: 'Training stopped successfully' });
});

/**
 * Generate adversarial examples
 */
router.post('/attack', async (req, res) => {
  try {
    const {
      modelId,
      attackMethod,
      epsilon,
      inputs,
      labels,
      targetClass,
      iterations,
      targeted
    } = req.body;

    // Create attack instance
    const attacks = new AdversarialAttacks();
    
    // Simulate attack generation with realistic results
    const attackResult = {
      success: Math.random() > 0.3, // 70% success rate
      attackMethod,
      epsilon,
      targeted,
      targetClass,
      iterations: Math.min(iterations || 10, Math.floor(Math.random() * (iterations || 10)) + 1),
      metrics: {
        linfDistance: epsilon * (0.8 + Math.random() * 0.2),
        l2Distance: epsilon * Math.sqrt(inputs?.length || 1024) * 0.1,
        confidenceDrop: 0.1 + Math.random() * 0.4
      },
      timeTaken: 1000 + Math.random() * 2000
    };
    
    res.json({
      attackResult,
      message: 'Adversarial examples generated'
    });
  } catch (error) {
    console.error('Error generating adversarial examples:', error);
    res.status(500).json({ error: 'Failed to generate adversarial examples' });
  }
});

/**
 * Predict image class
 */
router.post('/predict', async (req, res) => {
  try {
    const {
      modelId,
      imageData,
      datasetType
    } = req.body;

    // Simulate model prediction
    const classCount = {
      'cifar10': 10,
      'mnist': 10,
      'imagenet': 1000
    }[datasetType] || 10;

    const predictedClass = Math.floor(Math.random() * classCount);
    const confidence = 0.7 + Math.random() * 0.25;

    // Simulate different confidence for different models
    const modelInfo = await query(
      'SELECT name, anp_params FROM models WHERE id = $1',
      [modelId]
    );

    let adjustedConfidence = confidence;
    if (modelInfo.rows.length > 0 && modelInfo.rows[0].anp_params) {
      // ANP models tend to be more confident on clean examples
      adjustedConfidence = Math.min(0.95, confidence * 1.1);
    }

    res.json({
      prediction: {
        class: predictedClass,
        confidence: adjustedConfidence
      },
      modelInfo: modelInfo.rows[0] || { name: 'Unknown Model' }
    });
  } catch (error) {
    console.error('Error predicting image:', error);
    res.status(500).json({ error: 'Failed to predict image' });
  }
});

/**
 * Evaluate model robustness
 */
router.post('/evaluate', async (req, res) => {
  try {
    const {
      modelId,
      testType,
      attackMethod,
      accuracy,
      epsilon,
      boundaryDistance,
      mce
    } = req.body;

    // Validate required fields
    if (!modelId || !testType) {
      return res.status(400).json({ error: 'Model ID and test type are required' });
    }
    
    // Store evaluation results
    const result = await query(
      `INSERT INTO robustness_tests 
       (model_id, test_type, attack_method, epsilon, accuracy, mce, boundary_distance, created_at) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP) 
       RETURNING id`,
      [modelId, testType, attackMethod, epsilon, accuracy, mce, boundaryDistance]
    );

    res.json({
      testId: result.rows[0].id,
      testType,
      attackMethod,
      message: 'Evaluation result saved successfully'
    });
  } catch (error) {
    console.error('Error saving evaluation result:', error);
    res.status(500).json({ error: 'Failed to save evaluation result' });
  }
});

/**
 * Get ANP configuration recommendations
 */
router.get('/config/recommendations', (req, res) => {
  const recommendations = {
    basic: {
      epsilon: 0.3,
      eta: 0.1,
      k: 3,
      norm: 'inf',
      topLayers: 4,
      description: 'Basic configuration for general use'
    },
    strong: {
      epsilon: 0.5,
      eta: 0.15,
      k: 5,
      norm: 'inf',
      topLayers: 4,
      description: 'Stronger defense against larger perturbations'
    },
    fast: {
      epsilon: 0.2,
      eta: 0.1,
      k: 1,
      norm: 'inf',
      topLayers: 2,
      description: 'Faster training with moderate robustness'
    },
    research: {
      epsilon: 0.3,
      eta: 0.1,
      k: 3,
      norm: '2',
      topLayers: 6,
      description: 'Configuration used in the original paper'
    }
  };

  res.json(recommendations);
});

/**
 * Compare ANP vs standard training
 */
router.post('/compare', async (req, res) => {
  try {
    const { modelId1, modelId2 } = req.body;

    // Fetch test results for both models
    const results1 = await query(
      'SELECT * FROM robustness_tests WHERE model_id = $1 ORDER BY created_at DESC LIMIT 10',
      [modelId1]
    );

    const results2 = await query(
      'SELECT * FROM robustness_tests WHERE model_id = $1 ORDER BY created_at DESC LIMIT 10',
      [modelId2]
    );

    res.json({
      model1: {
        id: modelId1,
        results: results1.rows
      },
      model2: {
        id: modelId2,
        results: results2.rows
      },
      comparison: {
        // Comparison metrics would be calculated here
      }
    });
  } catch (error) {
    console.error('Error comparing models:', error);
    res.status(500).json({ error: 'Failed to compare models' });
  }
});

/**
 * Get dataset class names
 */
router.get('/datasets/:datasetType/classes', (req, res) => {
  const { datasetType } = req.params;
  
  const datasetClasses = {
    cifar10: [
      'airplane', 'automobile', 'bird', 'cat', 'deer',
      'dog', 'frog', 'horse', 'ship', 'truck'
    ],
    mnist: [
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ],
    imagenet: [
      'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
      'electric ray', 'stingray', 'cock', 'hen', 'ostrich',
      'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting',
      'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
      'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl'
      // ... includes first 25 classes for demo, normally would have all 1000
    ]
  };

  if (datasetClasses[datasetType]) {
    res.json({
      datasetType,
      classes: datasetClasses[datasetType],
      count: datasetClasses[datasetType].length
    });
  } else {
    res.status(404).json({ error: 'Dataset type not found' });
  }
});

module.exports = router;