const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const { query } = require('../utils/database');

// Configure multer for model uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/models/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.h5', '.json', '.pb', '.bin'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only model files are allowed.'));
    }
  }
});

/**
 * Create a new model
 */
router.post('/create', async (req, res) => {
  try {
    const {
      userId,
      name,
      modelType,
      architecture,
      description
    } = req.body;

    const result = await query(
      `INSERT INTO models 
       (user_id, name, model_type, architecture, status, created_at) 
       VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP) 
       RETURNING id`,
      [userId, name, modelType, JSON.stringify(architecture), 'created']
    );

    res.json({
      modelId: result.rows[0].id,
      message: 'Model created successfully'
    });
  } catch (error) {
    console.error('Error creating model:', error);
    res.status(500).json({ error: 'Failed to create model' });
  }
});

/**
 * List models for a user
 */
router.get('/list/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    
    const result = await query(
      `SELECT id, name, model_type, status, created_at, updated_at 
       FROM models 
       WHERE user_id = $1 
       ORDER BY created_at DESC`,
      [userId]
    );

    res.json({
      models: result.rows,
      count: result.rowCount
    });
  } catch (error) {
    console.error('Error listing models:', error);
    res.status(500).json({ error: 'Failed to list models' });
  }
});

/**
 * Get model details
 */
router.get('/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    
    const modelResult = await query(
      'SELECT * FROM models WHERE id = $1',
      [modelId]
    );

    if (modelResult.rowCount === 0) {
      return res.status(404).json({ error: 'Model not found' });
    }

    // Get training history
    const historyResult = await query(
      `SELECT epoch, loss, accuracy, val_loss, val_accuracy, 
              adversarial_accuracy, corruption_mce, timestamp
       FROM training_history 
       WHERE model_id = $1 
       ORDER BY epoch`,
      [modelId]
    );

    // Get latest test results
    const testResult = await query(
      `SELECT test_type, attack_method, epsilon, accuracy, 
              mce, mfr, boundary_distance, noise_insensitivity, created_at
       FROM robustness_tests 
       WHERE model_id = $1 
       ORDER BY created_at DESC 
       LIMIT 10`,
      [modelId]
    );

    res.json({
      model: modelResult.rows[0],
      trainingHistory: historyResult.rows,
      testResults: testResult.rows
    });
  } catch (error) {
    console.error('Error getting model details:', error);
    res.status(500).json({ error: 'Failed to get model details' });
  }
});

/**
 * Upload model weights
 */
router.post('/:modelId/upload', upload.single('modelFile'), async (req, res) => {
  try {
    const { modelId } = req.params;
    const file = req.file;

    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Update model with file path
    await query(
      `UPDATE models 
       SET status = $1, updated_at = CURRENT_TIMESTAMP 
       WHERE id = $2`,
      ['uploaded', modelId]
    );

    res.json({
      message: 'Model uploaded successfully',
      filename: file.filename,
      size: file.size
    });
  } catch (error) {
    console.error('Error uploading model:', error);
    res.status(500).json({ error: 'Failed to upload model' });
  }
});

/**
 * Update a model
 */
router.patch('/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    const { status, name, anpParams } = req.body;

    let updateQuery = 'UPDATE models SET updated_at = CURRENT_TIMESTAMP';
    let params = [];
    let paramCount = 0;

    if (status) {
      updateQuery += `, status = $${++paramCount}`;
      params.push(status);
    }

    if (name) {
      updateQuery += `, name = $${++paramCount}`;
      params.push(name);
    }

    if (anpParams) {
      updateQuery += `, anp_params = $${++paramCount}`;
      params.push(JSON.stringify(anpParams));
    }

    updateQuery += ` WHERE id = $${++paramCount} RETURNING id`;
    params.push(modelId);

    const result = await query(updateQuery, params);

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Model not found' });
    }

    res.json({ message: 'Model updated successfully', modelId });
  } catch (error) {
    console.error('Error updating model:', error);
    res.status(500).json({ error: 'Failed to update model' });
  }
});

/**
 * Delete a model
 */
router.delete('/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;

    // Delete related records first (due to foreign key constraints)
    await query('DELETE FROM model_weights WHERE model_id = $1', [modelId]);
    await query('DELETE FROM robustness_tests WHERE model_id = $1', [modelId]);
    await query('DELETE FROM training_history WHERE model_id = $1', [modelId]);
    
    // Delete the model
    const result = await query(
      'DELETE FROM models WHERE id = $1 RETURNING id',
      [modelId]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Model not found' });
    }

    res.json({ message: 'Model deleted successfully' });
  } catch (error) {
    console.error('Error deleting model:', error);
    res.status(500).json({ error: 'Failed to delete model' });
  }
});

/**
 * Update model training history
 */
router.post('/:modelId/history', async (req, res) => {
  try {
    const { modelId } = req.params;
    const {
      epoch,
      loss,
      accuracy,
      valLoss,
      valAccuracy,
      adversarialAccuracy,
      corruptionMce
    } = req.body;

    await query(
      `INSERT INTO training_history 
       (model_id, epoch, loss, accuracy, val_loss, val_accuracy, 
        adversarial_accuracy, corruption_mce, timestamp) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)`,
      [modelId, epoch, loss, accuracy, valLoss, valAccuracy, 
       adversarialAccuracy, corruptionMce]
    );

    res.json({ message: 'Training history updated' });
  } catch (error) {
    console.error('Error updating training history:', error);
    res.status(500).json({ error: 'Failed to update training history' });
  }
});

/**
 * Get model architectures
 */
router.get('/architectures/list', (req, res) => {
  const architectures = [
    {
      name: 'LeNet',
      type: 'CNN',
      description: 'Classic CNN for MNIST',
      inputShape: [28, 28, 1],
      classes: 10
    },
    {
      name: 'VGG16',
      type: 'CNN',
      description: 'Deep CNN for image classification',
      inputShape: [32, 32, 3],
      classes: 10
    },
    {
      name: 'ResNet18',
      type: 'CNN',
      description: 'Residual network with skip connections',
      inputShape: [32, 32, 3],
      classes: 10
    },
    {
      name: 'MobileNet',
      type: 'CNN',
      description: 'Lightweight CNN for mobile devices',
      inputShape: [224, 224, 3],
      classes: 1000
    },
    {
      name: 'Custom',
      type: 'Custom',
      description: 'Define your own architecture',
      inputShape: null,
      classes: null
    }
  ];

  res.json(architectures);
});

module.exports = router;