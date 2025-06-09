const express = require('express');
const router = express.Router();
const { query } = require('../utils/database');

/**
 * Get robustness metrics for a model
 */
router.get('/model/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    
    // Get all test results
    const testResults = await query(
      `SELECT test_type, attack_method, epsilon, accuracy, 
              mce, mfr, boundary_distance, noise_insensitivity, 
              test_params, results, created_at
       FROM robustness_tests 
       WHERE model_id = $1 
       ORDER BY created_at DESC`,
      [modelId]
    );

    // Calculate aggregate metrics
    const metrics = calculateAggregateMetrics(testResults.rows);

    res.json({
      modelId,
      metrics,
      rawResults: testResults.rows
    });
  } catch (error) {
    console.error('Error getting metrics:', error);
    res.status(500).json({ error: 'Failed to get metrics' });
  }
});

/**
 * Compare metrics between models
 */
router.post('/compare', async (req, res) => {
  try {
    const { modelIds } = req.body;
    
    if (!modelIds || modelIds.length < 2) {
      return res.status(400).json({ error: 'At least 2 model IDs required' });
    }

    const comparisons = [];
    
    for (const modelId of modelIds) {
      const testResults = await query(
        `SELECT test_type, attack_method, epsilon, accuracy, 
                mce, mfr, boundary_distance, noise_insensitivity
         FROM robustness_tests 
         WHERE model_id = $1`,
        [modelId]
      );

      const modelInfo = await query(
        'SELECT name, model_type FROM models WHERE id = $1',
        [modelId]
      );

      comparisons.push({
        modelId,
        modelName: modelInfo.rows[0]?.name,
        metrics: calculateAggregateMetrics(testResults.rows)
      });
    }

    res.json({
      comparison: comparisons,
      summary: generateComparisonSummary(comparisons)
    });
  } catch (error) {
    console.error('Error comparing metrics:', error);
    res.status(500).json({ error: 'Failed to compare metrics' });
  }
});

/**
 * Get metrics over time
 */
router.get('/timeline/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    const { days = 30 } = req.query;
    
    const results = await query(
      `SELECT DATE(created_at) as date, 
              test_type,
              AVG(accuracy) as avg_accuracy,
              AVG(mce) as avg_mce,
              AVG(mfr) as avg_mfr,
              COUNT(*) as test_count
       FROM robustness_tests 
       WHERE model_id = $1 
         AND created_at >= CURRENT_DATE - INTERVAL '${days} days'
       GROUP BY DATE(created_at), test_type
       ORDER BY date`,
      [modelId]
    );

    res.json({
      modelId,
      timeline: results.rows,
      period: `${days} days`
    });
  } catch (error) {
    console.error('Error getting timeline:', error);
    res.status(500).json({ error: 'Failed to get timeline' });
  }
});

/**
 * Get attack success rates
 */
router.get('/attacks/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    
    const results = await query(
      `SELECT attack_method, epsilon,
              COUNT(*) as total_tests,
              AVG(accuracy) as avg_accuracy,
              MIN(accuracy) as min_accuracy,
              MAX(accuracy) as max_accuracy
       FROM robustness_tests 
       WHERE model_id = $1 
         AND attack_method IS NOT NULL
       GROUP BY attack_method, epsilon
       ORDER BY attack_method, epsilon`,
      [modelId]
    );

    const attackStats = results.rows.map(row => ({
      ...row,
      successRate: 1 - row.avg_accuracy // Attack success is inverse of accuracy
    }));

    res.json({
      modelId,
      attackStatistics: attackStats
    });
  } catch (error) {
    console.error('Error getting attack stats:', error);
    res.status(500).json({ error: 'Failed to get attack statistics' });
  }
});

/**
 * Get corruption robustness summary
 */
router.get('/corruption/:modelId', async (req, res) => {
  try {
    const { modelId } = req.params;
    
    const results = await query(
      `SELECT test_params->>'corruption_type' as corruption_type,
              AVG(mce) as avg_mce,
              AVG(mfr) as avg_mfr,
              COUNT(*) as test_count
       FROM robustness_tests 
       WHERE model_id = $1 
         AND test_type = 'corruption'
         AND test_params->>'corruption_type' IS NOT NULL
       GROUP BY test_params->>'corruption_type'
       ORDER BY avg_mce`,
      [modelId]
    );

    res.json({
      modelId,
      corruptionStats: results.rows
    });
  } catch (error) {
    console.error('Error getting corruption stats:', error);
    res.status(500).json({ error: 'Failed to get corruption statistics' });
  }
});

/**
 * Get leaderboard
 */
router.get('/leaderboard', async (req, res) => {
  try {
    const { metric = 'accuracy', testType = 'all' } = req.query;
    
    let whereClause = '';
    if (testType !== 'all') {
      whereClause = `WHERE rt.test_type = '${testType}'`;
    }

    const results = await query(
      `SELECT m.id, m.name, m.model_type,
              AVG(rt.accuracy) as avg_accuracy,
              AVG(rt.mce) as avg_mce,
              AVG(rt.mfr) as avg_mfr,
              AVG(rt.boundary_distance) as avg_boundary_distance,
              COUNT(rt.id) as test_count
       FROM models m
       JOIN robustness_tests rt ON m.id = rt.model_id
       ${whereClause}
       GROUP BY m.id, m.name, m.model_type
       HAVING COUNT(rt.id) >= 5
       ORDER BY ${metric === 'mce' || metric === 'mfr' ? 'AVG(rt.' + metric + ') ASC' : 'AVG(rt.' + metric + ') DESC'}
       LIMIT 20`
    );

    res.json({
      leaderboard: results.rows,
      metric,
      testType
    });
  } catch (error) {
    console.error('Error getting leaderboard:', error);
    res.status(500).json({ error: 'Failed to get leaderboard' });
  }
});

/**
 * Helper functions
 */
function calculateAggregateMetrics(testResults) {
  if (!testResults || testResults.length === 0) {
    return {
      cleanAccuracy: null,
      adversarialRobustness: {},
      corruptionRobustness: {},
      structuralRobustness: {}
    };
  }

  const metrics = {
    cleanAccuracy: null,
    adversarialRobustness: {},
    corruptionRobustness: {},
    structuralRobustness: {}
  };

  // Clean accuracy
  const cleanTests = testResults.filter(r => r.test_type === 'clean');
  if (cleanTests.length > 0) {
    metrics.cleanAccuracy = cleanTests.reduce((sum, r) => sum + r.accuracy, 0) / cleanTests.length;
  }

  // Adversarial robustness by attack type
  const adversarialTests = testResults.filter(r => r.test_type === 'adversarial');
  const attackGroups = {};
  adversarialTests.forEach(test => {
    const key = `${test.attack_method}_${test.epsilon}`;
    if (!attackGroups[key]) {
      attackGroups[key] = [];
    }
    attackGroups[key].push(test.accuracy);
  });
  
  Object.entries(attackGroups).forEach(([key, accuracies]) => {
    metrics.adversarialRobustness[key] = {
      avgAccuracy: accuracies.reduce((a, b) => a + b, 0) / accuracies.length,
      minAccuracy: Math.min(...accuracies),
      maxAccuracy: Math.max(...accuracies)
    };
  });

  // Corruption robustness
  const corruptionTests = testResults.filter(r => r.test_type === 'corruption');
  if (corruptionTests.length > 0) {
    metrics.corruptionRobustness = {
      avgMCE: corruptionTests.reduce((sum, r) => sum + (r.mce || 0), 0) / corruptionTests.length,
      avgMFR: corruptionTests.reduce((sum, r) => sum + (r.mfr || 0), 0) / corruptionTests.length
    };
  }

  // Structural robustness
  const structuralTests = testResults.filter(r => r.boundary_distance !== null || r.noise_insensitivity !== null);
  if (structuralTests.length > 0) {
    metrics.structuralRobustness = {
      avgBoundaryDistance: structuralTests
        .filter(r => r.boundary_distance !== null)
        .reduce((sum, r, _, arr) => sum + r.boundary_distance / arr.length, 0),
      avgNoiseInsensitivity: structuralTests
        .filter(r => r.noise_insensitivity !== null)
        .reduce((sum, r, _, arr) => sum + r.noise_insensitivity / arr.length, 0)
    };
  }

  return metrics;
}

function generateComparisonSummary(comparisons) {
  if (comparisons.length < 2) return {};

  const summary = {
    bestCleanAccuracy: null,
    bestAdversarialRobustness: null,
    bestCorruptionRobustness: null,
    rankings: []
  };

  // Find best clean accuracy
  const cleanAccuracies = comparisons
    .filter(c => c.metrics.cleanAccuracy !== null)
    .sort((a, b) => b.metrics.cleanAccuracy - a.metrics.cleanAccuracy);
  
  if (cleanAccuracies.length > 0) {
    summary.bestCleanAccuracy = {
      modelId: cleanAccuracies[0].modelId,
      modelName: cleanAccuracies[0].modelName,
      accuracy: cleanAccuracies[0].metrics.cleanAccuracy
    };
  }

  // Calculate overall rankings
  comparisons.forEach(comp => {
    let score = 0;
    let factors = 0;

    if (comp.metrics.cleanAccuracy !== null) {
      score += comp.metrics.cleanAccuracy;
      factors++;
    }

    const advRobustness = Object.values(comp.metrics.adversarialRobustness);
    if (advRobustness.length > 0) {
      const avgAdv = advRobustness.reduce((sum, r) => sum + r.avgAccuracy, 0) / advRobustness.length;
      score += avgAdv;
      factors++;
    }

    if (comp.metrics.corruptionRobustness.avgMCE !== undefined) {
      score += (1 - comp.metrics.corruptionRobustness.avgMCE / 100); // Normalize MCE
      factors++;
    }

    summary.rankings.push({
      modelId: comp.modelId,
      modelName: comp.modelName,
      overallScore: factors > 0 ? score / factors : 0
    });
  });

  summary.rankings.sort((a, b) => b.overallScore - a.overallScore);

  return summary;
}

module.exports = router;