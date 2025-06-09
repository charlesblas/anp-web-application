const { query } = require('./database');

async function seedSampleData() {
  try {
    // Check if data already exists
    const existingData = await query('SELECT COUNT(*) FROM models');
    if (existingData.rows[0].count > 0) {
      console.log('Sample data already exists, skipping seeding...');
      return;
    }

    console.log('Seeding sample data...');

    // Create a sample user
    const userResult = await query(
      `INSERT INTO users (username, email, password_hash) 
       VALUES ($1, $2, $3) 
       ON CONFLICT (email) DO UPDATE SET username = EXCLUDED.username
       RETURNING id`,
      ['demo_user', 'demo@anp.com', 'hashed_password_placeholder']
    );
    
    const userId = userResult.rows[0].id;
    console.log('Created user with ID:', userId);

    // Create sample models
    const models = [
      {
        name: 'ANP-VGG16-CIFAR10',
        type: 'CNN',
        architecture: {
          layers: [
            { type: 'conv2d', filters: 64, kernelSize: 3, activation: 'relu' },
            { type: 'conv2d', filters: 64, kernelSize: 3, activation: 'relu' },
            { type: 'maxPooling2d', poolSize: 2 },
            { type: 'conv2d', filters: 128, kernelSize: 3, activation: 'relu' },
            { type: 'conv2d', filters: 128, kernelSize: 3, activation: 'relu' },
            { type: 'maxPooling2d', poolSize: 2 },
            { type: 'flatten' },
            { type: 'dense', units: 512, activation: 'relu' },
            { type: 'dense', units: 10, activation: 'softmax' }
          ]
        },
        anpParams: { epsilon: 0.3, eta: 0.1, k: 3, topLayers: 4, norm: 'inf' },
        status: 'trained'
      },
      {
        name: 'Standard-VGG16-CIFAR10',
        type: 'CNN',
        architecture: {
          layers: [
            { type: 'conv2d', filters: 64, kernelSize: 3, activation: 'relu' },
            { type: 'conv2d', filters: 64, kernelSize: 3, activation: 'relu' },
            { type: 'maxPooling2d', poolSize: 2 },
            { type: 'conv2d', filters: 128, kernelSize: 3, activation: 'relu' },
            { type: 'conv2d', filters: 128, kernelSize: 3, activation: 'relu' },
            { type: 'maxPooling2d', poolSize: 2 },
            { type: 'flatten' },
            { type: 'dense', units: 512, activation: 'relu' },
            { type: 'dense', units: 10, activation: 'softmax' }
          ]
        },
        anpParams: null,
        status: 'trained'
      },
      {
        name: 'ANP-ResNet18-CIFAR10',
        type: 'CNN',
        architecture: {
          layers: [
            { type: 'conv2d', filters: 64, kernelSize: 7, strides: 2, activation: 'relu' },
            { type: 'maxPooling2d', poolSize: 3, strides: 2 },
            { type: 'residualBlock', filters: 64, blocks: 2 },
            { type: 'residualBlock', filters: 128, blocks: 2, strides: 2 },
            { type: 'residualBlock', filters: 256, blocks: 2, strides: 2 },
            { type: 'residualBlock', filters: 512, blocks: 2, strides: 2 },
            { type: 'globalAveragePooling2d' },
            { type: 'dense', units: 10, activation: 'softmax' }
          ]
        },
        anpParams: { epsilon: 0.25, eta: 0.15, k: 5, topLayers: 4, norm: 'inf' },
        status: 'trained'
      }
    ];

    const modelIds = [];
    for (const model of models) {
      const result = await query(
        `INSERT INTO models (user_id, name, model_type, architecture, anp_params, status) 
         VALUES ($1, $2, $3, $4, $5, $6) 
         RETURNING id`,
        [userId, model.name, model.type, JSON.stringify(model.architecture), 
         model.anpParams ? JSON.stringify(model.anpParams) : null, model.status]
      );
      modelIds.push(result.rows[0].id);
      console.log(`Created model: ${model.name} with ID: ${result.rows[0].id}`);
    }

    // Create sample training history
    for (let i = 0; i < modelIds.length; i++) {
      const modelId = modelIds[i];
      const isANP = i !== 1; // Second model is standard training
      
      for (let epoch = 1; epoch <= 50; epoch++) {
        const loss = 2.5 * Math.exp(-epoch * 0.05) + 0.1 + Math.random() * 0.1;
        const accuracy = Math.min(0.95, 0.5 + (epoch * 0.008) + Math.random() * 0.02);
        const valLoss = loss + 0.1 + Math.random() * 0.05;
        const valAccuracy = accuracy - 0.02 + Math.random() * 0.01;
        const adversarialAccuracy = isANP ? accuracy * 0.9 : accuracy * 0.3; // ANP maintains better adversarial accuracy
        const corruptionMce = isANP ? 75 + Math.random() * 5 : 95 + Math.random() * 10;

        await query(
          `INSERT INTO training_history 
           (model_id, epoch, loss, accuracy, val_loss, val_accuracy, 
            adversarial_accuracy, corruption_mce) 
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
          [modelId, epoch, loss, accuracy, valLoss, valAccuracy, 
           adversarialAccuracy, corruptionMce]
        );
      }
    }

    // Create sample robustness test results
    const attacks = ['fgsm', 'pgd', 'bim', 'mifgsm'];
    const epsilons = [0.1, 0.2, 0.3, 0.4];

    for (let i = 0; i < modelIds.length; i++) {
      const modelId = modelIds[i];
      const isANP = i !== 1; // Second model is standard training

      // Clean accuracy test
      await query(
        `INSERT INTO robustness_tests 
         (model_id, test_type, accuracy, created_at) 
         VALUES ($1, $2, $3, CURRENT_TIMESTAMP)`,
        [modelId, 'clean', isANP ? 0.932 + Math.random() * 0.02 : 0.915 + Math.random() * 0.02]
      );

      // Adversarial tests
      for (const attack of attacks) {
        for (const epsilon of epsilons) {
          const baseAccuracy = isANP ? 0.85 : 0.45; // ANP models are more robust
          const accuracy = Math.max(0.1, baseAccuracy - (epsilon * 0.5) + Math.random() * 0.1);
          
          await query(
            `INSERT INTO robustness_tests 
             (model_id, test_type, attack_method, epsilon, accuracy, created_at) 
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)`,
            [modelId, 'adversarial', attack, epsilon, accuracy]
          );
        }
      }

      // Corruption tests
      const corruptionTypes = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'brightness'];
      for (const corruption of corruptionTypes) {
        const mce = isANP ? 75 + Math.random() * 10 : 95 + Math.random() * 15;
        const mfr = isANP ? 85 + Math.random() * 10 : 105 + Math.random() * 15;
        
        await query(
          `INSERT INTO robustness_tests 
           (model_id, test_type, mce, mfr, test_params, created_at) 
           VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)`,
          [modelId, 'corruption', mce, mfr, JSON.stringify({ corruption_type: corruption })]
        );
      }

      // Structural robustness tests
      const boundaryDistance = isANP ? 0.45 + Math.random() * 0.1 : 0.25 + Math.random() * 0.05;
      const noiseInsensitivity = isANP ? 0.15 + Math.random() * 0.05 : 0.35 + Math.random() * 0.1;
      
      await query(
        `INSERT INTO robustness_tests 
         (model_id, test_type, boundary_distance, noise_insensitivity, created_at) 
         VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)`,
        [modelId, 'structural', boundaryDistance, noiseInsensitivity]
      );
    }

    console.log('Sample data seeding completed successfully!');
    
  } catch (error) {
    console.error('Error seeding data:', error);
    throw error;
  }
}

// Run seeding if called directly
if (require.main === module) {
  seedSampleData()
    .then(() => {
      console.log('Seeding complete');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Seeding failed:', error);
      process.exit(1);
    });
}

module.exports = { seedSampleData };