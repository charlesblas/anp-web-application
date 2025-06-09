const { query } = require('./database');

const initializeDatabase = async () => {
  try {
    // Create users table
    await query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create models table
    await query(`
      CREATE TABLE IF NOT EXISTS models (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        name VARCHAR(255) NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        architecture JSONB,
        training_params JSONB,
        anp_params JSONB,
        status VARCHAR(50) DEFAULT 'created',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create training_history table
    await query(`
      CREATE TABLE IF NOT EXISTS training_history (
        id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES models(id),
        epoch INTEGER NOT NULL,
        loss FLOAT,
        accuracy FLOAT,
        val_loss FLOAT,
        val_accuracy FLOAT,
        adversarial_accuracy FLOAT,
        corruption_mce FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create robustness_tests table
    await query(`
      CREATE TABLE IF NOT EXISTS robustness_tests (
        id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES models(id),
        test_type VARCHAR(50) NOT NULL,
        attack_method VARCHAR(100),
        epsilon FLOAT,
        accuracy FLOAT,
        mce FLOAT,
        mfr FLOAT,
        boundary_distance FLOAT,
        noise_insensitivity FLOAT,
        test_params JSONB,
        results JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create model_weights table for storing model parameters
    await query(`
      CREATE TABLE IF NOT EXISTS model_weights (
        id SERIAL PRIMARY KEY,
        model_id INTEGER REFERENCES models(id),
        layer_name VARCHAR(255) NOT NULL,
        weights BYTEA,
        shape JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    console.log('Database tables initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    throw error;
  }
};

// Run initialization if called directly
if (require.main === module) {
  initializeDatabase()
    .then(() => {
      console.log('Database initialization complete');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Database initialization failed:', error);
      process.exit(1);
    });
}

module.exports = { initializeDatabase };