const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const { initializeDatabase } = require('./utils/initDB');
const { seedSampleData } = require('./utils/seedData');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, '../public')));

// View engine setup
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../views'));

// Routes (to be added)
app.get('/', (req, res) => {
  res.render('index');
});

// API Routes
const anpRoutes = require('./routes/anp');
const modelRoutes = require('./routes/models');
const metricsRoutes = require('./routes/metrics');

app.use('/api/anp', anpRoutes);
app.use('/api/models', modelRoutes);
app.use('/api/metrics', metricsRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Something went wrong!',
    message: err.message
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Initialize database and start server
async function startServer() {
  try {
    // Initialize database tables
    await initializeDatabase();
    console.log('Database initialized successfully');
    
    // Seed sample data for demo
    await seedSampleData();
    console.log('Sample data seeded successfully');
    
    // Start server
    app.listen(PORT, () => {
      console.log(`ANP Web Application running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

module.exports = app;