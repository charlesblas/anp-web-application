# ANP (Adversarial Noise Propagation) Web Application

A comprehensive web application implementing the Adversarial Noise Propagation defense technique for training robust deep neural networks. This application provides an interactive interface for training models with ANP, evaluating robustness against various attacks, and comparing model performance.

## 🚀 Features

### Core ANP Implementation
- **Layer-wise Noise Injection**: Implements the complete ANP algorithm with noise propagation through hidden layers
- **Progressive Training**: k-step gradient descent for computing and propagating adversarial noise
- **Shallow Layer Focus**: Emphasizes robustness training on shallow layers as described in the research
- **Configurable Parameters**: Full control over ε (epsilon), η (eta), k-steps, and layer selection

### Web Interface
- **Interactive Training**: Real-time training progress with live metrics visualization
- **Model Evaluation**: Comprehensive robustness testing against multiple attack methods
- **Attack Testing**: Upload images and test adversarial attacks with visual results
- **Model Comparison**: Side-by-side comparison of ANP vs standard models
- **Metrics Dashboard**: Leaderboards and detailed robustness analytics

### Supported Features
- **Attack Methods**: FGSM, BIM, PGD, MI-FGSM, C&W
- **Robustness Metrics**: mCE, mFR, Empirical Boundary Distance, ε-Empirical Noise Insensitivity
- **Datasets**: MNIST, CIFAR-10, ImageNet (configurable)
- **Architectures**: LeNet, VGG-16, ResNet-18, MobileNet

## 📋 Prerequisites

- Node.js (v14 or higher)
- PostgreSQL (v12 or higher)
- npm or yarn

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/anp-web-application.git
   cd anp-web-application
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Database Setup**
   
   Create a PostgreSQL database and update the connection settings in `.env` file:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=anp_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   PORT=3000
   ```

4. **Start the application**
   ```bash
   npm start
   ```

5. **Access the application**
   
   Open your browser and navigate to `http://localhost:3000`

## 🏗️ Project Structure

```
anp-web-application/
├── src/
│   ├── anp/                    # ANP algorithm implementation
│   │   ├── anpAlgorithm.js     # Core ANP training logic
│   │   ├── attackMethods.js    # Adversarial attack implementations
│   │   └── robustnessMetrics.js # Robustness evaluation metrics
│   ├── routes/                 # API endpoints
│   │   ├── anp.js             # ANP training and attack routes
│   │   ├── models.js          # Model management routes
│   │   └── metrics.js         # Metrics and comparison routes
│   ├── utils/                  # Utilities
│   │   ├── database.js        # Database connection
│   │   ├── initDB.js          # Database initialization
│   │   └── seedData.js        # Sample data seeding
│   └── index.js               # Main server file
├── public/                    # Frontend assets
│   ├── css/
│   │   └── style.css         # Application styles
│   └── js/
│       └── app.js            # Frontend JavaScript
├── views/
│   └── index.ejs             # Main HTML template
└── package.json
```

## 🔬 Research Background

This implementation is based on the research paper on Adversarial Noise Propagation (ANP), which introduces a novel defense training technique that:

- Injects adversarial noise into hidden layers during training (not just input)
- Uses progressive backward-forward propagation to compute layer-wise perturbations
- Focuses on shallow layers which are more critical for robustness
- Achieves state-of-the-art defense against both adversarial attacks and natural corruption

### Key Algorithm Components

1. **Layer-wise Perturbation**: ANP computes perturbations for each layer using gradients
2. **Noise Propagation**: Previous layer noise contributes to current layer computation
3. **k-step Training**: Iterative refinement of adversarial examples during training
4. **Shallow Layer Emphasis**: Top-k layers receive focused robustness training

## 🎯 Usage Guide

### Training a Model

1. Navigate to the **Training** tab
2. Configure model parameters:
   - Model name and architecture
   - Dataset selection
   - ANP parameters (ε, η, k-steps, top layers)
3. Click "Start Training" to begin
4. Monitor real-time progress with live charts

### Evaluating Robustness

1. Go to the **Evaluation** tab
2. Select a trained model
3. Choose evaluation types:
   - Clean accuracy
   - Adversarial robustness (FGSM, PGD, etc.)
   - Corruption robustness
   - Structural robustness
4. View comprehensive results and metrics

### Testing Attacks

1. Visit the **Attacks** tab
2. Upload an image or load a sample
3. Configure attack parameters:
   - Attack method (FGSM, PGD, C&W, etc.)
   - Epsilon (perturbation budget)
   - Target class (for targeted attacks)
4. Generate adversarial examples and view results

### Comparing Models

1. Access the **Metrics** tab
2. Select two models to compare
3. View side-by-side performance charts
4. Analyze detailed metric comparisons
5. Check leaderboard rankings

## 🔧 API Reference

### Training Endpoints
- `POST /api/models/create` - Create a new model
- `POST /api/anp/train` - Start ANP training
- `GET /api/anp/train/status/:sessionId` - Get training status

### Evaluation Endpoints
- `POST /api/anp/evaluate` - Run robustness evaluation
- `GET /api/models/:modelId` - Get model details and results

### Attack Endpoints
- `POST /api/anp/attack` - Generate adversarial examples
- `POST /api/anp/predict` - Get model predictions

### Metrics Endpoints
- `POST /api/metrics/compare` - Compare multiple models
- `GET /api/metrics/leaderboard` - Get performance leaderboard

## 🧪 Technical Implementation

### ANP Algorithm Core
The application implements the complete ANP algorithm including:
- Gradient-based layer perturbation computation
- Progressive noise propagation through network layers
- k-step iterative training with adversarial examples
- Configurable norm constraints (L∞, L2)

### Database Schema
- **Models**: Store model configurations and training parameters
- **Training History**: Track epoch-by-epoch training progress
- **Robustness Tests**: Store evaluation results for various attacks
- **Model Weights**: Manage model parameter storage

### Frontend Architecture
- Responsive single-page application with tabbed interface
- Real-time chart updates using Chart.js
- Interactive image upload and attack visualization
- Comprehensive metrics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this implementation in your research, please cite the original ANP paper:

```bibtex
@article{anp_paper,
  title={Training Robust Deep Neural Networks via Adversarial Noise Propagation},
  author={[Original Authors]},
  journal={[Journal Name]},
  year={[Year]},
  note={Implementation available at https://github.com/yourusername/anp-web-application}
}
```

## 🙏 Acknowledgments

- Original ANP research paper authors
- TensorFlow.js team for browser-based ML capabilities
- Chart.js for visualization components
- PostgreSQL community for robust database support

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the API reference above

---

**Built with ❤️ for advancing adversarial robustness research**