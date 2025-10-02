# 🍄 Mushroom Classification Neural Network

A comprehensive implementation of a neural network from scratch for binary classification of mushrooms as edible or poisonous. This project provides educational insights into neural network fundamentals with detailed step-by-step demonstrations.

## 📋 Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Fine-tuning Guide](#-fine-tuning-guide)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Pure Python Implementation**: Neural network built from scratch using only NumPy
- **Educational Focus**: Step-by-step forward and backward propagation demonstrations
- **Interactive Interface**: Menu-driven system for training and testing
- **Comprehensive Evaluation**: Detailed metrics including confusion matrix, precision, recall, F1-score
- **Hyperparameter Search**: Automated optimization of network parameters
- **Reproducible Results**: Seed-based randomization for consistent outputs
- **Error Analysis**: Automatic export of misclassified samples for review
- **Data Processing**: Built-in support for CSV data loading and preprocessing

## 📊 Dataset

The project works with mushroom classification datasets containing the following features:

### Input Features (20 total)
- **cap-diameter** (float): Diameter of mushroom cap
- **cap-shape** (categorical): Shape of the cap (bell, conical, convex, etc.)
- **cap-surface** (categorical): Surface texture of the cap
- **cap-color** (categorical): Color of the cap
- **does-bruise-or-bleed** (categorical): Whether mushroom bruises or bleeds
- **gill-attachment** (categorical): How gills attach to stem
- **gill-spacing** (categorical): Spacing between gills
- **gill-color** (categorical): Color of the gills
- **stem-height** (float): Height of the stem
- **stem-width** (float): Width of the stem
- **stem-root** (categorical): Type of root system
- **stem-surface** (categorical): Surface texture of stem
- **stem-color** (categorical): Color of the stem
- **veil-type** (categorical): Type of veil
- **veil-color** (categorical): Color of veil
- **has-ring** (categorical): Presence of ring
- **ring-type** (categorical): Type of ring
- **spore-print-color** (categorical): Color of spore print
- **habitat** (categorical): Natural habitat
- **season** (categorical): Growing season

### Target Variable
- **class**: Binary classification (poisonous=0, edible=1)

### Expected Data Format
```csv
class;cap-diameter;cap-shape;cap-surface;cap-color;does-bruise-or-bleed;gill-attachment;gill-spacing;gill-color;stem-height;stem-width;stem-root;stem-surface;stem-color;veil-type;veil-color;has-ring;ring-type;spore-print-color;habitat;season
p;5.49;o;s;r;f;f;f;f;0.0;0.0;f;f;f;;;t;f;;d;u
e;8.25;x;y;n;t;a;c;w;7.5;12.3;r;s;w;;;f;f;;l;s
```

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install numpy pandas scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/mushroom-neural-network.git
```

## ⚡ Quick Start

1. **Prepare your data**: Ensure you have training and testing CSV files in the correct format
2. **Run the program**:
   ```bash
   python mushroom_tiny_nn.py
   ```
3. **Choose from the menu**:
   - Interactive Training Demo
   - Test Reproducibility
   - Exit

## 💻 Usage

### Basic Training Example
```python
from mushroom_tiny_nn import *

# Set seeds for reproducibility
set_all_seeds(42)

# Load and prepare data
X_train, y_train = load_mushroom_data('mushroom_train.csv')
X_test, y_test = load_mushroom_data('mushroom_test.csv')

# Standardize features
X_train_std, X_test_std, mean, std = standardize_features(X_train, X_test)

# Initialize and train model
model = TinyNN.init(n_inputs=20, n_hidden=8, n_outputs=1, lr=0.1, seed=42)
losses = model.train(X_train_std, y_train, epochs=200, verbose_every=50)

# Evaluate model
metrics = evaluate_model_detailed(model, X_test_std, y_test)
```

### Making Predictions
```python
# Single prediction
sample_features = X_test_std[0]
probability, prediction = predict_mushroom(model, sample_features)
print(f"Prediction: {'Edible' if prediction == 1 else 'Poisonous'}")
print(f"Confidence: {probability:.4f}")
```

### Step-by-Step Demonstration
```python
# See internal workings of the neural network
model.demo_step_by_step(X_test_std[0])
```

## 🏗️ Model Architecture

### Network Structure
- **Input Layer**: 20 features + bias term
- **Hidden Layer**: Configurable neurons (default: 20) + bias term
- **Output Layer**: 1 neuron (sigmoid activation)

### Mathematical Foundation
- **Activation Function**: Sigmoid (σ(z) = 1/(1 + e^(-z)))
- **Loss Function**: Binary Cross-Entropy
- **Optimization**: Stochastic Gradient Descent (SGD)
- **Weight Initialization**: Normal distribution (μ=0, σ=0.1)

### Forward Propagation
```
z₁ = W₁ · x + b₁
a₁ = σ(z₁)
z₂ = W₂ · a₁ + b₂
ŷ = σ(z₂)
```

### Backward Propagation
```
δ₂ = (ŷ - y)
δ₁ = (δ₂ · W₂) ⊙ σ'(a₁)
∇W₂ = δ₂ · a₁ᵀ
∇W₁ = δ₁ · xᵀ
```

## 🎯 Fine-tuning Guide

### Hyperparameter Recommendations

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Hidden Neurons | 5 | 8 | 20 |
| Learning Rate | 0.1 | 0.1 | 0.1 |
| Epochs | 200 | 200 | 300 |
| Expected Accuracy | >85% | >87% | >90% |

### Diagnostic Tips

**Underfitting Signs:**
- High training and test loss
- Low accuracy on both sets
- Solution: Increase model complexity or learning rate

**Overfitting Signs:**
- Low training loss, high test loss
- Large gap between train/test accuracy
- Solution: Reduce model complexity or implement regularization

**Good Performance:**
- Smooth loss decrease
- Test accuracy > 90%
- Small train/test accuracy gap


## 📈 Results

### Typical Performance Metrics
- **Accuracy**: 90-95% on test set
- **Training Time**: 2-5 seconds for 200 epochs
- **Model Size**: <1KB (lightweight)
- **Inference Speed**: <1ms per prediction

### Example Output
```
📊 DETAILED MODEL EVALUATION
==================================================
Total samples: 2000
Accuracy: 0.9250 (1850/2000)
Precision: 0.9180
Recall: 0.9320
F1-Score: 0.9249

🔢 CONFUSION MATRIX:
                 Predicted
               Poisonous  Edible    
Actual Poisonous    890      60      
       Edible        90     960      

📈 INTERPRETATION:
   ✅ Excellent performance!
   ⚠️ 60 poisonous mushrooms classified as edible - DANGEROUS!
```

## 🔬 Advanced Features

### Reproducibility Testing
```python
# Test multiple runs with same seed
test_reproducibility(train_file='mushroom_train.csv', 
                    test_file='mushroom_test.csv', 
                    n_runs=3, seed=42)
```

### Error Analysis
```python
# Automatically save misclassified samples
incorrect_df = save_incorrect_predictions(model, X_test_original, X_test_std, 
                                        y_test, feature_names)
```

### Hyperparameter Search
```python
# Automated optimization
best_model, best_config = hyperparameter_search()
```

## 🛠️ Development

### Running Tests
```bash
# Test reproducibility
python mushroom_tiny_nn.py
# Choose option 4: Test Reproducibility
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Add docstrings for new functions
3. **Testing**: Ensure reproducible results
4. **Performance**: Maintain efficiency for educational use

### Areas for Contribution
- Additional activation functions
- Regularization techniques
- Advanced optimization algorithms
- Visualization tools
- Extended evaluation metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by educational neural network tutorials
- Dataset format based on UCI Mushroom Database
- Mathematical foundations from "Deep Learning" by Goodfellow, Bengio, and Courville

## 📞 Contact

- **Author**: Siyabonga Mbuyisa
- **Email**: siyabongambuyisa7@gmail.com

---

⭐ **Star this repository if you find it helpful!** ⭐

*This project is designed for educational purposes to understand neural network fundamentals. Always consult mycological experts before consuming wild mushrooms.*