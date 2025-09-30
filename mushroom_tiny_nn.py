
"""
Reproducible Neural Network for Mushroom Classification

🔢 ENSURING REPRODUCIBILITY:
1. Set global seeds using set_all_seeds(seed) 
2. Use the same seed for model initialization
3. Use seeded training with model.train(..., seed=seed)
4. Data loading order is preserved when using same file
5. All random operations (shuffling, weight init) are seeded

📋 REPRODUCIBILITY CHECKLIST:
✅ Random seed set for: Python random, NumPy, model weights
✅ Training data shuffling uses seeded RNG  
✅ Weight initialization uses seeded RNG
✅ All randomness is controlled and reproducible

🚀 USAGE:
- Run option 1 for training with reproducible results
- Run option 2 to test that identical seeds produce identical results
- Use the same seed value to get exactly the same model performance
"""

import numpy as np
import pandas as pd
import random

def set_all_seeds(seed=42):
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Also set the default random number generator
    np.random.default_rng(seed)
    
    print(f"🔢 All random seeds set to: {seed}")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid_from_activation(a):
    return a * (1.0 - a)

def bce_loss(y_true, y_hat, eps=1e-8):
    y_hat = np.clip(y_hat, eps, 1-eps)
    return -np.mean(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))

class TinyNN:
    def __init__(self, n_inputs, n_hidden, n_outputs, W1, W2, lr=0.1):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.W1 = W1
        self.W2 = W2
        self.lr = lr

    @staticmethod
    def init(n_inputs, n_hidden, n_outputs, seed=42, wscale=0.1, lr=0.1):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0, wscale, size=(n_hidden, n_inputs+1))
        W2 = rng.normal(0, wscale, size=(n_outputs, n_hidden+1))
        return TinyNN(n_inputs, n_hidden, n_outputs, W1, W2, lr)

    def forward(self, x_no_bias):
        x = np.concatenate([x_no_bias, [1.0]])
        n1 = self.W1 @ x
        a1_nb = sigmoid(n1)
        a1 = np.concatenate([a1_nb, [1.0]])
        n2 = self.W2 @ a1
        a2 = sigmoid(n2)
        return {"x_b": x, "n1": n1, "a1_nb": a1_nb, "a1_b": a1, "n2": n2, "a2": a2}

    def backward(self, cache, y_true):
        x_b = cache["x_b"]
        a1_nb = cache["a1_nb"]
        a1_b = cache["a1_b"]
        a2 = cache["a2"]
        delta2 = (a2 - y_true)
        dW2 = delta2.reshape(-1,1) @ a1_b.reshape(1,-1)
        W2_no_bias = self.W2[:, :self.n_hidden]
        delta1_raw = (delta2 @ W2_no_bias).reshape(-1)
        delta1 = delta1_raw * d_sigmoid_from_activation(a1_nb)
        dW1 = delta1.reshape(-1,1) @ x_b.reshape(1,-1)
        return {"dW1": dW1, "dW2": dW2, "delta2": delta2, "delta1": delta1}

    def step(self, dW1, dW2):
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2

    def train(self, X, y, epochs=200, verbose_every=50, seed=None):
        """
        Train the neural network with reproducible shuffling.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            verbose_every: Print loss every N epochs (0 for no printing)
            seed: Random seed for shuffling (if None, uses current numpy state)
        """
        losses = []
        
        # Create a separate RNG for training if seed is provided
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        for ep in range(1, epochs+1):
            idxs = np.arange(len(X))
            rng.shuffle(idxs)  # Use seeded RNG for reproducible shuffling
            for i in idxs:
                cache = self.forward(X[i])
                grads = self.backward(cache, y[i])
                self.step(grads["dW1"], grads["dW2"])
            
            # Calculate dataset loss
            y_hats = []
            for i in range(len(X)):
                y_hats.append(self.forward(X[i])["a2"][0])
            y_hats = np.array(y_hats).reshape(-1,1)
            L = bce_loss(y, y_hats)
            losses.append(L)
            if verbose_every and ep % verbose_every == 0:
                print(f"Epoch {ep:4d} | Loss: {L:.6f}")
        return losses

    def demo_step_by_step(self, x_in, threshold=0.5, print_round=5):
        cache = self.forward(x_in)
        rnd = lambda v: np.round(v, print_round)
        activated = cache["a1_nb"] > threshold
        print("\n=== FEEDFORWARD (Algorithm 3) ===")
        print(f"Input (standardized) x: {rnd(x_in)}")
        print("\nHidden layer pre-activations n1 = W1 · x_b:")
        print(rnd(cache["n1"]))
        print("\nHidden activations a1 = σ(n1):")
        print(rnd(cache['a1_nb']))
        print(f"Activated (a1 > {threshold}): {activated.astype(int)}")
        print("\nOutput pre-activation n2 = W2 · a1_b:")
        print(rnd(cache["n2"]))
        print("Output activation a2 = σ(n2):")
        print(rnd(cache["a2"]))
        return cache

    def demo_backprop(self, x_in, y_true, print_round=5):
        cache = self.forward(x_in)
        grads = self.backward(cache, y_true)
        rnd = lambda v: np.round(v, print_round)
        print("\n=== BACKPROP (Algorithm 4) ===")
        print("Output error term delta2 = (y_hat - y):", rnd(grads["delta2"]))
        print("Weight/bias corrections at output dW2 (delta2 * a1_b^T):\n", rnd(grads["dW2"]))
        print("\nHidden error term delta1:\n", rnd(grads["delta1"]))
        print("Weight/bias corrections at hidden dW1 (delta1 * x_b^T): shape", grads["dW1"].shape)
        print(rnd(grads["dW1"]))

def load_mushroom_data(filepath):
    """
    Load mushroom data from CSV file with semicolon delimiter.
    Expected format: class;cap-diameter;cap-shape;cap-surface;cap-color;does-bruise-or-bleed;gill-attachment;gill-spacing;gill-color;stem-height;stem-width;stem-root;stem-surface;stem-color;veil-type;veil-color;has-ring;ring-type;spore-print-color;habitat;season
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector
    """
    try:
        # Read CSV file with semicolon delimiter
        df = pd.read_csv(filepath, delimiter=';')
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Handle missing values (replace empty strings with NaN, then with mode/median)
        df = df.replace('', np.nan)
        
        # Define numerical and categorical columns
        numerical_cols = ['cap-diameter', 'stem-height', 'stem-width']
        categorical_cols = [col for col in df.columns if col not in numerical_cols + ['class']]
        
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Process numerical columns
        for col in numerical_cols:
            if col in df.columns:
                # Convert to float, replacing non-numeric values with median
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: converted to float, filled NaN with {median_val}")
        
        # Process categorical columns with label encoding
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Fill missing values with mode (most frequent value)
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_val)
                
                # Apply label encoding
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                print(f"  {col}: encoded {len(le.classes_)} categories")
        
        # Process target variable
        if 'class' in df.columns:
            # Map 'p' (poisonous) to 0, 'e' (edible) to 1
            class_mapping = {'p': 0, 'e': 1}
            df['class'] = df['class'].map(class_mapping)
            
            # Handle any unmapped values
            if df['class'].isna().any():
                print("Warning: Some class values could not be mapped!")
                df = df.dropna(subset=['class'])
        
        # Separate features and target
        X = df.drop('class', axis=1).values.astype(float)
        y = df['class'].values.reshape(-1, 1).astype(int)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Class distribution: Poisonous={np.sum(y==0)}, Edible={np.sum(y==1)}")
        
        return X, y
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        return None, None
    except ImportError:
        print("Error: sklearn not found. Installing...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            from sklearn.preprocessing import LabelEncoder
            print("sklearn installed successfully. Please run again.")
        except Exception as install_error:
            print(f"Failed to install sklearn: {install_error}")
            print("Please install manually: pip install scikit-learn")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def reproducible_train_test_split(X, y, test_size=0.2, seed=42):
    """
    Split data into training and testing sets in a reproducible way.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test data (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Set seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Get total number of samples
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create shuffled indices
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"🔀 Reproducible split (seed={seed}):")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"   Testing: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test=None):
    """
    Standardize features to have mean 0 and std 1.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
    
    Returns:
        Standardized features and statistics
    """
    # Calculate mean and std from training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    # Standardize training data
    X_train_std = (X_train - mean) / std
    
    # Standardize test data if provided
    X_test_std = None
    if X_test is not None:
        X_test_std = (X_test - mean) / std
    
    return X_train_std, X_test_std, mean, std

def save_incorrect_predictions(model, X_test_std, y_test, X_test_original, feature_names, 
                             filename="incorrect_predictions.csv"):
    """
    Save misclassified rows to a CSV file for analysis.
    
    Args:
        model: Trained neural network model
        X_test_std: Standardized test features
        y_test: True test labels
        X_test_original: Original (non-standardized) test features
        feature_names: List of feature column names
        filename: Name of output CSV file
    
    Returns:
        DataFrame of incorrect predictions
    """
    print(f"\n📊 Analyzing incorrect predictions...")
    
    # Get predictions for all test samples
    predictions = []
    probabilities = []
    
    for i in range(len(X_test_std)):
        cache = model.forward(X_test_std[i])
        probability = cache["a2"][0]
        prediction = 1 if probability > 0.5 else 0
        predictions.append(prediction)
        probabilities.append(probability)
    
    predictions = np.array(predictions)
    actual_labels = y_test.flatten()
    
    # Find incorrect predictions
    incorrect_mask = predictions != actual_labels
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) == 0:
        print("🎉 Perfect model! No incorrect predictions found.")
        return None
    
    # Create DataFrame with incorrect predictions
    incorrect_data = []
    
    for idx in incorrect_indices:
        row_data = {}
        
        # Add original features
        for j, feature_name in enumerate(feature_names):
            row_data[feature_name] = X_test_original[idx, j]
        
        # Add prediction information
        row_data['true_label'] = actual_labels[idx]
        row_data['predicted_label'] = predictions[idx]
        row_data['prediction_probability'] = probabilities[idx]
        row_data['true_class_name'] = 'Edible' if actual_labels[idx] == 1 else 'Poisonous'
        row_data['predicted_class_name'] = 'Edible' if predictions[idx] == 1 else 'Poisonous'
        row_data['test_sample_index'] = idx
        
        incorrect_data.append(row_data)
    
    # Create DataFrame and save to CSV
    incorrect_df = pd.DataFrame(incorrect_data)
    incorrect_df.to_csv(filename, index=False)
    
    print(f"❌ Found {len(incorrect_indices)} incorrect predictions out of {len(X_test_std)} test samples")
    print(f"📝 Saved incorrect predictions to: {filename}")
    
    # Show summary statistics
    print(f"\n📈 INCORRECT PREDICTION ANALYSIS:")
    print(f"   • Total errors: {len(incorrect_indices)}")
    print(f"   • Error rate: {(len(incorrect_indices)/len(X_test_std))*100:.2f}%")
    
    # Show breakdown by error type
    false_positives = np.sum((predictions[incorrect_mask] == 1) & (actual_labels[incorrect_mask] == 0))
    false_negatives = np.sum((predictions[incorrect_mask] == 0) & (actual_labels[incorrect_mask] == 1))
    
    print(f"   • False Positives (predicted Edible, actually Poisonous): {false_positives} ⚠️ DANGEROUS!")
    print(f"   • False Negatives (predicted Poisonous, actually Edible): {false_negatives}")
    
    # Show sample of incorrect predictions
    if len(incorrect_df) <= 5:
        print(f"\n📋 All incorrect predictions:")
        for _, row in incorrect_df.iterrows():
            print(f"   Sample {int(row['test_sample_index'])}: True={row['true_class_name']}, "
                  f"Predicted={row['predicted_class_name']} (prob={row['prediction_probability']:.3f})")
    else:
        print(f"\n📋 First 5 incorrect predictions:")
        for i in range(5):
            row = incorrect_df.iloc[i]
            print(f"   Sample {int(row['test_sample_index'])}: True={row['true_class_name']}, "
                  f"Predicted={row['predicted_class_name']} (prob={row['prediction_probability']:.3f})")
    
    return incorrect_df

def predict_mushroom(model, x_input, feature_names=None):
    """
    Make prediction for a single mushroom sample.
    
    Args:
        model: Trained TinyNN model
        x_input: Input features (should be standardized)
        feature_names: Optional list of feature names for display
    
    Returns:
        Prediction probability and classification
    """
    cache = model.forward(x_input)
    probability = cache["a2"][0]
    
    # Standard threshold-based prediction
    # Since our target is: 0=poisonous, 1=edible
    # If probability > 0.5, predict class 1 (edible), else predict class 0 (poisonous)
    prediction = 1 if probability > 0.5 else 0
    
    if feature_names:
        print(f"\nInput features:")
        for i, (name, value) in enumerate(zip(feature_names, x_input)):
            print(f"  {name}: {value:.3f}")
    
    print(f"\nPrediction:")
    print(f"  Probability: {probability:.4f}")
    print(f"  Class: {prediction} ({'Edible' if prediction == 1 else 'Poisonous'})")
    
    return probability, prediction

def interactive_demo():
    """
    Interactive demo for loading data and training the model with full reproducibility.
    """
    print("=== Mushroom Classification Neural Network ===")
    
    # Set reproducibility seed
    seed_input = input("Enter random seed for reproducibility (default: 42): ").strip()
    global_seed = int(seed_input) if seed_input else 42
    set_all_seeds(global_seed)
    
    # Get training file path from user
    train_filepath = input("Enter path to training CSV file (or press Enter for 'mushroom_train.csv'): ").strip()
    if not train_filepath:
        train_filepath = "mushroom_train.csv"
    
    # Get testing file path from user
    test_filepath = input("Enter path to testing CSV file (or press Enter for 'mushroom_test.csv'): ").strip()
    if not test_filepath:
        test_filepath = "mushroom_test.csv"
    
    # Load training data
    print(f"\nLoading training data from: {train_filepath}")
    X_train, y_train = load_mushroom_data(train_filepath)
    
    if X_train is None or y_train is None:
        print("Failed to load training data. Exiting.")
        return
    
    # Load testing data
    print(f"Loading testing data from: {test_filepath}")
    X_test, y_test = load_mushroom_data(test_filepath)
    
    if X_test is None or y_test is None:
        print("Failed to load testing data. Exiting.")
        return
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features (keep original for incorrect predictions analysis)
    X_train_std, X_test_std, mean, std = standardize_features(X_train, X_test)
    
    # Store feature names for analysis
    feature_names = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
                    'gill-attachment', 'gill-spacing', 'gill-color', 'stem-height', 'stem-width',
                    'stem-root', 'stem-surface', 'stem-color', 'veil-type', 'veil-color', 
                    'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season']
    
    # Initialize and train model with reproducible settings
    n_features = X_train_std.shape[1]
    n_hidden = int(input(f"\nEnter number of hidden neurons (default 10): ") or "10")
    learning_rate = float(input("Enter learning rate (default 0.1): ") or "0.1")
    epochs = int(input("Enter number of epochs (default 200): ") or "200")
    
    print(f"\n🧠 Initializing neural network (seed={global_seed})...")
    print(f"  Input features: {n_features}")
    print(f"  Hidden neurons: {n_hidden}")
    print(f"  Output neurons: 1")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Global seed: {global_seed}")
    
    # Initialize model with the same seed for reproducible weights
    model = TinyNN.init(n_features, n_hidden, 1, seed=global_seed, lr=learning_rate)
    
    print(f"\n🏃 Training for {epochs} epochs with reproducible shuffling...")
    losses = model.train(X_train_std, y_train, epochs=epochs, verbose_every=50, seed=global_seed+1)
    
    # Evaluate on test set and save incorrect predictions
    print(f"\nEvaluating on test set...")
    correct = 0
    for i in range(len(X_test_std)):
        cache = model.forward(X_test_std[i])
        prediction = 1 if cache["a2"][0] > 0.5 else 0
        if prediction == y_test[i][0]:
            correct += 1
    
    accuracy = correct / len(X_test_std)
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{len(X_test_std)})")
    
    # Save incorrect predictions to CSV
    incorrect_df = save_incorrect_predictions(model, X_test_std, y_test, X_test, 
                                            feature_names, "incorrect_predictions.csv")
    
    # Show model internal demo
    demo_choice = input("\nWould you like to see the internal workings of the model? (y/n): ").lower()
    if demo_choice == 'y':
        demo_sample_idx = int(input(f"Enter test sample index for demo (0-{len(X_test_std)-1}): ") or "0")
        if 0 <= demo_sample_idx < len(X_test_std):
            print(f"\n{'='*60}")
            print(f"DEMO: Internal Model Workings for Test Sample {demo_sample_idx}")
            print(f"{'='*60}")
            
            # Show step-by-step forward pass
            model.demo_step_by_step(X_test_std[demo_sample_idx])
            
            # Show backpropagation if user wants
            show_backprop = input("\nShow backpropagation demo? (y/n): ").lower()
            if show_backprop == 'y':
                model.demo_backprop(X_test_std[demo_sample_idx], y_test[demo_sample_idx])
            
            print(f"\nActual class: {y_test[demo_sample_idx][0]} ({'Edible' if y_test[demo_sample_idx][0] == 1 else 'Poisonous'})")
    
    # Interactive prediction and analysis
    while True:
        print("\n" + "="*50)
        choice = input("Options:\n1. Make prediction on test sample\n2. Enter custom values\n3. Show model internals for sample\n4. Analyze incorrect predictions\n5. Exit\nChoice (1-5): ")
        
        if choice == "1":
            if len(X_test_std) == 0:
                print("No test samples available!")
                continue
            idx = int(input(f"Enter test sample index (0-{len(X_test_std)-1}): "))
            if 0 <= idx < len(X_test_std):
                feature_names = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 
                               'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
                               'stem-height', 'stem-width', 'stem-root', 'stem-surface', 
                               'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type',
                               'spore-print-color', 'habitat', 'season']
                prob, pred = predict_mushroom(model, X_test_std[idx], feature_names)
                print(f"Actual class: {y_test[idx][0]} ({'Edible' if y_test[idx][0] == 1 else 'Poisonous'})")
            else:
                print("Invalid index!")
                
        elif choice == "2":
            print("Enter feature values:")
            try:
                features = []
                feature_names = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 
                               'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
                               'stem-height', 'stem-width', 'stem-root', 'stem-surface', 
                               'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type',
                               'spore-print-color', 'habitat', 'season']
                for name in feature_names:
                    value = float(input(f"  {name}: "))
                    features.append(value)
                
                # Standardize the input
                features = np.array(features)
                features_std = (features - mean) / std
                
                prob, pred = predict_mushroom(model, features_std, feature_names)
                
                # Ask if user wants to see internals
                show_internals = input("Show internal model workings for this prediction? (y/n): ").lower()
                if show_internals == 'y':
                    print(f"\n{'='*60}")
                    print("INTERNAL MODEL WORKINGS")
                    print(f"{'='*60}")
                    model.demo_step_by_step(features_std)
                
            except ValueError:
                print("Invalid input! Please enter numeric values.")
        
        elif choice == "3":
            if len(X_test_std) == 0:
                print("No test samples available!")
                continue
            idx = int(input(f"Enter test sample index (0-{len(X_test_std)-1}): "))
            if 0 <= idx < len(X_test_std):
                print(f"\n{'='*60}")
                print(f"INTERNAL MODEL WORKINGS - Test Sample {idx}")
                print(f"{'='*60}")
                
                # Show step-by-step forward pass
                model.demo_step_by_step(X_test_std[idx])
                
                # Show backpropagation
                show_backprop = input("\nShow backpropagation? (y/n): ").lower()
                if show_backprop == 'y':
                    model.demo_backprop(X_test_std[idx], y_test[idx])
                
                print(f"\nActual class: {y_test[idx][0]} ({'Edible' if y_test[idx][0] == 1 else 'Poisonous'})")
            else:
                print("Invalid index!")
                
        elif choice == "4":
            # Analyze incorrect predictions
            print("\n🔍 INCORRECT PREDICTIONS ANALYSIS")
            print("="*50)
            
            if incorrect_df is not None and len(incorrect_df) > 0:
                print(f"Found {len(incorrect_df)} incorrect predictions.")
                
                # Ask user what they want to see
                analysis_choice = input("\nWhat would you like to see?\n1. Show all incorrect predictions\n2. Show false positives only (DANGEROUS errors)\n3. Show false negatives only\n4. Statistical summary\nChoice (1-4): ")
                
                if analysis_choice == "1":
                    print("\n📋 ALL INCORRECT PREDICTIONS:")
                    print(incorrect_df[['test_sample_index', 'true_class_name', 'predicted_class_name', 
                                     'prediction_probability']].to_string(index=False))
                
                elif analysis_choice == "2":
                    false_positives = incorrect_df[
                        (incorrect_df['predicted_label'] == 1) & (incorrect_df['true_label'] == 0)
                    ]
                    print(f"\n⚠️ FALSE POSITIVES (Predicted Edible, Actually Poisonous): {len(false_positives)}")
                    if len(false_positives) > 0:
                        print(false_positives[['test_sample_index', 'prediction_probability']].to_string(index=False))
                    else:
                        print("✅ No false positives found!")
                
                elif analysis_choice == "3":
                    false_negatives = incorrect_df[
                        (incorrect_df['predicted_label'] == 0) & (incorrect_df['true_label'] == 1)
                    ]
                    print(f"\n📝 FALSE NEGATIVES (Predicted Poisonous, Actually Edible): {len(false_negatives)}")
                    if len(false_negatives) > 0:
                        print(false_negatives[['test_sample_index', 'prediction_probability']].to_string(index=False))
                    else:
                        print("✅ No false negatives found!")
                
                elif analysis_choice == "4":
                    print(f"\n📊 STATISTICAL SUMMARY:")
                    print(f"   Total test samples: {len(X_test_std)}")
                    print(f"   Incorrect predictions: {len(incorrect_df)}")
                    print(f"   Error rate: {(len(incorrect_df)/len(X_test_std))*100:.2f}%")
                    print(f"   Accuracy: {accuracy:.4f}")
                    
                    fp_count = len(incorrect_df[(incorrect_df['predicted_label'] == 1) & (incorrect_df['true_label'] == 0)])
                    fn_count = len(incorrect_df[(incorrect_df['predicted_label'] == 0) & (incorrect_df['true_label'] == 1)])
                    print(f"   False Positives: {fp_count} (⚠️ DANGEROUS!)")
                    print(f"   False Negatives: {fn_count}")
            else:
                print("🎉 Perfect model! No incorrect predictions to analyze.")
                
        elif choice == "5":
            break
        else:
            print("Invalid choice!")
    
    return model, X_train_std, y_train, X_test_std, y_test

def test_reproducibility():
    """
    Test that the model produces identical results with the same seed.
    """
    print("🔬 REPRODUCIBILITY TEST")
    print("=" * 50)
    
    # Test with small synthetic data
    print("Creating synthetic data...")
    set_all_seeds(42)
    X_synth = np.random.randn(100, 4)
    y_synth = (np.sum(X_synth, axis=1) > 0).astype(int).reshape(-1, 1)
    
    # Run 1
    print("\n🔄 Run 1 (seed=123):")
    set_all_seeds(123)
    model1 = TinyNN.init(4, 5, 1, seed=123, lr=0.1)
    losses1 = model1.train(X_synth, y_synth, epochs=50, verbose_every=0, seed=123)
    final_loss1 = losses1[-1]
    print(f"Final loss: {final_loss1:.8f}")
    
    # Run 2 (same seed)
    print("\n🔄 Run 2 (seed=123):")
    set_all_seeds(123)
    model2 = TinyNN.init(4, 5, 1, seed=123, lr=0.1)
    losses2 = model2.train(X_synth, y_synth, epochs=50, verbose_every=0, seed=123)
    final_loss2 = losses2[-1]
    print(f"Final loss: {final_loss2:.8f}")
    
    # Run 3 (different seed)
    print("\n🔄 Run 3 (seed=456):")
    set_all_seeds(456)
    model3 = TinyNN.init(4, 5, 1, seed=456, lr=0.1)
    losses3 = model3.train(X_synth, y_synth, epochs=50, verbose_every=0, seed=456)
    final_loss3 = losses3[-1]
    print(f"Final loss: {final_loss3:.8f}")
    
    # Check reproducibility
    print("\n📊 RESULTS:")
    if abs(final_loss1 - final_loss2) < 1e-10:
        print("✅ REPRODUCIBLE: Runs 1 and 2 identical (same seed)")
    else:
        print("❌ NOT REPRODUCIBLE: Runs 1 and 2 differ")
        
    if abs(final_loss1 - final_loss3) > 1e-6:
        print("✅ RANDOMNESS WORKING: Run 3 different (different seed)")
    else:
        print("⚠️ Runs 1 and 3 too similar - may indicate issue")
    
    print(f"\nDifference Run1-Run2: {abs(final_loss1 - final_loss2):.2e}")
    print(f"Difference Run1-Run3: {abs(final_loss1 - final_loss3):.2e}")

if __name__ == "__main__":
    print("🍄 MUSHROOM CLASSIFICATION NEURAL NETWORK")
    print("=" * 50)
    print("Choose an option:")
    print("1. Interactive Training Demo")
    print("2. Test Reproducibility") 
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        test_reproducibility()
    elif choice == "3":
        print("Goodbye! 🍄")
    else:
        print("Invalid choice! Running interactive demo...")
        interactive_demo()
