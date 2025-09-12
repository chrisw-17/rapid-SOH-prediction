#!/usr/bin/env python3
"""
Neural Network for State of Health (SOH) Prediction from EIS Data

This script implements a fully connected neural network for predicting battery 
State of Health (SOH) from Electrochemical Impedance Spectroscopy (EIS) data.
The model processes impedance spectra from multiple datasets and predicts SOH
with high accuracy.

Dependencies:
    - numpy
    - pandas
    - scikit-learn
    - tensorflow
    - matplotlib
    - seaborn
    - visualkeras
    - scienceplots
    - keras-tuner (optional, for hyperparameter tuning)

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import glob
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import optimizers, callbacks
import visualkeras
import scienceplots

# Set plotting style for publication-quality figures
plt.style.use(['science', 'notebook'])

# Optional imports for hyperparameter tuning
try:
    import keras_tuner as kt
    from tensorflow.keras import layers, models
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False
    print("Warning: keras_tuner not available. Hyperparameter tuning will be disabled.")


def load_eis_datasets():
    """
    Load EIS data from multiple experimental datasets.
    
    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    # Define file paths for different datasets
    datasets = {
        'Experimental_Data_RH': {'pattern': 'Experimental_Data_RH/*.txt', 'params': {
            'decimal': ',', 'encoding': 'unicode_escape', 'sep': '\t', 
            'header': 0, 'skiprows': 57, 'usecols': [1,2,3,4,5,6,7,8,9,10,11]
        }},
        'Warwick': {'pattern': 'Warwick/*.txt', 'params': {
            'decimal': '.', 'sep': '\t', 'header': 0
        }},
        'Multisine_EIS_RH': {'pattern': 'Multisine_EIS_RH/*.txt', 'params': {
            'decimal': '.', 'sep': '\t', 'header': 0
        }},
        'Stanford': {'pattern': 'Stanford/*.txt', 'params': {
            'decimal': '.', 'sep': '\t', 'header': 0
        }}
    }
    
    loaded_data = {}
    
    for dataset_name, config in datasets.items():
        txt_files = glob.glob(config['pattern'])
        # Exclude SOH_values_singlesine.txt from Experimental_Data_RH
        if dataset_name == 'Experimental_Data_RH':
            txt_files = [f for f in txt_files if os.path.basename(f) != 'SOH_values_singlesine.txt']
        for txt_file in txt_files:
            file_name = os.path.splitext(os.path.basename(txt_file))[0]
            try:
                df = pd.read_csv(txt_file, **config['params'])
                loaded_data[file_name] = df
                print(f"Loaded {file_name} from {dataset_name}")
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
    
    return loaded_data


def extract_impedance_features(data_dict, column_pos_re=3, column_pos_im=4, 
                              column_pos_soh=6, freq_range=(10, 50)):
    """
    Extract impedance features from EIS data.
    
    Args:
        data_dict (dict): Dictionary of dataframes with EIS data
        column_pos_re (int): Column position for real impedance
        column_pos_im (int): Column position for imaginary impedance  
        column_pos_soh (int): Column position for SOH values
        freq_range (tuple): Row range for frequency selection (start, end)
        
    Returns:
        tuple: (feature_matrix, soh_values, column_names)
    """
    # Separate datasets by naming convention
    datasets = {
        'Single': [name for name in data_dict.keys() if name.startswith('Single')],
        'Cell': [name for name in data_dict.keys() if name.startswith('Cell')],
        'Multi': [name for name in data_dict.keys() if name.startswith('Multi')],
        'W': [name for name in data_dict.keys() if name.startswith('W')]
    }
    
    all_rows = []
    extracted_soh = []
    row_names = []

    for dataset_type, file_names in datasets.items():
        for name in file_names:
            if name in data_dict:
                df = data_dict[name]
                # Extract impedance data (exclude inductive and final diffusion parts)
                row = df.iloc[freq_range[0]:freq_range[1], [column_pos_re, column_pos_im]].values.flatten()
                all_rows.append(row)
                row_names.append(name)
                # Extract SOH values for datasets that have them in the file
                if dataset_type != 'Single':
                    soh_value = df.iloc[0, column_pos_soh]
                    extracted_soh.append(soh_value)

    column_names = [f'Frequency_{i+1}' for i in range(len(all_rows[0]))]
    feature_matrix = pd.DataFrame(all_rows, index=row_names)
    feature_matrix.columns = column_names
    extracted_soh = [float(val) * 0.01 if val > 1 else float(val) for val in extracted_soh]
    return feature_matrix, extracted_soh, column_names


def apply_ohmic_shift(feature_matrix):
    """
    Apply ohmic shift correction for improved generalizability.
    
    This removes the ohmic resistance (R0) bias by subtracting the first 
    real impedance value from all real impedance values.
    
    Args:
        feature_matrix (pd.DataFrame): Feature matrix with impedance data
        
    Returns:
        pd.DataFrame: Corrected feature matrix
    """
    corrected_matrix = feature_matrix.copy()
    
    for j in range(len(corrected_matrix)):
        # Subtract R0 (first real impedance value) from all real impedance values
        corrected_matrix.iloc[j, ::2] = (corrected_matrix.iloc[j, ::2] - 
                                        corrected_matrix.iloc[j, 0])
    
    return corrected_matrix


def extract_soc_from_filename(filename):
    """
    Extract State of Charge (SOC) from filename using regex patterns.
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        int or None: SOC value in percentage, None if not found
    """
    # Case 1 & 4: "Cell02_95SOH_20SOC_9505_60" or "W10_50SOC_300Cyc"
    match_soc_suffix = re.search(r'(\d+)[ _]?SOC', filename, re.IGNORECASE)
    if match_soc_suffix:
        return int(match_soc_suffix.group(1))

    # Case 2: "[...]_SOC[...]"
    match_eisgalv = re.search(r'_SOC(\d+)', filename, re.IGNORECASE)
    if match_eisgalv:
        return int(match_eisgalv.group(1))

    # Case 3: filename starts with "Multi" → SOC = 50
    if filename.lower().startswith('multi'):
        return 50

    # Fallback: unable to parse
    return None


def create_visualization_with_zoom(feature_matrix, save_path=None):
    """
    Create publication-quality visualization of EIS data with zoom inset.
    
    Args:
        feature_matrix (pd.DataFrame): Feature matrix with impedance data
        save_path (str, optional): Path to save the figure
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Prepare plot data
    plot_data = []
    for k in range(len(feature_matrix)):
        real_z = feature_matrix.iloc[k, ::2]
        imag_z = -feature_matrix.iloc[k, 1::2]
        max_x = real_z.max()
        plot_data.append((max_x, real_z, imag_z))
    
    # Sort by maximum real impedance for consistent coloring
    plot_data.sort(key=lambda x: x[0])
    
    # Setup colors
    num_plots = len(plot_data)
    colors = sns.color_palette("RdYlBu", num_plots)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
    
    # Create main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, real_z, imag_z in plot_data:
        ax.plot(real_z, imag_z, zorder=1)
    
    ax.set_xlabel(r'$Z_{Re}$ [$\Omega$]')
    ax.set_ylabel(r'$-Z_{Im}$ [$\Omega$]')
    ax.set_xlim(-0.002, 0.07)
    ax.set_ylim(0, 0.03)
    ax.set_aspect('equal', adjustable='box')
    
    # Define zoom area for ohmic resistance region
    zoom_xlim = (-0.0008, 0.01)
    zoom_ylim = (0.0, 0.0032)
    
    # Add inset axis
    axins = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=1)
    axins.set_zorder(10)
    
    # Plot same data in the inset
    for _, real_z, imag_z in plot_data:
        axins.plot(real_z, imag_z)
    
    # Set zoom limits
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect('equal', adjustable='box')
    
    # Add R0 arrow annotation in the inset
    axins.annotate('', xytext=(0.005, 0.002), xy=(-0.00035, 0.002),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    axins.text(0.0006, 0.0032, r'$R_0$', ha='center', va='top', fontsize=10)
    
    # Connect main plot to inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def build_neural_network(input_shape=80):
    """
    Build the neural network architecture for SOH prediction.
    
    Args:
        input_shape (int): Number of input features
        
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dropout(0.1),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def plot_prediction_results(y_true, y_pred, soc_labels=None, title_suffix="", save_path=None):
    """
    Create scatter plot of prediction results with performance metrics.
    
    Args:
        y_true (array-like): True SOH values
        y_pred (array-like): Predicted SOH values
        soc_labels (array-like, optional): SOC labels for color coding
        title_suffix (str): Suffix for plot title
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    if soc_labels is not None:
        scatter = plt.scatter(y_true, y_pred, c=soc_labels, cmap='viridis', s=50)
        plt.colorbar(scatter, label="SOC (%)")
    else:
        plt.scatter(y_true, y_pred, s=50)
    
    # Add identity line (perfect predictions)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate and display metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    metrics_text = f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def build_tuning_model(hp):
    """
    Build model for hyperparameter tuning (requires keras-tuner).
    
    Args:
        hp: Hyperparameter object from keras-tuner
        
    Returns:
        tf.keras.Model: Model with hyperparameters to be tuned
    """
    if not KERAS_TUNER_AVAILABLE:
        raise ImportError("keras-tuner is required for hyperparameter tuning")
    
    model = models.Sequential()
    model.add(layers.Input(shape=(80,)))

    # Tune number of hidden layers
    for i in range(hp.Int('n_layers', 2, 10)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=1024, step=64),
            activation=hp.Choice('activation', ['relu', 'elu', 'tanh'])
        ))
        model.add(layers.Dropout(rate=hp.Float('dropout_rate', 0.0, 0.5, step=0.05)))

    model.add(layers.Dense(1))  # Output layer

    # Tune optimizer and learning rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    lr = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = optimizers.SGD(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def perform_hyperparameter_tuning(X_train, y_train, max_trials=30):
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        max_trials (int): Maximum number of trials for tuning
        
    Returns:
        tuple: (best_model, best_hyperparameters)
    """
    if not KERAS_TUNER_AVAILABLE:
        print("Keras-tuner not available. Skipping hyperparameter tuning.")
        return None, None
    
    tuner = kt.BayesianOptimization(
        build_tuning_model,
        objective='val_mae',
        max_trials=max_trials,
        directory='kt_dir',
        project_name='soh_bayes_tuning_01'
    )
    
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=100, 
                                        restore_best_weights=True)
    
    tuner.search(X_train, y_train, epochs=400, validation_split=0.1, 
                callbacks=[stop_early])
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]
    
    return best_model, best_hp


def main():
    """
    Main execution function.
    """
    print("Starting Neural Network SOH Prediction Pipeline...")
    
    # Load datasets
    print("\n1. Loading EIS datasets...")
    data_dict = load_eis_datasets()
    
    if not data_dict:
        print("No data loaded. Please check your data directories.")
        return
    
    # Extract features and SOH values
    print("\n2. Extracting impedance features...")
    feature_matrix, extracted_soh, column_names = extract_impedance_features(data_dict)
    
    # Load experimental SOH values for single-sine data
    try:
        exp_soh_values = np.loadtxt("Experimental_Data_RH/SOH_values_singlesine.txt").tolist()
        all_soh_values = exp_soh_values + extracted_soh
    except FileNotFoundError:
        print("Warning: SOH_values_singlesine.txt not found in Experimental_Data_RH. Using only extracted SOH values.")
        all_soh_values = extracted_soh
    
    # Apply ohmic shift correction
    print("\n3. Applying ohmic shift correction...")
    feature_matrix = apply_ohmic_shift(feature_matrix)
    
    # Extract SOC labels
    soc_labels = feature_matrix.index.to_series().astype(str).apply(extract_soc_from_filename)
    
    # Create final dataframe
    dataframe = feature_matrix.copy()
    dataframe['SOH'] = all_soh_values
    
    # Create visualization
    print("\n4. Creating EIS data visualization...")
    create_visualization_with_zoom(feature_matrix, save_path='eis_visualization.png')
    
    # Prepare training data
    print("\n5. Preparing training data...")
    train_set, test_set, soc_train, soc_test = train_test_split(
        dataframe, soc_labels, test_size=0.2, random_state=42
    )
    # Fill missing SOC values in soc_train and soc_test with 50 for plotting
    soc_train = pd.Series(soc_train).fillna(50).astype(int)
    soc_test = pd.Series(soc_test).fillna(50).astype(int)
    
    # Print dataset distribution
    print('\nTest set distribution:')
    print(test_set['SOH'].value_counts().sort_index())
    print('\nTraining set distribution:')
    print(train_set['SOH'].value_counts().sort_index())
    
    # Reset indices
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    
    # Split features and labels
    X_train, y_train = train_set[column_names], train_set['SOH']
    X_test, y_test = test_set[column_names], test_set['SOH']
    
    # Save untouched test set for validation
    X_validation = X_test.copy(deep=True)
    y_validation = y_test.copy(deep=True)
    
    # Scale features
    print("\n6. Scaling features...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validation_scaled = scaler.transform(X_validation)
    
    # Option 1: Train new model
    print("\n7. Training neural network...")
    model = build_neural_network(input_shape=len(column_names))
    
    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=1000, 
                       validation_split=0.1, verbose=1)
    
    # Evaluate model
    print("\n8. Evaluating model performance...")
    train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"Training - Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Testing - Loss: {test_loss:.6f}, MAE: {test_mae:.6f}")
    
    # Generate predictions
    y_pred_train = model.predict(X_train_scaled).flatten()
    y_pred_test = model.predict(X_test_scaled).flatten()
    
    # Create prediction plots
    print("\n9. Creating prediction plots...")
    plot_prediction_results(y_train, y_pred_train, soc_train, 
                           title_suffix="Training", save_path='training_results.png')
    plot_prediction_results(y_test, y_pred_test, soc_test, 
                           title_suffix="Testing", save_path='testing_results.png')
    
    # Save the trained model
    model.save('trained_soh_model.keras')
    print("Model saved as 'trained_soh_model.keras'")
    
    # Option 2: Load pre-trained model (alternative)
    """
    print("\n7. Loading pre-trained model...")
    try:
        model = load_model('Saved_NN_Models/FCNN_SOH_pred_standard.keras')
        print("Pre-trained model loaded successfully.")
    except:
        print("Pre-trained model not found. Using newly trained model.")
    """
    
    # Optional: Hyperparameter tuning
    """
    if KERAS_TUNER_AVAILABLE:
        print("\n10. Performing hyperparameter tuning (optional)...")
        best_model, best_hp = perform_hyperparameter_tuning(X_train_scaled, y_train)
        
        if best_model is not None:
            # Retrain best model
            best_model.fit(X_train_scaled, y_train, validation_split=0.1, 
                          epochs=1000, verbose=1)
            
            # Evaluate best model
            best_test_loss, best_test_mae = best_model.evaluate(X_test_scaled, y_test, verbose=0)
            print(f"Best model - Test Loss: {best_test_loss:.6f}, Test MAE: {best_test_mae:.6f}")
    """
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
