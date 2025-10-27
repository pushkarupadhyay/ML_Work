"""
Data Processing Module

This module contains functions for loading and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data, target_column, test_size=0.2, random_state=42):
    """
    Preprocess data for machine learning.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of the target column
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        strategy (str): Strategy for imputation ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    if strategy == 'drop':
        return data.dropna()
    elif strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
