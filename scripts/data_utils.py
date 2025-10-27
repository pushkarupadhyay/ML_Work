"""
Data Utilities

Common utility functions for data manipulation and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def display_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Display comprehensive information about a DataFrame.
    
    Args:
        df: Input DataFrame
        name: Name to display for the DataFrame
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"{'='*60}\n")


def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualize missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        figsize: Figure size as (width, height)
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values found!")
        return
    
    plt.figure(figsize=figsize)
    missing.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot correlation matrix heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        figsize: Figure size as (width, height)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str] = None, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the IQR method.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers (None = all numeric columns)
        factor: IQR factor for outlier detection (default: 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"Original shape: {df.shape}")
    print(f"Shape after removing outliers: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows")
    
    return df_clean


def encode_categorical_features(df: pd.DataFrame, columns: List[str] = None, 
                                method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode (None = all object columns)
        method: Encoding method ('onehot' or 'label')
        
    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
    elif method == 'label':
        le = LabelEncoder()
        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = 'oversample') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance an imbalanced dataset.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Balancing method ('oversample', 'undersample', 'smote')
        
    Returns:
        Tuple of (balanced_X, balanced_y)
    """
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'smote':
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    print(f"Original class distribution:\n{y.value_counts()}")
    print(f"\nBalanced class distribution:\n{pd.Series(y_balanced).value_counts()}")
    
    return X_balanced, y_balanced
