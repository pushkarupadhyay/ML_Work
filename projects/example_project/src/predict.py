"""
Prediction Script

This script loads a trained model and makes predictions on new data.
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path


def load_model_and_scaler(model_path, scaler_path):
    """
    Load trained model and scaler.
    
    Args:
        model_path (str): Path to the model file
        scaler_path (str): Path to the scaler file
        
    Returns:
        tuple: (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, data_path, output_path=None):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        data_path (str): Path to input data
        output_path (str): Path to save predictions
        
    Returns:
        pd.DataFrame: Predictions
    """
    # Load data
    data = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {data.shape}")
    
    # Scale data
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data_scaled)
    
    # Create output dataframe
    results = pd.DataFrame({
        'prediction': predictions
    })
    
    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(data_scaled)
            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]
        except Exception as e:
            print(f"Warning: Could not generate probabilities: {e}")
    
    print(f"\nPredictions made: {len(predictions)}")
    print(f"Unique predictions: {results['prediction'].unique()}")
    
    # Save predictions
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--scaler-path',
        type=str,
        required=True,
        help='Path to fitted scaler file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/predictions.csv',
        help='Path to save predictions'
    )
    
    args = parser.parse_args()
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)
    
    # Make predictions
    predict(model, scaler, args.input, args.output)


if __name__ == '__main__':
    main()
