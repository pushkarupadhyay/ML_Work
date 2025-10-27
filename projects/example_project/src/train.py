"""
Training Script

This script trains a machine learning model on the provided data.
"""

import argparse
import yaml
import joblib
from pathlib import Path

from data_processing import load_data, preprocess_data
from model import MLModel
from sklearn.metrics import accuracy_score, classification_report


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config):
    """
    Train the model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Load and preprocess data
    data = load_data(config['data']['path'])
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data,
        target_column=config['data']['target_column'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Create and train model
    model = MLModel(
        model_type=config['model']['type'],
        **config['model']['params']
    )
    model.train(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{config['model']['type']}_model.joblib"
    scaler_path = output_dir / "scaler.joblib"
    
    joblib.dump(model.model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save metrics
    metrics_path = Path(config['output']['results_dir']) / "metrics.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        f.write(f"Model Type: {config['model']['type']}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
    
    print(f"Metrics saved to: {metrics_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    train(config)


if __name__ == '__main__':
    main()
