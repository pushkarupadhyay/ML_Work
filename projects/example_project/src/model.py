"""
Model Definition Module

This module contains machine learning model definitions and architectures.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class MLModel:
    """
    Base class for machine learning models.
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use
            **kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        
    def _create_model(self, **kwargs):
        """
        Create the specified model.
        
        Returns:
            Model instance
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed.")
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)
