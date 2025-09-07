
import pickle
import json
import numpy as np
import pandas as pd
import os

class BigMartProductionPipeline:
    """
    Production-ready BigMart sales prediction pipeline

    This class encapsulates the complete pipeline including:
    - Data preprocessing with the exact same preprocessor used in training
    - Ensemble prediction using optimized weighted models
    - Multiple ensemble strategies (weighted average + neural adaptive)

    Usage:
        # Initialize and load pipeline
        pipeline = BigMartProductionPipeline()
        pipeline.load_pipeline(config_path, models_directory)

        # Make predictions
        results = pipeline.predict_complete(raw_data)

        # Get ensemble prediction
        ensemble_pred = results['weighted_ensemble']['ensemble_prediction']

        # Get neural adaptive prediction (best performer)
        neural_pred = results['neural_adaptive']
    """

    def __init__(self, config_path=None, models_directory=None):
        """
        Initialize the production pipeline

        Args:
            config_path: Path to ensemble configuration JSON file
            models_directory: Directory containing saved models
        """
        self.preprocessor = None
        self.models = {}
        self.ensemble_config = None
        self.sophisticated_ensembles = {}
        self.is_loaded = False

        if config_path and models_directory:
            self.load_pipeline(config_path, models_directory)

    def load_pipeline(self, config_path, models_directory):
        """Load all pipeline components from saved files"""
        print(f"Loading production pipeline from: {models_directory}")

        # Load configuration
        with open(config_path, 'rb') as f:
            self.ensemble_config = json.loads(f.read().decode('utf-8'))
        print(f"✓ Configuration loaded")

        # Load preprocessor
        with open(self.ensemble_config['preprocessor_path'], 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded")

        # Load individual models
        for model_name, model_path in self.ensemble_config['model_paths'].items():
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"✓ {model_name} loaded")

        # Load sophisticated ensembles
        sophisticated_path = models_directory + f"/sophisticated_ensembles_{self.ensemble_config['timestamp']}.pkl"
        if os.path.exists(sophisticated_path):
            with open(sophisticated_path, 'rb') as f:
                self.sophisticated_ensembles = pickle.load(f)
            print(f"✓ Sophisticated ensembles loaded: {len(self.sophisticated_ensembles)} models")

        self.is_loaded = True
        print(f"✓ Pipeline fully loaded and ready for production!")

    def preprocess_data(self, raw_data):
        """
        Preprocess raw input data using the exact same preprocessor

        Args:
            raw_data: Raw DataFrame with same structure as training data

        Returns:
            Preprocessed DataFrame ready for model prediction
        """
        if not self.is_loaded:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        if self.preprocessor is None:
            raise ValueError("Preprocessor not available.")

        # Apply the same preprocessing as used during training
        processed_data = self.preprocessor.transform(raw_data)

        return processed_data

    def predict_weighted_ensemble(self, processed_data):
        """
        Make predictions using the optimized weighted ensemble

        Args:
            processed_data: Preprocessed data

        Returns:
            dict: Predictions and confidence metrics
        """
        if not self.is_loaded:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        # Get predictions from all individual models
        model_predictions = []
        model_names = []

        for model_name, model in self.models.items():
            pred = model.predict(processed_data)
            model_predictions.append(pred)
            model_names.append(model_name)

        # Create prediction matrix
        pred_matrix = np.column_stack(model_predictions)

        # Apply best weights for ensemble prediction
        best_weights = np.array(self.ensemble_config['best_weights'])
        ensemble_prediction = np.dot(pred_matrix, best_weights)

        # Calculate prediction confidence (based on model agreement)
        pred_std = np.std(model_predictions, axis=0)
        confidence = 1 / (1 + pred_std)  # Higher when models agree

        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': dict(zip(model_names, model_predictions)),
            'confidence': confidence,
            'strategy_used': self.ensemble_config['best_strategy'],
            'weights_used': best_weights.tolist()
        }

    def predict_neural_adaptive(self, processed_data):
        """
        Make predictions using the neural adaptive ensemble

        Args:
            processed_data: Preprocessed data

        Returns:
            Neural adaptive ensemble predictions
        """
        if not self.is_loaded:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        if 'neural_adaptive' not in self.sophisticated_ensembles:
            raise ValueError("Neural adaptive ensemble not available.")

        # Get base model predictions
        base_predictions = []
        for model_name, model in self.models.items():
            pred = model.predict(processed_data)
            base_predictions.append(pred)

        # Create feature matrix for neural ensemble
        base_pred_matrix = np.column_stack(base_predictions)

        # Neural adaptive prediction
        neural_prediction = self.sophisticated_ensembles['neural_adaptive'].predict(base_pred_matrix)

        return neural_prediction

    def predict_complete(self, raw_data):
        """
        Complete prediction pipeline: preprocess + predict with all methods

        Args:
            raw_data: Raw DataFrame with same structure as training data

        Returns:
            dict: Complete prediction results with all ensemble methods
        """
        if not self.is_loaded:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        # Step 1: Preprocess data
        processed_data = self.preprocess_data(raw_data)

        # Step 2: Get weighted ensemble predictions
        weighted_results = self.predict_weighted_ensemble(processed_data)

        # Step 3: Get neural adaptive predictions (if available)
        neural_prediction = None
        if 'neural_adaptive' in self.sophisticated_ensembles:
            neural_prediction = self.predict_neural_adaptive(processed_data)

        # Step 4: Get other sophisticated ensemble predictions
        other_ensemble_predictions = {}
        for ensemble_name, ensemble_model in self.sophisticated_ensembles.items():
            if ensemble_name != 'neural_adaptive':
                try:
                    if hasattr(ensemble_model, 'predict'):
                        # For voting/stacking ensembles that can predict directly
                        other_prediction = ensemble_model.predict(processed_data)
                        other_ensemble_predictions[ensemble_name] = other_prediction
                except Exception as e:
                    # Some ensembles might need base predictions as input
                    pass

        return {
            'weighted_ensemble': weighted_results,
            'neural_adaptive': neural_prediction,
            'other_ensembles': other_ensemble_predictions,
            'data_shape': processed_data.shape,
            'pipeline_info': {
                'timestamp': self.ensemble_config['timestamp'],
                'best_strategy': self.ensemble_config['best_strategy'],
                'training_performance': self.ensemble_config['performance_metrics']
            }
        }

    def get_pipeline_info(self):
        """Get information about the loaded pipeline"""
        if not self.is_loaded:
            return "Pipeline not loaded"

        info = {
            'loaded_models': list(self.models.keys()),
            'sophisticated_ensembles': list(self.sophisticated_ensembles.keys()),
            'best_strategy': self.ensemble_config['best_strategy'],
            'training_performance': self.ensemble_config['performance_metrics'],
            'timestamp': self.ensemble_config['timestamp']
        }
        return info

# Convenience function for quick predictions
def predict_bigmart_sales(raw_data, models_directory, config_file=None):
    """
    Convenience function for making BigMart sales predictions

    Args:
        raw_data: Raw DataFrame with BigMart data
        models_directory: Directory containing saved models
        config_file: Optional specific config file (latest will be used if None)

    Returns:
        Predictions from the best ensemble method
    """
    # Find latest config file if not specified
    if config_file is None:
        config_files = [f for f in os.listdir(models_directory) if f.startswith('ensemble_config_')]
        if not config_files:
            raise ValueError(f"No ensemble config found in {models_directory}")
        config_file = max(config_files)  # Latest file

    config_path = os.path.join(models_directory, config_file)

    # Initialize and load pipeline
    pipeline = BigMartProductionPipeline()
    pipeline.load_pipeline(config_path, models_directory)

    # Make predictions
    results = pipeline.predict_complete(raw_data)

    # Return the best performing predictions (neural adaptive)
    if results['neural_adaptive'] is not None:
        return results['neural_adaptive']
    else:
        return results['weighted_ensemble']['ensemble_prediction']
