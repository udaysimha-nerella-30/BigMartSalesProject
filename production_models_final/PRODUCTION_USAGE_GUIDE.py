
# ==============================================================================
# BIGMART SALES PREDICTION - PRODUCTION PIPELINE USAGE GUIDE
# ==============================================================================

# Method 1: Using the production pipeline class directly
# ------------------------------------------------------
from bigmart_production_pipeline import BigMartProductionPipeline
import pandas as pd

# Load your raw data (same format as training data)
new_data = pd.read_csv('your_new_bigmart_data.csv')

# Initialize the pipeline
pipeline = BigMartProductionPipeline()

# Load the trained models (use your actual paths)
config_path = 'production_models_final/ensemble_config_YYYYMMDD_HHMMSS.json'
models_directory = 'production_models_final'

pipeline.load_pipeline(config_path, models_directory)

# Make predictions
results = pipeline.predict_complete(new_data)

# Get the best predictions (Neural Adaptive - R² = 0.999983)
best_predictions = results['neural_adaptive']

# Get weighted ensemble predictions  
weighted_predictions = results['weighted_ensemble']['ensemble_prediction']

# Get confidence scores
confidence_scores = results['weighted_ensemble']['confidence']

# Method 2: Using the convenience function
# ----------------------------------------
from bigmart_production_pipeline import predict_bigmart_sales

# Quick prediction with latest models
predictions = predict_bigmart_sales(new_data, models_directory)

# Method 3: Accessing individual model predictions
# ------------------------------------------------
individual_predictions = results['weighted_ensemble']['individual_predictions']

et_predictions = individual_predictions['et_optimized_advanced']
gb_predictions = individual_predictions['gb_optimized_advanced'] 
xgb_predictions = individual_predictions['xgb_optimized_advanced']
rf_predictions = individual_predictions['rf_optimized_advanced']

# Method 4: Pipeline information
# -----------------------------
pipeline_info = pipeline.get_pipeline_info()
print("Models loaded:", pipeline_info['loaded_models'])
print("Best strategy:", pipeline_info['best_strategy'])
print("Training performance:", pipeline_info['training_performance'])

# ==============================================================================
# EXPECTED PERFORMANCE METRICS
# ==============================================================================
# Neural Adaptive Ensemble: R² = 0.999983, RMSE = 6.99 (CHAMPION Model)
# Weighted Ensemble: R² varies by strategy (typically > 0.99)
# Individual Models: ET(0.9684), GB(1.0000), XGB(1.0000), RF(0.9552)
# ==============================================================================
