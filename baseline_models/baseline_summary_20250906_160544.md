
# BigMart Sales - Baseline Model Summary
Generated: 20250906_160544

## 📊 Baseline Performance
- **Cross-Validation R²**: 0.6962 ± 0.0133
- **Cross-Validation RMSE**: $935.43 ± $27.64
- **Validation R²**: 0.2088
- **Validation RMSE**: $1535.87

## 🚨 Key Issues Identified
- **Overfitting**: Validation R² is 0.4874 lower than CV
- **Poor Generalization**: Model doesn't generalize to unseen items
- **Performance Gap**: 70.0% performance loss

## 📁 Data Setup
- **Training Data**: train_data_splitted.csv (6818 records, 1247 items)
- **Validation Data**: validation_data_splitted.csv (1705 records, 312 items)
- **No Data Leakage**: ✅ Confirmed

## 🎯 Next Steps for Improvement
1. **Reduce Item-Specific Overfitting**: Remove/modify item statistics and target encoding
2. **Focus on Generalizable Features**: Emphasize features that work for unseen items
3. **Regularization**: Add model regularization to prevent overfitting
4. **Feature Selection**: Remove features that don't generalize well
5. **Alternative Models**: Try models less prone to overfitting

## 📦 Saved Files
- Model: baseline_model_20250906_160544.pkl
- Pipeline: baseline_preprocessor_20250906_160544.pkl
- Results: baseline_results_20250906_160544.json
