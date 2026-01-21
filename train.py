"""
Football Player Salary Prediction - Training Script
This script trains multiple models and saves the best performing one.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = 'data/wages.csv'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

print("="*70)
print("FOOTBALL PLAYER SALARY PREDICTION - MODEL TRAINING")
print("="*70)

# 1. LOAD DATA
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Remove Last_Transfer_Fee if it exists (all zeros)
if 'Last_Transfer_Fee' in df.columns:
    df = df.drop('Last_Transfer_Fee', axis=1)

print("\nFeatures:")
for col in df.columns:
    if col != 'Salary':
        print(f"  - {col}")

print(f"\nTarget: Salary")
print(f"  Range: ${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}")
print(f"  Mean: ${df['Salary'].mean():,.0f}")
print(f"  Median: ${df['Salary'].median():,.0f}")
print(f"  Skewness: {df['Salary'].skew():.2f}")

# 2. DATA PREPROCESSING
print("\n[2/6] Preprocessing data...")

# Log transformation of target (handles skewness)
df['Log_Salary'] = np.log1p(df['Salary'])
print("Applied log transformation to target variable")

# Separate features and target
X = df.drop(['Salary', 'Log_Salary'], axis=1)
y_original = df['Salary']
y_log = df['Log_Salary']

# Train-test split
X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y_original, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Feature scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using RobustScaler")


# 3. MODEL TRAINING
print("\n[3/6] Training models...")

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
        verbose=0
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train on log-transformed target
    model.fit(X_train, y_train_log)
    
    # Predictions
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    # Transform back to original scale
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)
    
    # Calculate metrics
    train_r2 = r2_score(y_train_orig, y_pred_train)
    test_r2 = r2_score(y_test_orig, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
    train_mae = mean_absolute_error(y_train_orig, y_pred_train)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test)
    
    # Cross-validation on log scale
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Test RMSE: ${test_rmse:,.2f}")
    print(f"  Test MAE: ${test_mae:,.2f}")
    
    results.append({
        'model_name': name,
        'model_object': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'overfitting_gap': train_r2 - test_r2
    })


# 4. MODEL SELECTION
print("\n[4/6] Selecting best model...")

# Sort by test R2
results_sorted = sorted(results, key=lambda x: x['test_r2'], reverse=True)
best_result = results_sorted[0]
best_model = best_result['model_object']
best_model_name = best_result['model_name']

print(f"\nBest Model: {best_model_name}")
print(f"  Test R²: {best_result['test_r2']:.4f}")
print(f"  Test RMSE: ${best_result['test_rmse']:,.2f}")
print(f"  Test MAE: ${best_result['test_mae']:,.2f}")
print(f"  Overfitting Gap: {best_result['overfitting_gap']:.4f}")


# 5. SAVE MODEL AND ARTIFACTS
print("\n[5/6] Saving model and artifacts...")

# Save best model
model_path = MODEL_DIR / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Model saved to: {model_path}")

# Save scaler
scaler_path = MODEL_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_path}")

# Save feature names
feature_names_path = MODEL_DIR / 'feature_names.json'
with open(feature_names_path, 'w') as f:
    json.dump(list(X.columns), f)
print(f"Feature names saved to: {feature_names_path}")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'test_r2': float(best_result['test_r2']),
    'test_rmse': float(best_result['test_rmse']),
    'test_mae': float(best_result['test_mae']),
    'cv_r2_mean': float(best_result['cv_r2_mean']),
    'cv_r2_std': float(best_result['cv_r2_std']),
    'features': list(X.columns),
    'n_features': len(X.columns),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'random_state': RANDOM_STATE
}

metadata_path = MODEL_DIR / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")


# 6. FEATURE IMPORTANCE ANALYSIS


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head().iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Save feature importance
importance_path = MODEL_DIR / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"\nFeature importance saved to: {importance_path}")


# GENERATE VISUALIZATIONS

# 1. Model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

model_names = [r['model_name'] for r in results_sorted]
test_r2_scores = [r['test_r2'] for r in results_sorted]
test_rmse_scores = [r['test_rmse'] for r in results_sorted]

axes[0].barh(model_names, test_r2_scores, color='steelblue')
axes[0].set_xlabel('Test R²')
axes[0].set_title('Model Performance Comparison (R²)')
axes[0].grid(True, alpha=0.3)

axes[1].barh(model_names, test_rmse_scores, color='coral')
axes[1].set_xlabel('Test RMSE ($)')
axes[1].set_title('Model Performance Comparison (RMSE)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"Model comparison saved to: {MODEL_DIR / 'model_comparison.png'}")

# 2. Feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title(f'Feature Importance - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved to: {MODEL_DIR / 'feature_importance.png'}")

# 3. Predictions vs Actual
y_pred_test_log = best_model.predict(X_test)
y_pred_test = np.expm1(y_pred_test_log)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Linear scale
axes[0].scatter(y_test_orig, y_pred_test, alpha=0.5, s=10)
axes[0].plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Salary ($)')
axes[0].set_ylabel('Predicted Salary ($)')
axes[0].set_title('Predictions vs Actual (Linear Scale)')
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].scatter(y_test_orig, y_pred_test, alpha=0.5, s=10)
axes[1].plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Salary ($)')
axes[1].set_ylabel('Predicted Salary ($)')
axes[1].set_title('Predictions vs Actual (Log Scale)')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print(f"Predictions plot saved to: {MODEL_DIR / 'predictions_vs_actual.png'}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nAll artifacts saved to: {MODEL_DIR}/")
print("\nNext steps:")
print("  1. Review model performance in visualizations")
print("  2. Test predictions using predict.py")
print("  3. Deploy the model using Docker or cloud services")
print("\n" + "="*70)
