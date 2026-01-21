"""
Football Player Salary Prediction - Flask API Service
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)

# Load model and artifacts
MODEL_DIR = Path('models')

print("Loading model and artifacts...")
with open(MODEL_DIR / 'best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(MODEL_DIR / 'feature_names.json', 'r') as f:
    feature_names = json.load(f)

with open(MODEL_DIR / 'model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model loaded: {metadata['model_name']}")
print(f"Test R²: {metadata['test_r2']:.4f}")
print(f"Features: {feature_names}")

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Football Player Salary Prediction API',
        'model': metadata['model_name'],
        'version': '1.0',
        'test_r2': metadata['test_r2'],
        'test_rmse': metadata['test_rmse'],
        'test_mae': metadata['test_mae'],
        'endpoints': {
            '/': 'API information (this page)',
            '/predict': 'POST - Make salary prediction',
            '/health': 'GET - Health check'
        },
        'features_required': feature_names,
        'example_request': {
            'Is_top_5_League': 1,
            'Based_rich_nation': 1,
            'Is_top_ranked_nation': 2,
            'EU_National': 1,
            'Caps': 70,
            'Apps': 221,
            'Age': 24,
            'Reputation': 9415,
            'Is_top_prev_club': 0
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict player salary
    
    Expected JSON format:
    {
        "Is_top_5_League": int,
        "Based_rich_nation": int,
        "Is_top_ranked_nation": int,
        "EU_National": int,
        "Caps": int,
        "Apps": int,
        "Age": int,
        "Reputation": int,
        "Is_top_prev_club": int
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide player features in JSON format'
            }), 400
        
        # Check for missing features
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing features',
                'missing_features': missing_features,
                'required_features': feature_names
            }), 400
        
        # Extract features in correct order
        features = [data[f] for f in feature_names]
        
        # Create DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction (in log space)
        prediction_log = model.predict(input_scaled)[0]
        
        # Transform back to original scale
        prediction = np.expm1(prediction_log)
        
        # Determine confidence based on feature values
        confidence = 'medium'
        if data.get('Reputation', 0) > 8000 and data.get('Caps', 0) > 50:
            confidence = 'high'
        elif data.get('Reputation', 0) < 3000:
            confidence = 'low'
        
        # Format response
        response = {
            'predicted_salary': float(prediction),
            'predicted_salary_formatted': f"${prediction:,.0f}",
            'model_used': metadata['model_name'],
            'confidence': confidence,
            'input_features': data,
            'model_performance': {
                'test_r2': metadata['test_r2'],
                'test_mae': metadata['test_mae']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict salaries for multiple players
    
    Expected JSON format:
    {
        "players": [
            {...player1_features...},
            {...player2_features...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'players' not in data:
            return jsonify({
                'error': 'Invalid format',
                'message': 'Please provide "players" array in JSON'
            }), 400
        
        players = data['players']
        predictions = []
        
        for i, player in enumerate(players):
            # Check for missing features
            missing_features = [f for f in feature_names if f not in player]
            if missing_features:
                predictions.append({
                    'player_index': i,
                    'error': 'Missing features',
                    'missing_features': missing_features
                })
                continue
            
            # Extract and predict
            features = [player[f] for f in feature_names]
            input_df = pd.DataFrame([features], columns=feature_names)
            input_scaled = scaler.transform(input_df)
            prediction_log = model.predict(input_scaled)[0]
            prediction = np.expm1(prediction_log)
            
            predictions.append({
                'player_index': i,
                'predicted_salary': float(prediction),
                'predicted_salary_formatted': f"${prediction:,.0f}",
                'input_features': player
            })
        
        return jsonify({
            'predictions': predictions,
            'total_players': len(players),
            'successful_predictions': len([p for p in predictions if 'error' not in p]),
            'model_used': metadata['model_name']
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING SALARY PREDICTION API SERVICE")
    print("="*70)
    print(f"Model: {metadata['model_name']}")
    print(f"Performance: R² = {metadata['test_r2']:.4f}, MAE = ${metadata['test_mae']:,.0f}")
    print("\nEndpoints:")
    print("  GET  /         - API information")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Single prediction")
    print("  POST /batch_predict - Batch predictions")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
