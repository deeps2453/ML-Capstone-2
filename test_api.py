"""
Test script for the Salary Prediction API
"""

import requests
import json

# Configuration
API_URL = "http://localhost:5000"  # Change this for deployed service

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Health check passed")

def test_api_info():
    """Test API info endpoint"""
    print("\n" + "="*70)
    print("Testing API Info Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ API info retrieved successfully")

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Single Prediction")
    print("="*70)
    
    # Example: High-value player
    player_data = {
        "Is_top_5_League": 1,
        "Based_rich_nation": 1,
        "Is_top_ranked_nation": 2,
        "EU_National": 1,
        "Caps": 100,
        "Apps": 350,
        "Age": 27,
        "Reputation": 8500,
        "Is_top_prev_club": 1
    }
    
    print(f"Input: {json.dumps(player_data, indent=2)}")
    
    response = requests.post(
        f"{API_URL}/predict",
        json=player_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Single prediction successful")
    
    return response.json()

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Batch Prediction")
    print("="*70)
    
    players_data = {
        "players": [
            {
                "Is_top_5_League": 1,
                "Based_rich_nation": 1,
                "Is_top_ranked_nation": 2,
                "EU_National": 1,
                "Caps": 100,
                "Apps": 350,
                "Age": 27,
                "Reputation": 8500,
                "Is_top_prev_club": 1
            },
            {
                "Is_top_5_League": 0,
                "Based_rich_nation": 0,
                "Is_top_ranked_nation": 0,
                "EU_National": 0,
                "Caps": 10,
                "Apps": 50,
                "Age": 22,
                "Reputation": 2500,
                "Is_top_prev_club": 0
            },
            {
                "Is_top_5_League": 1,
                "Based_rich_nation": 1,
                "Is_top_ranked_nation": 1,
                "EU_National": 1,
                "Caps": 50,
                "Apps": 200,
                "Age": 25,
                "Reputation": 7000,
                "Is_top_prev_club": 0
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/batch_predict",
        json=players_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Batch prediction successful")

def test_error_handling():
    """Test error handling"""
    print("\n" + "="*70)
    print("Testing Error Handling")
    print("="*70)
    
    # Test with missing features
    incomplete_data = {
        "Is_top_5_League": 1,
        "Age": 27
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=incomplete_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 400
    print("✓ Error handling works correctly")

def test_different_player_profiles():
    """Test predictions for different player profiles"""
    print("\n" + "="*70)
    print("Testing Different Player Profiles")
    print("="*70)
    
    profiles = [
        {
            "name": "Superstar Player",
            "data": {
                "Is_top_5_League": 1,
                "Based_rich_nation": 1,
                "Is_top_ranked_nation": 2,
                "EU_National": 1,
                "Caps": 120,
                "Apps": 400,
                "Age": 28,
                "Reputation": 9200,
                "Is_top_prev_club": 1
            }
        },
        {
            "name": "Rising Star",
            "data": {
                "Is_top_5_League": 1,
                "Based_rich_nation": 1,
                "Is_top_ranked_nation": 1,
                "EU_National": 1,
                "Caps": 15,
                "Apps": 80,
                "Age": 21,
                "Reputation": 6500,
                "Is_top_prev_club": 0
            }
        },
        {
            "name": "Veteran Player",
            "data": {
                "Is_top_5_League": 0,
                "Based_rich_nation": 0,
                "Is_top_ranked_nation": 1,
                "EU_National": 0,
                "Caps": 80,
                "Apps": 500,
                "Age": 35,
                "Reputation": 5000,
                "Is_top_prev_club": 1
            }
        },
        {
            "name": "Young Prospect",
            "data": {
                "Is_top_5_League": 0,
                "Based_rich_nation": 0,
                "Is_top_ranked_nation": 0,
                "EU_National": 0,
                "Caps": 0,
                "Apps": 20,
                "Age": 19,
                "Reputation": 2000,
                "Is_top_prev_club": 0
            }
        }
    ]
    
    results = []
    for profile in profiles:
        response = requests.post(
            f"{API_URL}/predict",
            json=profile["data"],
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                "profile": profile["name"],
                "salary": result["predicted_salary_formatted"],
                "confidence": result["confidence"]
            })
    
    print("\nPrediction Results:")
    print("-" * 70)
    for result in results:
        print(f"{result['profile']:20s} | {result['salary']:20s} | {result['confidence']}")
    print("-" * 70)
    
    print("✓ All player profiles tested successfully")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SALARY PREDICTION API - TEST SUITE")
    print("="*70)
    print(f"Testing API at: {API_URL}")
    
    try:
        # Run all tests
        test_health_check()
        test_api_info()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        test_different_player_profiles()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*70)
        print("ERROR: Cannot connect to API")
        print("="*70)
        print(f"Make sure the API is running at {API_URL}")
        print("Start it with: python src/predict.py")
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("TEST FAILED ✗")
        print("="*70)
        print(f"Error: {e}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("UNEXPECTED ERROR ✗")
        print("="*70)
        print(f"Error: {e}")
