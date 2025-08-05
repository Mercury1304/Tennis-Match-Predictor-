import requests
import json
import time

def test_api():
    """Test the tennis prediction API"""
    base_url = "http://localhost:5000"
    
    print("Testing Tennis Prediction API")
    print("=" * 40)
    
    # Test 1: Train model
    print("\n1. Testing model training...")
    try:
        response = requests.post(f"{base_url}/api/train")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Model trained successfully!")
            print(f"   Accuracy: {data.get('accuracy', 'N/A')}%")
            print(f"   Training samples: {data.get('training_samples', 'N/A')}")
        else:
            print(f"ERROR: Training failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Training error: {e}")
        return False
    
    # Test 2: Get statistics
    print("\n2. Testing statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Statistics retrieved successfully!")
            print(f"   Total matches: {data['stats']['total_matches']}")
            print(f"   Tournaments: {len(data['stats']['tournaments'])}")
        else:
            print(f"ERROR: Statistics failed: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Statistics error: {e}")
    
    # Test 3: Get players
    print("\n3. Testing players endpoint...")
    try:
        response = requests.get(f"{base_url}/api/players")
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Players list retrieved successfully!")
            print(f"   Total players: {len(data['players'])}")
        else:
            print(f"ERROR: Players failed: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Players error: {e}")
    
    # Test 4: Make prediction
    print("\n4. Testing prediction...")
    try:
        prediction_data = {
            'player1_ranking': 1,
            'player2_ranking': 5,
            'tournament': 'Wimbledon',
            'surface': 'Grass / Outdoor',
            'year': 2024
        }
        
        response = requests.post(f"{base_url}/api/predict", json=prediction_data)
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Prediction successful!")
            print(f"   Winner: {data['winner']}")
            print(f"   Confidence: {data['confidence']}%")
            print(f"   Player 1 win probability: {data['player1_win_probability']}%")
            print(f"   Player 2 win probability: {data['player2_win_probability']}%")
        else:
            print(f"ERROR: Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"ERROR: Prediction error: {e}")
    
    # Test 5: Test invalid prediction (without training)
    print("\n5. Testing invalid prediction (no model)...")
    try:
        # This should fail if model isn't trained
        prediction_data = {
            'player1_ranking': 1,
            'player2_ranking': 2,
            'tournament': 'Australian Open',
            'surface': 'Plexicushion Prestige',
            'year': 2024
        }
        
        response = requests.post(f"{base_url}/api/predict", json=prediction_data)
        if response.status_code == 400:
            print(f"SUCCESS: Correctly handled missing model!")
        else:
            print(f"WARNING: Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "=" * 40)
    print("API Testing Complete!")
    return True

def test_frontend():
    """Test if the frontend is accessible"""
    print("\nTesting Frontend Access...")
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code == 200:
            print("SUCCESS: Frontend is accessible!")
            print("   Open http://localhost:5000 in your browser")
        else:
            print(f"ERROR: Frontend error: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Frontend error: {e}")

if __name__ == "__main__":
    print("Starting Tennis Prediction API Tests...")
    print("Make sure the Flask app is running on http://localhost:5000")
    print("Run: python app.py")
    print()
    
    # Wait a moment for user to start the app
    input("Press Enter when the Flask app is running...")
    
    # Run tests
    test_api()
    test_frontend()
    
    print("\nAll tests completed!")
    print("If all tests passed, your tennis prediction system is working correctly!") 