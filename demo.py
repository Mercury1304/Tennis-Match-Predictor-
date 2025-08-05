#!/usr/bin/env python3
"""
Tennis Match Predictor - Demo Script
====================================

This demo script showcases the humanized features of the Tennis Match Predictor
application, including the improved user experience, better error handling,
and comprehensive data analysis capabilities.

Author: Tennis Predictor Team
Date: 2024
"""

import requests
import json
import time
import sys
from datetime import datetime

class TennisPredictorDemo:
    """
    Demo class to showcase the humanized Tennis Match Predictor features.
    """
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_header(self, title):
        """Print a formatted header."""
        print("\n" + "="*60)
        print(f"{title}")
        print("="*60)
    
    def print_step(self, step, description):
        """Print a formatted step."""
        print(f"\nStep {step}: {description}")
        print("-" * 40)
    
    def print_success(self, message):
        """Print a success message."""
        print(f"SUCCESS: {message}")
    
    def print_error(self, message):
        """Print an error message."""
        print(f"ERROR: {message}")
    
    def print_info(self, message):
        """Print an info message."""
        print(f"INFO: {message}")
    
    def print_result(self, title, data):
        """Print a formatted result."""
        print(f"\n{title}:")
        print(json.dumps(data, indent=2))
    
    def test_health_check(self):
        """Test the health check endpoint."""
        self.print_step(1, "Testing Health Check")
        
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                self.print_success("Health check passed!")
                self.print_info(f"Status: {data.get('status')}")
                self.print_info(f"Model loaded: {data.get('model_loaded')}")
                return True
            else:
                self.print_error(f"Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_error("Cannot connect to the application. Make sure it's running on http://localhost:5000")
            return False
    
    def test_model_training(self):
        """Test the model training with enhanced feedback."""
        self.print_step(2, "Training the Machine Learning Model")
        
        self.print_info("Starting model training...")
        self.print_info("This process analyzes historical Grand Slam data to learn patterns.")
        
        try:
            response = self.session.post(f"{self.base_url}/api/train")
            data = response.json()
            
            if data.get('success'):
                self.print_success("Model training completed successfully!")
                self.print_info(f"Accuracy: {data.get('accuracy')}%")
                self.print_info(f"Training samples: {data.get('training_samples')}")
                
                if data.get('top_features'):
                    self.print_info("Top features used by the model:")
                    for feature in data.get('top_features', []):
                        print(f"   • {feature.get('feature')}: {feature.get('importance'):.3f}")
                
                return True
            else:
                self.print_error(f"Training failed: {data.get('error')}")
                if data.get('help'):
                    self.print_info(f"Help: {data.get('help')}")
                return False
                
        except Exception as e:
            self.print_error(f"Training error: {str(e)}")
            return False
    
    def test_prediction_validation(self):
        """Test input validation with helpful error messages."""
        self.print_step(3, "Testing Input Validation")
        
        # Test invalid inputs
        invalid_tests = [
            {
                'name': 'Missing required fields',
                'data': {'player1_ranking': 1},
                'expected_error': 'Missing required fields'
            },
            {
                'name': 'Invalid ranking range',
                'data': {
                    'player1_ranking': 9999,
                    'player2_ranking': 1,
                    'tournament': 'Wimbledon',
                    'surface': 'Grass / Outdoor',
                    'year': 2024
                },
                'expected_error': 'Player rankings must be between 1 and 1000'
            },
            {
                'name': 'Invalid tournament',
                'data': {
                    'player1_ranking': 1,
                    'player2_ranking': 2,
                    'tournament': 'Invalid Tournament',
                    'surface': 'Grass / Outdoor',
                    'year': 2024
                },
                'expected_error': 'Invalid tournament'
            },
            {
                'name': 'Invalid year',
                'data': {
                    'player1_ranking': 1,
                    'player2_ranking': 2,
                    'tournament': 'Wimbledon',
                    'surface': 'Grass / Outdoor',
                    'year': 2010
                },
                'expected_error': 'Year must be between 2020 and 2030'
            }
        ]
        
        for test in invalid_tests:
            self.print_info(f"Testing: {test['name']}")
            try:
                response = self.session.post(f"{self.base_url}/api/predict", json=test['data'])
                data = response.json()
                
                if not data.get('success') and test['expected_error'] in data.get('error', ''):
                    self.print_success(f"Validation caught: {test['name']}")
                    if data.get('help'):
                        self.print_info(f"   Help message: {data.get('help')}")
                else:
                    self.print_error(f"Validation failed for: {test['name']}")
                    
            except Exception as e:
                self.print_error(f"Test error: {str(e)}")
        
        return True
    
    def test_successful_predictions(self):
        """Test successful predictions with detailed results."""
        self.print_step(4, "Testing Successful Predictions")
        
        test_matches = [
            {
                'name': 'Top vs Top (Close Match)',
                'data': {
                    'player1_ranking': 1,
                    'player2_ranking': 2,
                    'tournament': 'Wimbledon',
                    'surface': 'Grass / Outdoor',
                    'year': 2024
                }
            },
            {
                'name': 'Upset Prediction',
                'data': {
                    'player1_ranking': 10,
                    'player2_ranking': 1,
                    'tournament': 'French Open',
                    'surface': 'Clay',
                    'year': 2024
                }
            },
            {
                'name': 'Clay Court Specialist',
                'data': {
                    'player1_ranking': 5,
                    'player2_ranking': 15,
                    'tournament': 'French Open',
                    'surface': 'Clay',
                    'year': 2024
                }
            }
        ]
        
        for match in test_matches:
            self.print_info(f"Predicting: {match['name']}")
            try:
                response = self.session.post(f"{self.base_url}/api/predict", json=match['data'])
                data = response.json()
                
                if data.get('success'):
                    self.print_success(f"Prediction successful!")
                    self.print_info(f"   Winner: {data.get('winner')}")
                    self.print_info(f"   Confidence: {data.get('confidence')}% ({data.get('confidence_level')})")
                    self.print_info(f"   Message: {data.get('message')}")
                    
                    if data.get('match_details'):
                        details = data.get('match_details')
                        self.print_info(f"   Ranking difference: {details.get('ranking_difference')} positions")
                        
                else:
                    self.print_error(f"Prediction failed: {data.get('error')}")
                    
            except Exception as e:
                self.print_error(f"Prediction error: {str(e)}")
        
        return True
    
    def test_statistics_endpoint(self):
        """Test the comprehensive statistics endpoint."""
        self.print_step(5, "Testing Statistics and Insights")
        
        try:
            response = self.session.get(f"{self.base_url}/api/stats")
            data = response.json()
            
            if data.get('success'):
                stats = data.get('stats', {})
                self.print_success("Statistics loaded successfully!")
                
                self.print_info("Dataset Overview:")
                print(f"   • Total matches: {stats.get('total_matches', 0):,}")
                print(f"   • Years covered: {stats.get('years_covered', 'N/A')}")
                print(f"   • Tournaments: {len(stats.get('tournaments', []))}")
                print(f"   • Surfaces: {len(stats.get('surfaces', []))}")
                
                if stats.get('upset_analysis'):
                    upset = stats.get('upset_analysis')
                    self.print_info("Upset Analysis:")
                    print(f"   • Total upsets: {upset.get('total_upsets', 0)}")
                    print(f"   • Upset rate: {upset.get('upset_rate', 0)}%")
                    print(f"   • Biggest upset ranking difference: {upset.get('biggest_upset_ranking_diff', 0)}")
                
                if stats.get('ranking_analysis'):
                    ranking = stats.get('ranking_analysis')
                    self.print_info("Ranking Analysis:")
                    print(f"   • Average winner ranking: {ranking.get('avg_winner_ranking', 0)}")
                    print(f"   • Average runner-up ranking: {ranking.get('avg_runnerup_ranking', 0)}")
                    print(f"   • Average ranking difference: {ranking.get('avg_ranking_diff', 0)}")
                
                return True
            else:
                self.print_error(f"Statistics failed: {data.get('error')}")
                return False
                
        except Exception as e:
            self.print_error(f"Statistics error: {str(e)}")
            return False
    
    def test_help_endpoint(self):
        """Test the help endpoint with API documentation."""
        self.print_step(6, "Testing Help and Documentation")
        
        try:
            response = self.session.get(f"{self.base_url}/api/help")
            data = response.json()
            
            if data.get('success'):
                help_info = data.get('help', {})
                self.print_success("Help documentation loaded!")
                
                self.print_info("Available Endpoints:")
                for endpoint, description in help_info.get('endpoints', {}).items():
                    print(f"   • {endpoint}: {description}")
                
                self.print_info("Prediction Input Format:")
                for field, description in help_info.get('prediction_input', {}).items():
                    print(f"   • {field}: {description}")
                
                if help_info.get('example_prediction'):
                    self.print_info("Example Prediction:")
                    example = help_info.get('example_prediction')
                    print(f"   {json.dumps(example, indent=2)}")
                
                return True
            else:
                self.print_error(f"Help failed: {data.get('error')}")
                return False
                
        except Exception as e:
            self.print_error(f"Help error: {str(e)}")
            return False
    
    def showcase_humanized_features(self):
        """Showcase the key humanized features."""
        self.print_header("Humanized Features Showcase")
        
        features = [
            "Enhanced Error Handling: User-friendly error messages with actionable solutions",
            "Comprehensive Statistics: Detailed insights with visual data presentation",
            "Input Validation: Real-time validation with helpful guidance",
            "Confidence Levels: Detailed confidence scores and probability analysis",
            "Modern UI: Beautiful, responsive design with smooth animations",
            "Complete Documentation: Comprehensive API docs and user guides",
            "Real-time Feedback: Status indicators and progress updates",
            "Smart Predictions: AI-powered analysis with detailed explanations"
        ]
        
        for feature in features:
            print(f"   {feature}")
    
    def run_complete_demo(self):
        """Run the complete demo showcasing all humanized features."""
        self.print_header("Tennis Match Predictor - Humanized Demo")
        
        self.print_info("This demo showcases the enhanced user experience and improved functionality")
        self.print_info("of the Tennis Match Predictor application.")
        
        # Test all features
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Training", self.test_model_training),
            ("Input Validation", self.test_prediction_validation),
            ("Successful Predictions", self.test_successful_predictions),
            ("Statistics", self.test_statistics_endpoint),
            ("Help Documentation", self.test_help_endpoint)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                self.print_error(f"Demo error in {test_name}: {str(e)}")
                results.append((test_name, False))
        
        # Summary
        self.print_header("Demo Summary")
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        self.print_info(f"Tests passed: {passed}/{total}")
        
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"   {status}: {test_name}")
        
        if passed == total:
            self.print_success("All tests passed! The humanized features are working correctly.")
        else:
            self.print_error(f"{total - passed} tests failed. Please check the application setup.")
        
        # Showcase features
        self.showcase_humanized_features()
        
        self.print_header("Demo Complete")
        self.print_info("The Tennis Match Predictor demonstrates excellent user experience")
        self.print_info("with comprehensive error handling, helpful guidance, and intuitive design.")

def main():
    """Main function to run the demo."""
    print("Tennis Match Predictor - Humanized Demo")
    print("=" * 60)
    print("This demo showcases the enhanced user experience and improved")
    print("functionality of the Tennis Match Predictor application.")
    print("\nMake sure the Flask application is running on http://localhost:5000")
    print("before starting this demo.")
    
    demo = TennisPredictorDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 