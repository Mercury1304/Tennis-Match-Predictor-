#!/usr/bin/env python3
"""
Tennis Match Predictor - Flask Web Application
==============================================

A user-friendly web application for predicting tennis match outcomes
using machine learning. Features include real-time predictions,
comprehensive statistics, and an intuitive web interface.

Author: Tennis Predictor Team
Date: 2024
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tennis_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
label_encoders = {}
feature_columns = ['YEAR', 'TOURNAMENT', 'WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING', 
                  'WINNER_LEFT_OR_RIGHT_HANDED', 'TOURNAMENT_SURFACE', 'WINNER_NATIONALITY']

# Valid tournament and surface options
VALID_TOURNAMENTS = ['Australian Open', 'French Open', 'Wimbledon', 'U.S. Open']
VALID_SURFACES = ['Plexicushion Prestige', 'Clay', 'Grass / Outdoor', 'DecoTurf - outdoors']

@app.route('/')
def index():
    """Render the main application page."""
    logger.info("Main page accessed")
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train the tennis prediction model with historical data.
    
    Returns:
        JSON response with training results and accuracy metrics
    """
    logger.info("Model training requested")
    
    try:
        # Check if data file exists
        if not os.path.exists('Mens_Tennis_Grand_Slam_Winner.csv'):
            logger.error("Dataset file not found")
            return jsonify({
                'success': False, 
                'error': 'Dataset file not found. Please ensure Mens_Tennis_Grand_Slam_Winner.csv is in the project directory.',
                'help': 'Download the dataset and place it in the project root directory.'
            }), 404
        
        # Load data
        logger.info("Loading tennis dataset...")
        df = pd.read_csv('Mens_Tennis_Grand_Slam_Winner.csv')
        
        # Validate data
        required_columns = ['YEAR', 'TOURNAMENT', 'WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                'success': False,
                'error': f'Dataset is missing required columns: {missing_columns}',
                'help': 'Please ensure the dataset contains all required columns.'
            }), 400
        
        # Clean data
        logger.info("Cleaning and preparing data...")
        initial_count = len(df)
        df = df.dropna(subset=['WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING'])
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} matches with missing rankings")
        
        # Create features for prediction
        df['ranking_diff'] = df['RUNNER-UP_ATP_RANKING'] - df['WINNER_ATP_RANKING']
        df['is_higher_ranked_winner'] = (df['WINNER_ATP_RANKING'] < df['RUNNER-UP_ATP_RANKING']).astype(int)
        
        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        label_encoders = {}
        for col in ['TOURNAMENT', 'TOURNAMENT_SURFACE', 'WINNER_LEFT_OR_RIGHT_HANDED']:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
            label_encoders[col] = le
        
        # Features for model
        X = df[['YEAR', 'WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING', 'ranking_diff', 
                'is_higher_ranked_winner', 'TOURNAMENT_encoded', 'TOURNAMENT_SURFACE_encoded', 
                'WINNER_LEFT_OR_RIGHT_HANDED_encoded']]
        
        # Target: 1 if higher ranked player wins, 0 otherwise
        y = df['is_higher_ranked_winner']
        
        # Train model
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(X, y)
        
        # Save model and encoders
        joblib.dump(model, 'tennis_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        # Calculate accuracy and additional metrics
        accuracy = model.score(X, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model training completed successfully. Accuracy: {accuracy:.3f}")
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully! Ready to predict matches.',
            'accuracy': round(accuracy * 100, 2),
            'training_samples': len(df),
            'removed_samples': removed_count,
            'top_features': feature_importance.head(3).to_dict('records'),
            'model_info': {
                'algorithm': 'Random Forest',
                'n_estimators': 100,
                'max_depth': 10
            }
        })
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({
            'success': False, 
            'error': 'Dataset file not found. Please check if Mens_Tennis_Grand_Slam_Winner.csv exists.',
            'help': 'Download the dataset and place it in the project directory.'
        }), 404
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'Training failed: {str(e)}',
            'help': 'Please check the dataset format and try again.'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """
    Predict match outcome based on player rankings and match conditions.
    
    Expected JSON input:
    {
        "player1_ranking": int,
        "player2_ranking": int,
        "tournament": str,
        "surface": str,
        "year": int
    }
    """
    logger.info("Match prediction requested")
    
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'help': 'Please provide player rankings, tournament, surface, and year.'
            }), 400
        
        # Validate required fields
        required_fields = ['player1_ranking', 'player2_ranking', 'tournament', 'surface', 'year']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}',
                'help': 'Please provide all required match information.'
            }), 400
        
        # Load model and encoders
        if not os.path.exists('tennis_model.pkl'):
            return jsonify({
                'success': False, 
                'error': 'Model not trained yet',
                'help': 'Please train the model first using the /api/train endpoint.'
            }), 400
        
        model = joblib.load('tennis_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        
        # Extract and validate features
        try:
            player1_ranking = int(data['player1_ranking'])
            player2_ranking = int(data['player2_ranking'])
            tournament = str(data['tournament'])
            surface = str(data['surface'])
            year = int(data['year'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid data types: {str(e)}',
                'help': 'Please ensure rankings and year are numbers, tournament and surface are text.'
            }), 400
        
        # Validate input ranges
        if not (1 <= player1_ranking <= 1000 and 1 <= player2_ranking <= 1000):
            return jsonify({
                'success': False,
                'error': 'Player rankings must be between 1 and 1000',
                'help': 'Please enter valid ATP rankings.'
            }), 400
        
        if not (2020 <= year <= 2030):
            return jsonify({
                'success': False,
                'error': 'Year must be between 2020 and 2030',
                'help': 'Please enter a valid year for the match.'
            }), 400
        
        # Validate tournament and surface
        if tournament not in VALID_TOURNAMENTS:
            return jsonify({
                'success': False,
                'error': f'Invalid tournament: {tournament}',
                'help': f'Valid tournaments: {", ".join(VALID_TOURNAMENTS)}'
            }), 400
        
        if surface not in VALID_SURFACES:
            return jsonify({
                'success': False,
                'error': f'Invalid surface: {surface}',
                'help': f'Valid surfaces: {", ".join(VALID_SURFACES)}'
            }), 400
        
        # Calculate features
        ranking_diff = player2_ranking - player1_ranking
        is_higher_ranked_player1 = 1 if player1_ranking < player2_ranking else 0
        
        # Encode categorical variables
        try:
            tournament_encoded = label_encoders['TOURNAMENT'].transform([tournament])[0]
            surface_encoded = label_encoders['TOURNAMENT_SURFACE'].transform([surface])[0]
            handed_encoded = label_encoders['WINNER_LEFT_OR_RIGHT_HANDED'].transform(['right'])[0]
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid tournament or surface: {str(e)}',
                'help': 'Please use valid tournament and surface names from the dropdown.'
            }), 400
        
        # Create feature vector
        features = np.array([[
            year, player1_ranking, player2_ranking, ranking_diff,
            is_higher_ranked_player1, tournament_encoded, surface_encoded, handed_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Determine winner and create user-friendly messages
        if prediction == 1:
            if player1_ranking < player2_ranking:
                winner = "Player 1 (Higher Ranked)"
                confidence = probability[1]
                message = f"Player 1 (Rank #{player1_ranking}) is predicted to win!"
            else:
                winner = "Player 2 (Higher Ranked)"
                confidence = probability[1]
                message = f"Player 2 (Rank #{player2_ranking}) is predicted to win!"
        else:
            if player1_ranking < player2_ranking:
                winner = "Player 2 (Lower Ranked)"
                confidence = probability[0]
                message = f"Upset alert! Player 2 (Rank #{player2_ranking}) is predicted to win!"
            else:
                winner = "Player 1 (Lower Ranked)"
                confidence = probability[0]
                message = f"Upset alert! Player 1 (Rank #{player1_ranking}) is predicted to win!"
        
        # Add confidence level description
        if confidence >= 0.8:
            confidence_level = "Very High"
        elif confidence >= 0.6:
            confidence_level = "High"
        elif confidence >= 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        logger.info(f"Prediction made: {winner} with {confidence:.3f} confidence")
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'winner': winner,
            'message': message,
            'confidence': round(confidence * 100, 2),
            'confidence_level': confidence_level,
            'player1_win_probability': round(probability[1] * 100, 2),
            'player2_win_probability': round(probability[0] * 100, 2),
            'match_details': {
                'player1_ranking': player1_ranking,
                'player2_ranking': player2_ranking,
                'tournament': tournament,
                'surface': surface,
                'year': year,
                'ranking_difference': abs(ranking_diff)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': f'Prediction failed: {str(e)}',
            'help': 'Please check your input and try again.'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get comprehensive tennis statistics and insights.
    
    Returns:
        JSON response with dataset statistics and analysis
    """
    logger.info("Statistics requested")
    
    try:
        if not os.path.exists('Mens_Tennis_Grand_Slam_Winner.csv'):
            return jsonify({
                'success': False,
                'error': 'Dataset not found',
                'help': 'Please ensure the dataset file exists.'
            }), 404
        
        df = pd.read_csv('Mens_Tennis_Grand_Slam_Winner.csv')
        
        # Clean data for statistics
        df_clean = df.dropna(subset=['WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING'])
        
        # Calculate comprehensive statistics
        stats = {
            'total_matches': len(df),
            'clean_matches': len(df_clean),
            'tournaments': df['TOURNAMENT'].unique().tolist(),
            'surfaces': df['TOURNAMENT_SURFACE'].unique().tolist(),
            'years_covered': f"{df['YEAR'].min()} - {df['YEAR'].max()}",
            'total_years': df['YEAR'].max() - df['YEAR'].min() + 1,
            'top_players': df['WINNER'].value_counts().head(5).to_dict(),
            'surface_distribution': df['TOURNAMENT_SURFACE'].value_counts().to_dict(),
            'tournament_distribution': df['TOURNAMENT'].value_counts().to_dict(),
            'upset_analysis': {
                'total_upsets': len(df_clean[df_clean['WINNER_ATP_RANKING'] > df_clean['RUNNER-UP_ATP_RANKING']]),
                'upset_rate': round(len(df_clean[df_clean['WINNER_ATP_RANKING'] > df_clean['RUNNER-UP_ATP_RANKING']]) / len(df_clean) * 100, 1),
                'biggest_upset_ranking_diff': int((df_clean['RUNNER-UP_ATP_RANKING'] - df_clean['WINNER_ATP_RANKING']).max())
            },
            'ranking_analysis': {
                'avg_winner_ranking': round(df_clean['WINNER_ATP_RANKING'].mean(), 1),
                'avg_runnerup_ranking': round(df_clean['RUNNER-UP_ATP_RANKING'].mean(), 1),
                'avg_ranking_diff': round((df_clean['RUNNER-UP_ATP_RANKING'] - df_clean['WINNER_ATP_RANKING']).mean(), 1)
            }
        }
        
        logger.info("Statistics generated successfully")
        
        return jsonify({
            'success': True, 
            'stats': stats,
            'message': 'Tennis statistics loaded successfully!'
        })
        
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Failed to load statistics: {str(e)}',
            'help': 'Please check the dataset and try again.'
        }), 500

@app.route('/api/players', methods=['GET'])
def get_players():
    """
    Get list of all players in the dataset.
    
    Returns:
        JSON response with sorted list of player names
    """
    logger.info("Player list requested")
    
    try:
        if not os.path.exists('Mens_Tennis_Grand_Slam_Winner.csv'):
            return jsonify({
                'success': False,
                'error': 'Dataset not found',
                'help': 'Please ensure the dataset file exists.'
            }), 404
        
        df = pd.read_csv('Mens_Tennis_Grand_Slam_Winner.csv')
        players = sorted(df['WINNER'].unique().tolist())
        
        logger.info(f"Retrieved {len(players)} unique players")
        
        return jsonify({
            'success': True, 
            'players': players,
            'total_players': len(players),
            'message': f'Found {len(players)} unique players in the dataset!'
        })
        
    except Exception as e:
        logger.error(f"Player list error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Failed to load players: {str(e)}',
            'help': 'Please check the dataset and try again.'
        }), 500

@app.route('/api/help', methods=['GET'])
def get_help():
    """
    Get help information and API documentation.
    
    Returns:
        JSON response with API documentation and usage examples
    """
    help_info = {
        'endpoints': {
            'GET /': 'Main web interface',
            'POST /api/train': 'Train the machine learning model',
            'POST /api/predict': 'Predict match outcome',
            'GET /api/stats': 'Get dataset statistics',
            'GET /api/players': 'Get list of players',
            'GET /api/health': 'Health check',
            'GET /api/help': 'This help information'
        },
        'prediction_input': {
            'player1_ranking': 'ATP ranking of player 1 (1-1000)',
            'player2_ranking': 'ATP ranking of player 2 (1-1000)',
            'tournament': 'Tournament name (Australian Open, French Open, Wimbledon, U.S. Open)',
            'surface': 'Court surface (Clay, Grass / Outdoor, etc.)',
            'year': 'Match year (2020-2030)'
        },
        'example_prediction': {
            'player1_ranking': 1,
            'player2_ranking': 5,
            'tournament': 'Wimbledon',
            'surface': 'Grass / Outdoor',
            'year': 2024
        }
    }
    
    return jsonify({
        'success': True,
        'help': help_info,
        'message': 'API documentation and usage examples'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with helpful message."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'help': 'Use /api/help to see available endpoints'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with helpful message."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'help': 'Please try again later or contact support'
    }), 500

if __name__ == '__main__':
    logger.info("Starting Tennis Match Predictor...")
    logger.info("Application ready at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 