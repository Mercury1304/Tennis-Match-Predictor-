#!/usr/bin/env python3
"""
Tennis Match Predictor - Data Analysis Module
==============================================

This module provides comprehensive analysis of tennis Grand Slam data,
including statistical insights, visualizations, and model performance metrics.

Author: Tennis Predictor Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TennisDataAnalyzer:
    """
    A comprehensive analyzer for tennis Grand Slam data.
    
    This class provides methods to analyze tennis match data, create visualizations,
    and evaluate machine learning models for predicting match outcomes.
    """
    
    def __init__(self, data_path='Mens_Tennis_Grand_Slam_Winner.csv'):
        """
        Initialize the analyzer with the tennis dataset.
        
        Args:
            data_path (str): Path to the CSV file containing tennis data
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        
    def load_data(self):
        """Load and perform initial data exploration."""
        print("Loading tennis Grand Slam data...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.df)} matches from {self.df['YEAR'].min()}-{self.df['YEAR'].max()}")
            
            # Display basic info
            print(f"\nDataset Overview:")
            print(f"   • Total matches: {len(self.df):,}")
            print(f"   • Years covered: {self.df['YEAR'].min()} - {self.df['YEAR'].max()}")
            print(f"   • Tournaments: {', '.join(self.df['TOURNAMENT'].unique())}")
            print(f"   • Surfaces: {', '.join(self.df['TOURNAMENT_SURFACE'].unique())}")
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
        return True
    
    def clean_data(self):
        """Clean and prepare the dataset for analysis."""
        print("\nCleaning and preparing data...")
        
        # Remove rows with missing rankings
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING'])
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"   • Removed {removed_count} matches with missing rankings")
        
        # Create useful features
        self.df['ranking_diff'] = self.df['RUNNER-UP_ATP_RANKING'] - self.df['WINNER_ATP_RANKING']
        self.df['is_higher_ranked_winner'] = (self.df['WINNER_ATP_RANKING'] < self.df['RUNNER-UP_ATP_RANKING']).astype(int)
        self.df['upset'] = (self.df['WINNER_ATP_RANKING'] > self.df['RUNNER-UP_ATP_RANKING']).astype(int)
        
        print(f"   • Created ranking difference and upset indicators")
        print(f"   • Final dataset: {len(self.df)} matches")
        
    def analyze_tournaments(self):
        """Analyze tournament-specific statistics."""
        print("\nTournament Analysis:")
        
        tournament_stats = self.df.groupby('TOURNAMENT').agg({
            'YEAR': ['count', 'min', 'max'],
            'upset': 'mean',
            'ranking_diff': 'mean'
        }).round(2)
        
        tournament_stats.columns = ['Matches', 'First Year', 'Last Year', 'Upset Rate', 'Avg Ranking Diff']
        print(tournament_stats)
        
        # Create tournament visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tournament Analysis', fontsize=16, fontweight='bold')
        
        # Match count by tournament
        match_counts = self.df['TOURNAMENT'].value_counts()
        axes[0, 0].bar(match_counts.index, match_counts.values, color='skyblue')
        axes[0, 0].set_title('Matches by Tournament')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Upset rate by tournament
        upset_rates = self.df.groupby('TOURNAMENT')['upset'].mean()
        axes[0, 1].bar(upset_rates.index, upset_rates.values, color='lightcoral')
        axes[0, 1].set_title('Upset Rate by Tournament')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Surface distribution
        surface_counts = self.df['TOURNAMENT_SURFACE'].value_counts()
        axes[1, 0].pie(surface_counts.values, labels=surface_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Surface Distribution')
        
        # Ranking difference over time
        yearly_ranking_diff = self.df.groupby('YEAR')['ranking_diff'].mean()
        axes[1, 1].plot(yearly_ranking_diff.index, yearly_ranking_diff.values, marker='o')
        axes[1, 1].set_title('Average Ranking Difference Over Time')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Ranking Difference')
        
        plt.tight_layout()
        plt.savefig('tournament_analysis.png', dpi=300, bbox_inches='tight')
        print("   Tournament analysis chart saved as 'tournament_analysis.png'")
        
    def analyze_players(self):
        """Analyze player performance statistics."""
        print("\nPlayer Analysis:")
        
        # Top winners
        top_winners = self.df['WINNER'].value_counts().head(10)
        print(f"Top 10 Winners:")
        for i, (player, wins) in enumerate(top_winners.items(), 1):
            print(f"   {i:2d}. {player}: {wins} wins")
        
        # Upset analysis
        upset_matches = self.df[self.df['upset'] == 1]
        print(f"\nUpset Analysis:")
        print(f"   • Total upsets: {len(upset_matches)} ({len(upset_matches)/len(self.df)*100:.1f}%)")
        print(f"   • Biggest upset ranking difference: {upset_matches['ranking_diff'].max()}")
        
        # Create player visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top winners bar chart
        top_10_winners = self.df['WINNER'].value_counts().head(10)
        axes[0].barh(range(len(top_10_winners)), top_10_winners.values, color='gold')
        axes[0].set_yticks(range(len(top_10_winners)))
        axes[0].set_yticklabels(top_10_winners.index)
        axes[0].set_title('Top 10 Winners')
        axes[0].set_xlabel('Number of Wins')
        
        # Upset frequency over time
        yearly_upsets = self.df.groupby('YEAR')['upset'].mean()
        axes[1].plot(yearly_upsets.index, yearly_upsets.values, marker='o', color='red')
        axes[1].set_title('Upset Rate Over Time')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Upset Rate')
        
        plt.tight_layout()
        plt.savefig('player_analysis.png', dpi=300, bbox_inches='tight')
        print("   Player analysis chart saved as 'player_analysis.png'")
        
    def prepare_model_data(self):
        """Prepare data for machine learning model."""
        print("\nPreparing data for machine learning...")
        
        # Encode categorical variables
        for col in ['TOURNAMENT', 'TOURNAMENT_SURFACE', 'WINNER_LEFT_OR_RIGHT_HANDED']:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
            self.label_encoders[col] = le
        
        # Select features for model
        feature_cols = [
            'YEAR', 'WINNER_ATP_RANKING', 'RUNNER-UP_ATP_RANKING', 
            'ranking_diff', 'is_higher_ranked_winner',
            'TOURNAMENT_encoded', 'TOURNAMENT_SURFACE_encoded', 
            'WINNER_LEFT_OR_RIGHT_HANDED_encoded'
        ]
        
        X = self.df[feature_cols]
        y = self.df['is_higher_ranked_winner']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Training set: {len(X_train)} samples")
        print(f"   • Test set: {len(X_test)} samples")
        print(f"   • Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_model(self, X_train, X_test, y_train, y_test, feature_cols):
        """Train and evaluate the machine learning model."""
        print("\nTraining Random Forest model...")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        print(f"   • Training accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
        print(f"   • Test accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
        print(f"   • Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for i, row in self.feature_importance.head().iterrows():
            print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        # Create model performance visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance
        top_features = self.feature_importance.head(8)
        axes[0].barh(range(len(top_features)), top_features['importance'], color='lightgreen')
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_title('Feature Importance')
        axes[0].set_xlabel('Importance Score')
        
        # Model performance comparison
        metrics = ['Training', 'Test', 'CV Mean']
        scores = [train_score, test_score, cv_scores.mean()]
        colors = ['green', 'blue', 'orange']
        
        bars = axes[1].bar(metrics, scores, color=colors)
        axes[1].set_title('Model Performance')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("   Model performance chart saved as 'model_performance.png'")
        
        return train_score, test_score, cv_scores
    
    def generate_insights(self):
        """Generate key insights from the analysis."""
        print("\nKey Insights:")
        
        insights = [
            f"Dataset covers {len(self.df)} Grand Slam matches from {self.df['YEAR'].min()}-{self.df['YEAR'].max()}",
            f"Most successful player: {self.df['WINNER'].value_counts().index[0]} with {self.df['WINNER'].value_counts().iloc[0]} wins",
            f"Upset rate: {len(self.df[self.df['upset']==1])/len(self.df)*100:.1f}% of matches were upsets",
            f"Average ranking difference: {self.df['ranking_diff'].mean():.1f} positions",
            f"Model achieves {self.model.score(self.df[self.feature_importance['feature'].tolist()], self.df['is_higher_ranked_winner'])*100:.1f}% accuracy",
            f"Most important feature: {self.feature_importance.iloc[0]['feature']}"
        ]
        
        for insight in insights:
            print(f"   {insight}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🎾 Tennis Match Predictor - Data Analysis")
        print("=" * 50)
        
        # Load and clean data
        if not self.load_data():
            return
        
        self.clean_data()
        
        # Perform analyses
        self.analyze_tournaments()
        self.analyze_players()
        
        # Train model
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_model_data()
        train_score, test_score, cv_scores = self.train_model(X_train, X_test, y_train, y_test, feature_cols)
        
        # Generate insights
        self.generate_insights()
        
        print("\nAnalysis complete! Check the generated PNG files for visualizations.")
        print("Generated files:")
        print("   • tournament_analysis.png")
        print("   • player_analysis.png") 
        print("   • model_performance.png")

def main():
    """Main function to run the analysis."""
    analyzer = TennisDataAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 