# Update the RandomForestClassifier parameters to:
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

# Add cross-validation after model training:
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean():.2f}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_PATH = Path('fertilizer_model.pkl')
CROP_ENCODER_PATH = Path('crop_encoder.pkl')
SOIL_ENCODER_PATH = Path('soil_encoder.pkl')
DATA_PATH = Path('data/fertilizer_recommendation_dataset.csv')
CACHE_PATH = Path('data/cache')

def load_data():
    """Load and cache the dataset."""
    cache_file = CACHE_PATH / 'processed_data.pkl'
    
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    
    # Create cache directory if it doesn't exist
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Read data efficiently using pandas
    df = pd.read_csv(DATA_PATH)
    
    # Cache the processed data
    df.to_pickle(cache_file)
    return df

def load_model():
    """Load or train the model with caching."""
    try:
        # Try to load existing model and encoders
        model = joblib.load(MODEL_PATH)
        crop_encoder = joblib.load(CROP_ENCODER_PATH)
        soil_encoder = joblib.load(SOIL_ENCODER_PATH)
        print("Loaded existing model and encoders")
        return model, crop_encoder, soil_encoder
    except:
        print("Model or encoders not found. Training new model...")
        return train_model()

def train_model():
    """Train the model with optimized data loading."""
    # Load data
    df = load_data()
    
    # Prepare features and target
    X = df[['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 
            'Phosphorous', 'Potassium', 'Carbon', 'Soil', 'Crop']]
    y = df['Fertilizer']
    
    # Encode categorical variables
    crop_encoder = LabelEncoder()
    soil_encoder = LabelEncoder()
    
    X['Crop'] = crop_encoder.fit_transform(X['Crop'])
    X['Soil'] = soil_encoder.fit_transform(X['Soil'])
    
    # Split data with stratification to ensure balanced representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Ensure balanced representation of all crops
    )
    
    # Train model with improved parameters
    model = RandomForestClassifier(
        n_estimators=300,  # Increased number of trees
        max_depth=20,      # Increased depth
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train with progress feedback
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nModel Evaluation:")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(crop_encoder, CROP_ENCODER_PATH)
    joblib.dump(soil_encoder, SOIL_ENCODER_PATH)
    
    return model, crop_encoder, soil_encoder

def get_crop_specific_ranges(crop):
    """Return parameter ranges specific to the selected crop."""
    ranges = {
        'rice': {
            'temperature': (20, 35),
            'moisture': (0.6, 0.9),
            'rainfall': (150, 300),
            'ph': (5.5, 7.0),
            'nitrogen': (60, 100),
            'phosphorous': (40, 80),
            'potassium': (60, 100),
            'carbon': (0.5, 2.0)
        },
        'wheat': {
            'temperature': (15, 25),
            'moisture': (0.4, 0.7),
            'rainfall': (50, 150),
            'ph': (6.0, 7.5),
            'nitrogen': (50, 80),
            'phosphorous': (40, 70),
            'potassium': (40, 70),
            'carbon': (0.5, 2.0)
        },
        'mung bean': {
            'temperature': (25, 35),
            'moisture': (0.5, 0.8),
            'rainfall': (30, 80),
            'ph': (6.0, 7.0),
            'nitrogen': (50, 70),
            'phosphorous': (40, 60),
            'potassium': (40, 60),
            'carbon': (0.5, 2.0)
        }
    }
    return ranges.get(crop.lower(), ranges['rice'])  # Default to rice ranges if crop not found

def get_user_input():
    """Get user input for soil parameters with crop selection from all available crops."""
    # Load data to get available crops
    df = load_data()
    available_crops = sorted(df['Crop'].unique())
    available_soils = sorted(df['Soil'].unique())
    
    print("\nAvailable Crops:")
    for i, crop in enumerate(available_crops, 1):
        print(f"{i}. {crop}")
    
    print("\nAvailable Soil Types:")
    for i, soil in enumerate(available_soils, 1):
        print(f"{i}. {soil}")
    
    # Get crop selection
    while True:
        try:
            crop_choice = int(input(f"\nSelect crop number (1-{len(available_crops)}): "))
            if 1 <= crop_choice <= len(available_crops):
                crop_type = available_crops[crop_choice - 1]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get soil selection
    while True:
        try:
            soil_choice = int(input(f"\nSelect soil type number (1-{len(available_soils)}): "))
            if 1 <= soil_choice <= len(available_soils):
                soil_type = available_soils[soil_choice - 1]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get numerical parameters
    print("\nEnter soil parameters:")
    while True:
        try:
            temperature = float(input("Temperature (°C): "))
            moisture = float(input("Moisture (%): "))
            rainfall = float(input("Rainfall (mm): "))
            ph = float(input("pH: "))
            nitrogen = float(input("Nitrogen (kg/ha): "))
            phosphorous = float(input("Phosphorous (kg/ha): "))
            potassium = float(input("Potassium (kg/ha): "))
            carbon = float(input("Carbon (%): "))
            break
        except ValueError:
            print("Please enter valid numerical values.")
    
    return {
        'Temperature': temperature,
        'Moisture': moisture,
        'Rainfall': rainfall,
        'PH': ph,
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorous,
        'Potassium': potassium,
        'Carbon': carbon,
        'Soil': soil_type,
        'Crop': crop_type
    }

def analyze_low_confidence(prediction, confidence, input_data, df):
    """Analyze low confidence predictions."""
    if confidence < 0.6:
        print("\nLow confidence prediction. Analyzing possible reasons...")
        
        # Get similar cases from dataset
        similar_cases = df[
            (df['Crop'] == input_data['Crop']) & 
            (df['Soil'] == input_data['Soil'])
        ]
        
        if not similar_cases.empty:
            print("\nSimilar cases in dataset:")
            for _, case in similar_cases.head(3).iterrows():
                print(f"\nCrop: {case['Crop']}")
                print(f"Soil: {case['Soil']}")
                print(f"Recommended Fertilizer: {case['Fertilizer']}")
                print(f"Remark: {case['Remark']}")
        
        # Check for parameter ranges
        print("\nParameter analysis:")
        for param in ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']:
            value = input_data[param]
            param_range = df[param].agg(['min', 'max'])
            if value < param_range['min'] or value > param_range['max']:
                print(f"- {param} value ({value}) is outside typical range ({param_range['min']:.2f} - {param_range['max']:.2f})")

def main():
    # Load model and data
    model, crop_encoder, soil_encoder = load_model()
    df = load_data()
    
    print("Welcome to the Fertilizer Recommendation System!")
    print("This system supports predictions for 32 different crops.")
    
    while True:
        try:
            # Get user input
            input_data = get_user_input()
            
            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            input_df['Crop'] = crop_encoder.transform(input_df['Crop'])
            input_df['Soil'] = soil_encoder.transform(input_df['Soil'])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df).max()
            
            # Get fertilizer details
            fertilizer_details = df[df['Fertilizer'] == prediction].iloc[0]
            
            # Display results
            print("\nPrediction Results:")
            print(f"Recommended Fertilizer: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Description: {fertilizer_details['Remark']}")
            
            # Analyze low confidence predictions
            if confidence < 0.6:
                analyze_low_confidence(prediction, confidence, input_data, df)
            
            # Ask to continue
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error: {str(e)}")
            if input("Try again? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main() 