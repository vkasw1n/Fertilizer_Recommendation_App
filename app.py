import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        font-size: 60px !important; 
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.3);
        letter-spacing: 3px;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: scale(1.1);
        display: block;
        line-height: 1.2;
    }
    .title-text span {
        background: linear-gradient(45deg, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        font-size: 24px;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .range-info {
        font-size: 14px;
        color: #666;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# Parameter ranges based on sample data
PARAMETER_RANGES = {
    'Temperature': {'min': 20, 'max': 50, 'unit': '°C', 'description': 'Optimal temperature range for crop growth'},
    'Moisture': {'min': 0.3, 'max': 0.8, 'unit': '%', 'description': 'Soil moisture content as a percentage'},
    'Rainfall': {'min': 200, 'max': 400, 'unit': 'mm', 'description': 'Annual rainfall in millimeters'},
    'pH': {'min': 6.0, 'max': 7.5, 'unit': '', 'description': 'Soil pH level (neutral to slightly acidic)'},
    'Nitrogen': {'min': 50, 'max': 100, 'unit': 'kg/ha', 'description': 'Nitrogen content in soil'},
    'Phosphorous': {'min': 50, 'max': 150, 'unit': 'kg/ha', 'description': 'Phosphorous content in soil'},
    'Potassium': {'min': 50, 'max': 150, 'unit': 'kg/ha', 'description': 'Potassium content in soil'},
    'Carbon': {'min': 0.4, 'max': 1.5, 'unit': '%', 'description': 'Soil organic carbon content'},
    'Soil': {'options': ['Loamy Soil', 'Sandy Soil', 'Clay Soil', 'Black Soil', 'Red Soil'], 'description': 'Type of soil'},
    'Crop': {'options': [
        'rice', 'wheat', 'Mung Bean', 'Tea', 'millet', 'maize', 'Lentil', 'Jute', 
        'Coffee', 'Cotton', 'Ground Nut', 'Peas', 'Rubber', 'Sugarcane', 'Tobacco', 
        'Kidney Beans', 'Moth Beans', 'Coconut', 'Black gram', 'Adzuki Beans', 
        'Pigeon Peas', 'Chickpea', 'banana', 'grapes', 'apple', 'mango', 'muskmelon', 
        'orange', 'papaya', 'pomegranate', 'watermelon'
    ], 'description': 'Type of crop'},
    'Fertilizer': {'options': [
        'Compost', 'Balanced NPK Fertilizer', 'Urea', 'DAP', 'MOP',
        'General Purpose Fertilizer', 'Gypsum', 'Lime', 'Muriate of Potash',
        'Organic Fertilizer', 'Water Retaining Fertilizer'
    ], 'description': 'Recommended fertilizer type'},
    'Remark': {'description': 'Additional notes about the recommendation'}
}

def load_model():
    """Load the trained model and encoders if they exist, otherwise train a new model."""
    try:
        if os.path.exists('fertilizer_model.pkl') and os.path.exists('soil_encoder.pkl') and os.path.exists('crop_encoder.pkl'):
            with open('fertilizer_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('soil_encoder.pkl', 'rb') as f:
                soil_encoder = pickle.load(f)
            with open('crop_encoder.pkl', 'rb') as f:
                crop_encoder = pickle.load(f)
            return model, soil_encoder, crop_encoder
    except Exception as e:
        st.warning(f"Error loading existing model: {e}")
        st.info("Training new model...")
    
    # Load and prepare data
    df = pd.read_csv('data/fertilizer_recommendation_dataset.csv')
    X = df[['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon', 'Soil', 'Crop']].copy()
    y = df['Fertilizer']
    
    # Encode categorical variables
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    X['Soil'] = soil_encoder.fit_transform(X['Soil'])
    X['Crop'] = crop_encoder.fit_transform(X['Crop'])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    with open('fertilizer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('soil_encoder.pkl', 'wb') as f:
        pickle.dump(soil_encoder, f)
    with open('crop_encoder.pkl', 'wb') as f:
        pickle.dump(crop_encoder, f)
    
    return model, soil_encoder, crop_encoder

def analyze_low_confidence(input_values, confidence):
    """Analyze why the confidence might be low."""
    reasons = []
    
    params = ['Temperature', 'Moisture', 'Rainfall', 'pH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
    
    for i, param in enumerate(params):
        range_info = PARAMETER_RANGES[param]
        if input_values[i] < range_info['min'] or input_values[i] > range_info['max']:
            reasons.append(f"{param} ({input_values[i]}{range_info['unit']}) is outside the typical range ({range_info['min']}-{range_info['max']}{range_info['unit']})")
    
    if not reasons:
        reasons.append("The combination of parameters is unusual in our training data")
        
    return reasons

def main():
    st.markdown('<p class="title-text">🌱 <span>Fertilizer Recommendation System</span></p>', unsafe_allow_html=True)
    
    # Load model and encoders
    model, soil_encoder, crop_encoder = load_model()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="subtitle-text">Soil Parameters</p>', unsafe_allow_html=True)
        
        # Temperature
        temp_range = PARAMETER_RANGES['Temperature']
        temperature = st.slider(
            "Temperature (°C)",
            min_value=float(temp_range['min']),
            max_value=float(temp_range['max']),
            value=float((temp_range['min'] + temp_range['max']) / 2),
            help=temp_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {temp_range["min"]}-{temp_range["max"]}{temp_range["unit"]}</p>', unsafe_allow_html=True)
        
        # Moisture
        moisture_range = PARAMETER_RANGES['Moisture']
        moisture = st.slider(
            "Moisture (%)",
            min_value=float(moisture_range['min']),
            max_value=float(moisture_range['max']),
            value=float((moisture_range['min'] + moisture_range['max']) / 2),
            help=moisture_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {moisture_range["min"]}-{moisture_range["max"]}{moisture_range["unit"]}</p>', unsafe_allow_html=True)
        
        # Rainfall
        rainfall_range = PARAMETER_RANGES['Rainfall']
        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=float(rainfall_range['min']),
            max_value=float(rainfall_range['max']),
            value=float((rainfall_range['min'] + rainfall_range['max']) / 2),
            help=rainfall_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {rainfall_range["min"]}-{rainfall_range["max"]}{rainfall_range["unit"]}</p>', unsafe_allow_html=True)
        
        # pH
        ph_range = PARAMETER_RANGES['pH']
        ph = st.slider(
            "pH",
            min_value=float(ph_range['min']),
            max_value=float(ph_range['max']),
            value=float((ph_range['min'] + ph_range['max']) / 2),
            help=ph_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {ph_range["min"]}-{ph_range["max"]}{ph_range["unit"]}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="subtitle-text">Nutrient Levels</p>', unsafe_allow_html=True)
        
        # Nitrogen
        nitrogen_range = PARAMETER_RANGES['Nitrogen']
        nitrogen = st.slider(
            "Nitrogen (kg/ha)",
            min_value=float(nitrogen_range['min']),
            max_value=float(nitrogen_range['max']),
            value=float((nitrogen_range['min'] + nitrogen_range['max']) / 2),
            help=nitrogen_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {nitrogen_range["min"]}-{nitrogen_range["max"]}{nitrogen_range["unit"]}</p>', unsafe_allow_html=True)
        
        # Phosphorous
        phosphorous_range = PARAMETER_RANGES['Phosphorous']
        phosphorous = st.slider(
            "Phosphorous (kg/ha)",
            min_value=float(phosphorous_range['min']),
            max_value=float(phosphorous_range['max']),
            value=float((phosphorous_range['min'] + phosphorous_range['max']) / 2),
            help=phosphorous_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {phosphorous_range["min"]}-{phosphorous_range["max"]}{phosphorous_range["unit"]}</p>', unsafe_allow_html=True)
        
        # Potassium
        potassium_range = PARAMETER_RANGES['Potassium']
        potassium = st.slider(
            "Potassium (kg/ha)",
            min_value=float(potassium_range['min']),
            max_value=float(potassium_range['max']),
            value=float((potassium_range['min'] + potassium_range['max']) / 2),
            help=potassium_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {potassium_range["min"]}-{potassium_range["max"]}{potassium_range["unit"]}</p>', unsafe_allow_html=True)
        
        # Carbon
        carbon_range = PARAMETER_RANGES['Carbon']
        carbon = st.slider(
            "Carbon (%)",
            min_value=float(carbon_range['min']),
            max_value=float(carbon_range['max']),
            value=float((carbon_range['min'] + carbon_range['max']) / 2),
            help=carbon_range['description']
        )
        st.markdown(f'<p class="range-info">Typical range: {carbon_range["min"]}-{carbon_range["max"]}{carbon_range["unit"]}</p>', unsafe_allow_html=True)
    
    # Create two columns for soil and crop selection
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<p class="subtitle-text">Soil Type</p>', unsafe_allow_html=True)
        soil_types = PARAMETER_RANGES['Soil']['options']
        soil = st.selectbox(
            "Select Soil Type",
            soil_types,
            help=PARAMETER_RANGES['Soil']['description']
        )
    
    with col4:
        st.markdown('<p class="subtitle-text">Crop Type</p>', unsafe_allow_html=True)
        crop_types = PARAMETER_RANGES['Crop']['options']
        crop = st.selectbox(
            "Select Crop Type",
            crop_types,
            help=PARAMETER_RANGES['Crop']['description']
        )
    
    # Add some spacing
    st.write("")
    
    if st.button("Get Fertilizer Recommendation"):
        # Prepare input data
        input_data = [temperature, moisture, rainfall, ph, nitrogen, phosphorous, potassium, carbon, soil, crop]
        input_df = pd.DataFrame([input_data], columns=['Temperature', 'Moisture', 'Rainfall', 'PH', 
                                                     'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon', 
                                                     'Soil', 'Crop'])
        
        # Encode categorical variables
        input_df['Soil'] = soil_encoder.transform(input_df['Soil'])
        input_df['Crop'] = crop_encoder.transform(input_df['Crop'])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_fertilizers = model.classes_[top_3_idx]
        top_3_probs = probabilities[top_3_idx]
        
        # Create three columns for results
        res1, res2, res3 = st.columns(3)
        
        with res1:
            st.markdown('<p class="subtitle-text">Top Recommendation</p>', unsafe_allow_html=True)
            st.success(f"**{prediction}**")
            confidence = max(probabilities)
            st.metric("Confidence", f"{confidence:.1%}")
        
        with res2:
            st.markdown('<p class="subtitle-text">Alternative Options</p>', unsafe_allow_html=True)
            for fert, prob in zip(top_3_fertilizers[1:], top_3_probs[1:]):
                st.info(f"{fert}: {prob:.1%}")
        
        with res3:
            st.markdown('<p class="subtitle-text">Confidence Analysis</p>', unsafe_allow_html=True)
            if confidence < 0.8:
                reasons = analyze_low_confidence(input_data, confidence)
                st.warning("Low confidence reasons:")
                for reason in reasons:
                    st.write(f"• {reason}")
            else:
                st.success("High confidence prediction! The input parameters are within typical ranges.")
        
        # Create visualization
        st.markdown('<p class="subtitle-text">Prediction Confidence Visualization</p>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71' if p >= 0.8 else '#e74c3c' for p in top_3_probs]
        bars = ax.bar(top_3_fertilizers, top_3_probs, color=colors)
        ax.set_ylabel('Confidence')
        ax.set_title('Top 3 Fertilizer Recommendations')
        plt.xticks(rotation=45)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom')
        
        st.pyplot(fig)

if __name__ == "__main__":
    main() 