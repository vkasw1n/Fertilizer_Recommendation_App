# Fertilizer Recommendation System

A machine learning-based system that recommends the most suitable fertilizer for different crops based on soil parameters and environmental conditions.

## Project Structure

```
fertilizer_Prediction/
├── app.py                  # Streamlit web application
├── predict_fertilizer.py   # Core prediction model
├── fertilizer_analysis.ipynb  # Data analysis and visualization
├── requirements.txt        # Python dependencies
├── data/
│   └── fertilizer_recommendation_dataset.csv  # Dataset
└── README.md              # This file
```

## Features

- Supports 32 different crops including grains, legumes, cash crops, and fruits
- Considers multiple soil parameters:
  - Temperature
  - Moisture
  - Rainfall
  - pH
  - Nitrogen
  - Phosphorous
  - Potassium
  - Carbon
  - Soil Type
- Provides fertilizer recommendations with confidence scores
- Includes analysis for low-confidence predictions
- User-friendly web interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fertilizer_Prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

To run the Streamlit web interface:
```bash
streamlit run app.py
```

This will start a local web server and open the application in your default browser.

### Using the Command Line Interface

To use the command-line version:
```bash
python predict_fertilizer.py
```

## File Descriptions

### `app.py`
- Streamlit web application
- Provides a user-friendly interface for fertilizer prediction
- Features:
  - Interactive sliders for parameter input
  - Dropdown menus for crop and soil selection
  - Visual feedback and recommendations
  - Confidence analysis

### `predict_fertilizer.py`
- Core prediction model implementation
- Features:
  - Model training and loading
  - Data preprocessing
  - Prediction logic
  - Confidence analysis
  - Command-line interface

### `fertilizer_analysis.ipynb`
- Jupyter notebook for data analysis
- Contains:
  - Data exploration
  - Model development
  - Visualizations
  - Performance analysis
- Useful for:
  - Understanding the dataset
  - Model experimentation
  - Creating visualizations
  - Documentation

### `requirements.txt`
- Lists all Python package dependencies
- Includes:
  - scikit-learn
  - pandas
  - numpy
  - streamlit
  - matplotlib
  - seaborn
  - joblib

## Dataset

The dataset (`data/fertilizer_recommendation_dataset.csv`) contains:
- 32 different crops
- 100 samples per crop
- 12 features including soil parameters and environmental conditions
- Fertilizer recommendations and remarks

## Model Details

- Algorithm: Random Forest Classifier
- Features:
  - 300 decision trees
  - Maximum depth of 20
  - Balanced class weights
  - Cross-validation with 5 folds
- Performance metrics:
  - Cross-validation accuracy
  - Classification report
  - Feature importance analysis

## Contributing

Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your chosen license]

## Contact

[Your contact information] 