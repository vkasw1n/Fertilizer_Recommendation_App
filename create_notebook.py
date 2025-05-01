import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Fertilizer Recommendation Analysis\n",
                "\n",
                "This notebook analyzes the fertilizer recommendation dataset, evaluates the performance of a Random Forest model, and visualizes key insights. We'll cover:\n",
                "- Loading and preparing the data\n",
                "- Training the model and evaluating its accuracy\n",
                "- Visualizing feature importance, confusion matrix, and fertilizer distribution"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Import Libraries and Load Data\n",
                "\n",
                "In this cell, we import the necessary Python libraries for data processing, model training, and visualization. We also load the fertilizer recommendation dataset from the CSV file."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.preprocessing import LabelEncoder\n",
                "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Load the dataset\n",
                "df = pd.read_csv('data/fertilizer_recommendation_dataset.csv')\n",
                "\n",
                "# Display the first few rows to understand the data\n",
                "print(f\"Dataset shape: {df.shape}\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Prepare Data and Train the Model\n",
                "\n",
                "Here, we prepare the features and target variable for training. Categorical variables (Soil and Crop) are encoded using LabelEncoder. We then split the data into training and testing sets and train a Random Forest Classifier."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare features and target\n",
                "X = df[['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon', 'Soil', 'Crop']].copy()  # Create a copy to avoid warnings\n",
                "y = df['Fertilizer']\n",
                "\n",
                "# Encode categorical variables\n",
                "le_soil = LabelEncoder()\n",
                "le_crop = LabelEncoder()\n",
                "X.loc[:, 'Soil'] = le_soil.fit_transform(X['Soil'])  # Use .loc to avoid warnings\n",
                "X.loc[:, 'Crop'] = le_crop.fit_transform(X['Crop'])  # Use .loc to avoid warnings\n",
                "\n",
                "# Split the data into training and testing sets\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "print(f\"Training set size: {X_train.shape[0]} samples\")\n",
                "print(f\"Testing set size: {X_test.shape[0]} samples\")\n",
                "\n",
                "# Train the Random Forest model\n",
                "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                "rf_model.fit(X_train, y_train)\n",
                "\n",
                "# Make predictions on the test set\n",
                "y_pred = rf_model.predict(X_test)\n",
                "\n",
                "# Calculate and display the accuracy\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "print(f'Model Accuracy: {accuracy:.2%}')\n",
                "\n",
                "# Display the classification report for detailed metrics\n",
                "print('\\nClassification Report:')\n",
                "print(classification_report(y_test, y_pred))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Visualize Feature Importance\n",
                "\n",
                "This cell creates a bar plot to show the importance of each feature in predicting the fertilizer type. Feature importance helps us understand which factors (e.g., Nitrogen, pH) most influence the model's predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a DataFrame for feature importance\n",
                "feature_importance = pd.DataFrame({\n",
                "    'feature': X.columns,\n",
                "    'importance': rf_model.feature_importances_\n",
                "})\n",
                "feature_importance = feature_importance.sort_values('importance', ascending=False)\n",
                "\n",
                "# Plot feature importance\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(x='importance', y='feature', data=feature_importance)\n",
                "plt.title('Feature Importance in Fertilizer Prediction')\n",
                "plt.xlabel('Importance')\n",
                "plt.ylabel('Feature')\n",
                "plt.savefig('feature_importance.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Visualize Confusion Matrix\n",
                "\n",
                "This cell generates a heatmap of the confusion matrix, which shows how well the model predicts each fertilizer type. The diagonal values represent correct predictions, while off-diagonal values indicate misclassifications."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compute the confusion matrix\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "\n",
                "# Plot the confusion matrix as a heatmap\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
                "            xticklabels=rf_model.classes_, \n",
                "            yticklabels=rf_model.classes_)\n",
                "plt.title('Confusion Matrix')\n",
                "plt.xlabel('Predicted')\n",
                "plt.ylabel('Actual')\n",
                "plt.savefig('confusion_matrix.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Visualize Fertilizer Recommendation Distribution\n",
                "\n",
                "In this cell, we create a bar plot to show the distribution of fertilizer recommendations in the dataset. This helps us understand how frequently each fertilizer type is recommended."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot the distribution of fertilizer recommendations\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.countplot(x='Fertilizer', data=df)\n",
                "plt.title('Distribution of Fertilizer Recommendations')\n",
                "plt.xlabel('Fertilizer')\n",
                "plt.ylabel('Count')\n",
                "plt.xticks(rotation=45)\n",
                "plt.savefig('fertilizer_distribution.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Summary\n",
                "\n",
                "1. **Feature Importance Plot**: Shows which features most influence fertilizer predictions.\n",
                "2. **Confusion Matrix**: Displays prediction accuracy across different fertilizer types.\n",
                "3. **Fertilizer Distribution**: Shows the frequency of each fertilizer recommendation in the dataset.\n",
                "\n",
                "The model's performance metrics (accuracy, precision, recall) are now based on a proper train-test split, which provides a more realistic evaluation of the model's performance."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('fertilizer_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1) 