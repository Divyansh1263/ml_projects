
# House Price Prediction

## Overview
This project predicts house prices in California based on various features like location, number of rooms, and population.

## Files
1.  **`house_price_analysis.ipynb` (Jupyter Notebook)**
    *   **Purpose**: Exploratory Data Analysis (EDA) and initial experimentation.
    *   **Content**:
        *   Loads the California Housing dataset.
        *   Visualizes data distributions and correlations.
        *   Trains a **Linear Regression** baseline model.
        *   Result: RMSE ~0.74, R2 ~0.57.

2.  **`house_price_rf.py` (Python Script)**
    *   **Purpose**: Model Improvement using a more advanced algorithm.
    *   **Content**:
        *   Re-loads and scales the data.
        *   Trains a **Random Forest Regressor** (a powerful ensemble method).
        *   Compares it against the Linear Regression baseline.
        *   **Key Result**: Random Forest achieves significantly better accuracy (R2 ~0.80).
        *   **Feature Importance**: Generates a plot showing which features (e.g., Median Income) matter most for price.

## How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the analysis script**:
    ```bash
    python house_price_rf.py
    ```
