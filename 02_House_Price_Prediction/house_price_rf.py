
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
print("Loading California Housing Data...")
housing_data = fetch_california_housing()
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['Target'] = housing_data.target
X = df.drop('Target', axis=1)
y = df['Target']

# 2. Preprocessing
print("Preprocessing (Splitting and Scaling)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Linear Regression (Baseline)
print("Training Linear Regression Baseline...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)
print(f"Linear Regression RMSE: {lr_rmse:.4f}")
print(f"Linear Regression R2: {lr_r2:.4f}")

# 4. Train Random Forest (Improvement)
print("\nTraining Random Forest Regressor (n_estimators=100)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest R2: {rf_r2:.4f}")

# 5. Comparison
print("\n--- Improvement Report ---")
print(f"RMSE Reduction: {lr_rmse - rf_rmse:.4f}")
print(f"R2 Score Increase: {rf_r2 - lr_r2:.4f}")

# 6. Feature Importance Visualization
print("\nGenerating Feature Importance Plot...")
importances = rf_model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis', hue='Feature', legend=False)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
print("Close the plot window to finish script execution if running interactively.")
plt.show()
