import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib

# Load the data
df = pd.read_csv('github_repositories_detailed.csv')

# Select features and target variable, excluding 'watchers_count'
features = ['forks_count', 'open_issues_count', 'network_count', 'subscribers_count']
target = 'stargazers_count'

# Check for missing values
print("Missing values in each feature:")
print(df[features].isnull().sum())

# Drop rows with missing values
df = df.dropna(subset=features)

# Ensure DataFrame is not empty
if df.empty:
    raise ValueError("No data available after dropping rows with missing values.")

X = df[features]
y = df[target]

# Ensure there are enough samples
if len(X) < 2:
    raise ValueError("Not enough data to split into training and testing sets.")

# Visualize the relationship between features and the target
pd.plotting.scatter_matrix(df[features + [target]], figsize=(12, 8))
plt.savefig('scatter_matrix.png')
plt.show()

# Compute correlation matrix
corr_matrix = df[features + [target]].corr()
print(corr_matrix)

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('correlation_matrix.png')
plt.show()

# Visualize the distribution of features and target
df[features + [target]].hist(bins=30, figsize=(15, 10))
plt.savefig('feature_distributions.png')
plt.show()

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Normalize/Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
cross_val_lr = cross_val_score(lr_model, X_scaled, y, cv=5)

# Ridge Regression Model
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
cross_val_ridge = cross_val_score(ridge_model, X_scaled, y, cv=5)

# Random Forest Regressor Model with Grid Search
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
cross_val_rf = cross_val_score(best_rf_model, X_scaled, y, cv=5)

# Gradient Boosting Regressor Model with Grid Search
param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
gbr_model = GradientBoostingRegressor(random_state=42)
grid_search_gbr = GridSearchCV(gbr_model, param_grid_gbr, cv=5, scoring='r2', n_jobs=-1)
grid_search_gbr.fit(X_train, y_train)
best_gbr_model = grid_search_gbr.best_estimator_
y_pred_gbr = best_gbr_model.predict(X_test)
r2_gbr = r2_score(y_test, y_pred_gbr)
cross_val_gbr = cross_val_score(best_gbr_model, X_scaled, y, cv=5)

# Print R-squared scores
print(f"Linear Regression R-squared: {r2_lr}")
print(f"Linear Regression Cross-Validation R-squared: {cross_val_lr.mean()}")
print(f"Ridge Regression R-squared: {r2_ridge}")
print(f"Ridge Regression Cross-Validation R-squared: {cross_val_ridge.mean()}")
print(f"Random Forest Regressor R-squared: {r2_rf}")
print(f"Random Forest Regressor Cross-Validation R-squared: {cross_val_rf.mean()}")
print(f"Gradient Boosting Regressor R-squared: {r2_gbr}")
print(f"Gradient Boosting Regressor Cross-Validation R-squared: {cross_val_gbr.mean()}")

# Identify the best model
if r2_lr > r2_ridge and r2_lr > r2_rf and r2_lr > r2_gbr:
    best_model = ('Linear Regression', lr_model)
elif r2_ridge > r2_lr and r2_ridge > r2_rf and r2_ridge > r2_gbr:
    best_model = ('Ridge Regression', ridge_model)
elif r2_rf > r2_lr and r2_rf > r2_ridge and r2_rf > r2_gbr:
    best_model = ('Random Forest Regressor', best_rf_model)
else:
    best_model = ('Gradient Boosting Regressor', best_gbr_model)

best_model_name, best_model_instance = best_model
print(f"Best model: {best_model_name}")

# Save the best model
joblib.dump(best_model_instance, 'best_model.pkl')
print(f"Best model is saved as best_model.pkl")
