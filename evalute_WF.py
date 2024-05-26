import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('github_repositories_detailed.csv')

# Select features and target variable, omitting 'commits_count'
features = ['forks_count', 'watchers_count', 'open_issues_count', 'network_count', 'subscribers_count']
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and split into training and testing sets.")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Print R-squared scores
print(f"Linear Regression R-squared: {r2_lr}")
print(f"Random Forest Regressor R-squared: {r2_rf}")

# Identify the best model
best_model = 'Linear Regression' if r2_lr > r2_rf else 'Random Forest Regressor'
print(f"Best model: {best_model}")
