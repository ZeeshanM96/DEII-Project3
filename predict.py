import argparse
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Set up the polynomial features and scaler
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

# Load the best model
model_path = '/app/best_model.pkl'
model = joblib.load(model_path)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Predict GitHub stars.')
parser.add_argument('--repos', type=str, required=True, help='JSON string of repositories to predict stars for.')
args = parser.parse_args()

# Parse the JSON string
import json
repos = json.loads(args.repos)

# Function to fit poly and scaler on sample data
def fit_poly_scaler():
    # Sample data similar to the one used during training to fit poly and scaler
    sample_data = [
        [100, 50, 30, 20],
        [200, 80, 60, 40],
        [150, 70, 50, 30],
        [120, 60, 40, 25],
        [180, 90, 70, 50]
    ]
    sample_poly = poly.fit_transform(sample_data)
    scaler.fit(sample_poly)

# Fit the polynomial features and scaler
fit_poly_scaler()

# Process each repository and make predictions
for repo in repos:
    name = repo['name']
    commits = repo['commits']
    forks = repo['forks']
    watchers = repo['watchers']
    stars = repo['stars']

    # Preprocess the input data
    input_data = [[commits, forks, watchers, stars]]
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Print the prediction
    print(f'Repository: {name}, Predicted stars: {prediction[0]}')