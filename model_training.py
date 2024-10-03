import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib as plt
import joblib

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Load your prepared dataset
df = pd.read_csv('data/nba_model_training_data.csv')
df = df.dropna()

spreads_totals = df[['gameID', 'spread', 'total']]
# Define the features (X) and the target (y)
# We are predicting Home_Score and Away_Score, spread and totals are just for accuracy checks
X = df.drop(columns=['Home_Score', 'Away_Score', 'spread', 'total'])  # Remove unnecessary columns
y_home = df['Home_Score']  # Home team score target
y_away = df['Away_Score']  # Away team score target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Retain the gameID for the test set
gameID_test = X_test['gameID']

# Now drop the gameID column from X, so it's not part of the model input
X_train = X_train.drop(columns=['gameID'])
X_test = X_test.drop(columns=['gameID'])

# # Initialize the Random Forest model
# rf = RandomForestRegressor(random_state=42)

# # Perform Randomized Search with Cross-Validation
# random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
#                                    n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
# # Fit the model on training data
# random_search.fit(X_train, y_home_train)

# # Use the best model for prediction
# model_home = random_search.best_estimator_
# # Train on away target as well using best parameters
# model_away = RandomForestRegressor(**random_search.best_params_)
# model_away.fit(X_train, y_away_train)

# RandomForestRegressor - 63% moneyline, 50.08% spread, 52.21% totals
model_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_away = RandomForestRegressor(n_estimators=100, random_state=42)

# features = model_home.feature_importances_
# features = X_train.columns

# # Plot feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances)
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance for Home Score Prediction")
# plt.show()


# Linear Regression - 60.27% moneyline, 49.01% spread, 48.71% total
# model_home = LinearRegression()
# model_away = LinearRegression()

# SGDClassifier - 54.95% moneyline, 48.55% spread, 49.01% total
# model_home = SGDClassifier(loss='modified_huber', random_state=42)
# model_away = SGDClassifier(loss='modified_huber', random_state=42)

# Initialize the K-Nearest Neighbors model - 57.99% moneyline, 47.64% spread, 46.73% total
# model_home = KNeighborsRegressor(n_neighbors=5)  # Set n_neighbors (k) to 5
# model_away = KNeighborsRegressor(n_neighbors=5)

# Train the models on the training data
model_home.fit(X_train, y_home_train)
model_away.fit(X_train, y_away_train)

joblib.dump(model_home, 'models/home_score_model.pkl')
joblib.dump(model_away, 'models/away_score_model.pkl')

# Make predictions on the test data
home_predictions = model_home.predict(X_test)
away_predictions = model_away.predict(X_test)

# Evaluate the model
home_mse = mean_squared_error(y_home_test, home_predictions)
away_mse = mean_squared_error(y_away_test, away_predictions)
home_r2 = r2_score(y_home_test, home_predictions)
away_r2 = r2_score(y_away_test, away_predictions)

print(f"Home Team Score Prediction - MSE: {home_mse:.2f}, R-squared: {home_r2:.2f}")
print(f"Away Team Score Prediction - MSE: {away_mse:.2f}, R-squared: {away_r2:.2f}")

# Create a DataFrame to compare predictions and actual values
comparison_df = pd.DataFrame({
    'gameID': gameID_test,  # Bring back gameID for evaluation
    'Home_Predicted_Score': home_predictions,
    'Home_Actual_Score': y_home_test.values,
    'Away_Predicted_Score': away_predictions,
    'Away_Actual_Score': y_away_test.values
})

print("Columns in comparison_df:", comparison_df.columns)
print("Columns in spreads_totals:", spreads_totals.columns)

comparison_df = comparison_df.merge(spreads_totals, on='gameID', how='left')

# Add predicted and actual winners (based on predicted and actual scores)
comparison_df['Predicted_Winner'] = comparison_df.apply(
    lambda row: 'home' if row['Home_Predicted_Score'] > row['Away_Predicted_Score'] else 'away', axis=1)

comparison_df['Actual_Winner'] = comparison_df.apply(
    lambda row: 'home' if row['Home_Actual_Score'] > row['Away_Actual_Score'] else 'away', axis=1)

# Calculate percentage of correct winner predictions
correct_predictions = (comparison_df['Predicted_Winner'] == comparison_df['Actual_Winner']).sum()
total_predictions = len(comparison_df)
accuracy = (correct_predictions / total_predictions) * 100

print(f"Correct winner prediction percentage: {accuracy:.2f}%")

# Spread calculation:
comparison_df['Predicted_Spread'] = comparison_df['Away_Predicted_Score'] - comparison_df['Home_Predicted_Score']
comparison_df['Actual_Spread'] = comparison_df['Away_Actual_Score'] - comparison_df['Home_Actual_Score']

# Check if the predicted spread is on the same side as the actual result
correct_spreads = comparison_df.apply(
    lambda row: (
        (row['Actual_Spread'] > row['spread'] and row['Predicted_Spread'] > row['spread']) or 
        (row['Actual_Spread'] < row['spread'] and row['Predicted_Spread'] < row['spread'])
    ), axis=1
).sum()

# Total points calculation:
comparison_df['Predicted_Total'] = comparison_df['Home_Predicted_Score'] + comparison_df['Away_Predicted_Score']
comparison_df['Actual_Total'] = comparison_df['Home_Actual_Score'] + comparison_df['Away_Actual_Score']

# Check if the predicted total is on the same side as the actual result
correct_totals = comparison_df.apply(
    lambda row: (
        (row['Actual_Total'] > row['total'] and row['Predicted_Total'] > row['total']) or
        (row['Actual_Total'] < row['total'] and row['Predicted_Total'] < row['total'])
    ), axis=1
).sum()

# Calculate percentage accuracy for spread and totals
spread_accuracy = (correct_spreads / len(comparison_df)) * 100
totals_accuracy = (correct_totals / len(comparison_df)) * 100

print(f"Correct spread prediction percentage: {spread_accuracy:.2f}%")
print(f"Correct totals prediction percentage: {totals_accuracy:.2f}%")

# Save the comparison DataFrame to a CSV file
comparison_df.to_csv('data/predicted_vs_actual_spread_totals.csv', index=False)

print("Predicted vs Actual scores, spreads, and totals saved to 'data/predicted_vs_actual_spread_totals.csv'.")
