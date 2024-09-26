import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your prepared dataset
df = pd.read_csv('data/nba_model_training_data.csv')

# Define the features (X) and the target (y)
# In this case, we're predicting both Home_Score and Away_Score, so we'll create a model for each
X = df.drop(columns=['Home_Score', 'Away_Score'])  # Drop unnecessary columns
y_home = df['Home_Score']  # Home team score target
y_away = df['Away_Score']  # Away team score target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Initialize the model (RandomForestRegressor in this example)
model_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_away = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data (for home team scores)
model_home.fit(X_train, y_home_train)
model_away.fit(X_train, y_away_train)

# Make predictions on the test data
home_predictions = model_home.predict(X_test)
away_predictions = model_away.predict(X_test)

# Evaluate the model (using Mean Squared Error and R-squared)
home_mse = mean_squared_error(y_home_test, home_predictions)
away_mse = mean_squared_error(y_away_test, away_predictions)

home_r2 = r2_score(y_home_test, home_predictions)
away_r2 = r2_score(y_away_test, away_predictions)

print(f"Home Team Score Prediction - MSE: {home_mse:.2f}, R-squared: {home_r2:.2f}")
print(f"Away Team Score Prediction - MSE: {away_mse:.2f}, R-squared: {away_r2:.2f}")

# Create a DataFrame to compare predictions and actual values
comparison_df = pd.DataFrame({
    'Home_Predicted_Score': home_predictions,
    'Home_Actual_Score': y_home_test.values,
    'Away_Predicted_Score': away_predictions,
    'Away_Actual_Score': y_away_test.values
})

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

# Save the comparison DataFrame to a CSV file
comparison_df.to_csv('data/predicted_vs_actual_scores.csv', index=False)

print("Predicted vs Actual scores saved to 'data/predicted_vs_actual_scores.csv'.")