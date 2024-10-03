import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error, r2_score

# Load your prepared dataset and drop NaN rows
df = pd.read_csv('data/nba_model_training_data.csv')
df = df.dropna()

spreads_totals = df[['gameID', 'spread', 'total']]

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, gameID_test):
    # Train the model on the home team score
    model.fit(X_train, y_home_train)
    home_predictions = model.predict(X_test)

    # Train the model on the away team score
    model.fit(X_train, y_away_train)
    away_predictions = model.predict(X_test)

    # Create a DataFrame to compare predictions and actual values
    comparison_df = pd.DataFrame({
        'gameID': gameID_test,
        'Home_Predicted_Score': home_predictions,
        'Home_Actual_Score': y_home_test.values,
        'Away_Predicted_Score': away_predictions,
        'Away_Actual_Score': y_away_test.values
    })

    # Merge the spread and totals data back in for evaluation
    comparison_df = comparison_df.merge(spreads_totals, on='gameID', how='left')

    # Calculate winner accuracy
    comparison_df['Predicted_Winner'] = comparison_df.apply(
        lambda row: 'home' if row['Home_Predicted_Score'] > row['Away_Predicted_Score'] else 'away', axis=1)
    comparison_df['Actual_Winner'] = comparison_df.apply(
        lambda row: 'home' if row['Home_Actual_Score'] > row['Away_Actual_Score'] else 'away', axis=1)
    
    correct_predictions = (comparison_df['Predicted_Winner'] == comparison_df['Actual_Winner']).sum()
    total_predictions = len(comparison_df)
    accuracy = (correct_predictions / total_predictions) * 100

    # Spread calculation:
    comparison_df['Predicted_Spread'] = comparison_df['Away_Predicted_Score'] - comparison_df['Home_Predicted_Score']
    comparison_df['Actual_Spread'] = comparison_df['Away_Actual_Score'] - comparison_df['Home_Actual_Score']

    correct_spreads = comparison_df.apply(
        lambda row: (
            (row['Actual_Spread'] > row['spread'] and row['Predicted_Spread'] > row['spread']) or 
            (row['Actual_Spread'] < row['spread'] and row['Predicted_Spread'] < row['spread'])
        ), axis=1
    ).sum()

    # Total points calculation:
    comparison_df['Predicted_Total'] = comparison_df['Home_Predicted_Score'] + comparison_df['Away_Predicted_Score']
    comparison_df['Actual_Total'] = comparison_df['Home_Actual_Score'] + comparison_df['Away_Actual_Score']

    correct_totals = comparison_df.apply(
        lambda row: (
            (row['Actual_Total'] > row['total'] and row['Predicted_Total'] > row['total']) or
            (row['Actual_Total'] < row['total'] and row['Predicted_Total'] < row['total'])
        ), axis=1
    ).sum()

    # Calculate percentage accuracy for spread and totals
    spread_accuracy = (correct_spreads / len(comparison_df)) * 100
    totals_accuracy = (correct_totals / len(comparison_df)) * 100

    return accuracy, spread_accuracy, totals_accuracy

# Define the models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SGD Classifier": SGDClassifier(loss='modified_huber', random_state=42),
    "Linear SVC": LinearSVC(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "LGBM Classifier": LGBMClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

# Create the stacking classifier with the models
stacking_classifier = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('sgd', SGDClassifier(loss='modified_huber', random_state=42)),
        ('svc', LinearSVC(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)

# Add Stacking Classifier to the models dictionary
models["Stacking Classifier"] = stacking_classifier

# Define the features (X) and the target (y)
X = df.drop(columns=['Home_Score', 'Away_Score', 'spread', 'total'])
y_home = df['Home_Score']
y_away = df['Away_Score']

# Split the data into training and testing sets
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Retain the gameID for the test set
gameID_test = X_test['gameID']

# Drop the gameID column from X
X_train = X_train.drop(columns=['gameID'])
X_test = X_test.drop(columns=['gameID'])

# Initialize an empty list to store results
results = []

# Loop over the models and evaluate each one
for model_name, model in models.items():
    accuracy, spread_accuracy, totals_accuracy = train_and_evaluate_model(
        model, X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, gameID_test
    )
    print(f"{model_name} - Win Prediction Accuracy: {accuracy:.2f}%, Spread Accuracy: {spread_accuracy:.2f}%, Totals Accuracy: {totals_accuracy:.2f}%")
    results.append([model_name, accuracy, spread_accuracy, totals_accuracy])

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Model', 'Win Accuracy (%)', 'Spread Accuracy (%)', 'Totals Accuracy (%)'])
results_df.to_csv('data/model_comparison_results.csv', index=False)

print("Model comparison results saved to 'data/model_comparison_results.csv'.")
