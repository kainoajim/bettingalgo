from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the models
home_model = joblib.load('models/home_score_model.pkl')
away_model = joblib.load('models/away_score_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    input_data = request.json['data']
    
    # Convert input data to DataFrame (assuming the input is in JSON format)
    input_df = pd.DataFrame(input_data)
    
    # Make predictions
    home_pred = home_model.predict(input_df)
    away_pred = away_model.predict(input_df)
    
    return jsonify({
        'home_score_prediction': home_pred.tolist(),
        'away_score_prediction': away_pred.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
