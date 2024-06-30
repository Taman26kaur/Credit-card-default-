from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('knn_model.pkl')

# Define the expected feature columns and interaction features
expected_columns = [
    'gender', 'education', 'marital_status', 'age', 'limit_balance',
    'repayment_status_april', 'repayment_status_may', 'repayment_status_june',
    'repayment_status_july', 'repayment_status_august', 'repayment_status_september',
    'bill_amount_april', 'bill_amount_may', 'bill_amount_june', 'bill_amount_july',
    'bill_amount_august', 'bill_amount_september', 'previous_payment_april',
    'previous_payment_may', 'previous_payment_june', 'previous_payment_july',
    'previous_payment_august', 'previous_payment_september'
]

# Define the selected features including new features
selected_features = [
    'limit_balance', 'age', 'bill_amount_april', 'bill_amount_may', 'bill_amount_june',
    'bill_amount_july', 'bill_amount_august', 'bill_amount_september', 'previous_payment_april',
    'previous_payment_may', 'previous_payment_june', 'previous_payment_july',
    'previous_payment_august', 'previous_payment_september', 'LIMIT_BAL_AGE',
    'BILL_TOTAL', 'PAY_TOTAL'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Transform the data into the correct format for prediction
    input_data = pd.DataFrame([data])
    
    # Convert data types
    input_data = input_data.astype({
        'gender': int,
        'education': int,
        'marital_status': int,
        'age': int,
        'limit_balance': float,
        'repayment_status_april': int,
        'repayment_status_may': int,
        'repayment_status_june': int,
        'repayment_status_july': int,
        'repayment_status_august': int,
        'repayment_status_september': int,
        'bill_amount_april': float,
        'bill_amount_may': float,
        'bill_amount_june': float,
        'bill_amount_july': float,
        'bill_amount_august': float,
        'bill_amount_september': float,
        'previous_payment_april': float,
        'previous_payment_may': float,
        'previous_payment_june': float,
        'previous_payment_july': float,
        'previous_payment_august': float,
        'previous_payment_september': float
    })
    
    # Feature engineering: create interaction features
    input_data['LIMIT_BAL_AGE'] = input_data['limit_balance'] * input_data['age']
    input_data['BILL_TOTAL'] = input_data[
        ['bill_amount_april', 'bill_amount_may', 'bill_amount_june',
         'bill_amount_july', 'bill_amount_august', 'bill_amount_september']
    ].sum(axis=1)
    input_data['PAY_TOTAL'] = input_data[
        ['previous_payment_april', 'previous_payment_may', 'previous_payment_june',
         'previous_payment_july', 'previous_payment_august', 'previous_payment_september']
    ].sum(axis=1)
    
    # Select and order the features
    input_data = input_data[selected_features]
    
    # Convert DataFrame to numpy array
    input_array = input_data.to_numpy()
    
    # Perform prediction
    prediction = model.predict(input_array)
    result = 'Defaulter' if prediction[0] == 1 else 'Not a Defaulter'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
