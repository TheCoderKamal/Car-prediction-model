from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load dataset and preprocess it
def train_model():
    # Load the dataset
    car_dataset = pd.read_csv('car_data.csv')

    # Encoding the categorical columns
    label_encoder = LabelEncoder()

    car_dataset['Fuel_Type'] = label_encoder.fit_transform(car_dataset['Fuel_Type'])
    car_dataset['Seller_Type'] = label_encoder.fit_transform(car_dataset['Seller_Type'])
    car_dataset['Transmission'] = label_encoder.fit_transform(car_dataset['Transmission'])

    # Convert Fuel_Type into separate columns for Diesel and Petrol (One-hot Encoding)
    car_dataset['Fuel_Type_Diesel'] = car_dataset['Fuel_Type'].apply(lambda x: 1 if x == 1 else 0)  # Diesel = 1
    car_dataset['Fuel_Type_Petrol'] = car_dataset['Fuel_Type'].apply(lambda x: 1 if x == 2 else 0)  # Petrol = 1

    # Drop 'Car_Name' and 'Fuel_Type' as they're not useful for prediction
    X = car_dataset.drop(['Car_Name', 'Selling_Price', 'Fuel_Type'], axis=1)
    Y = car_dataset['Selling_Price']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

    # Initialize and train the model (using Linear Regression)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Save the trained model using pickle
    with open('car_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model training complete and saved.")

from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load dataset and preprocess it
def train_model():
    # Load the dataset
    car_dataset = pd.read_csv('car_data.csv')

    # Encoding the categorical columns
    label_encoder = LabelEncoder()

    car_dataset['Fuel_Type'] = label_encoder.fit_transform(car_dataset['Fuel_Type'])
    car_dataset['Seller_Type'] = label_encoder.fit_transform(car_dataset['Seller_Type'])
    car_dataset['Transmission'] = label_encoder.fit_transform(car_dataset['Transmission'])

    # Convert Fuel_Type into separate columns for Diesel and Petrol (One-hot Encoding)
    car_dataset['Fuel_Type_Diesel'] = car_dataset['Fuel_Type'].apply(lambda x: 1 if x == 1 else 0)  # Diesel = 1
    car_dataset['Fuel_Type_Petrol'] = car_dataset['Fuel_Type'].apply(lambda x: 1 if x == 2 else 0)  # Petrol = 1

    # Drop 'Car_Name' and 'Fuel_Type' as they're not useful for prediction
    X = car_dataset.drop(['Car_Name', 'Selling_Price', 'Fuel_Type'], axis=1)
    Y = car_dataset['Selling_Price']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

    # Initialize and train the model (using Linear Regression)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Save the trained model using pickle
    with open('car_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model training complete and saved.")

# Route to display a welcome message
@app.route('/')
def home():
    return "<h1>Welcome to Car Price Prediction API</h1>"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        with open('car_price_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Get input data from the POST request (user inputs in JSON format)
        data = request.get_json()

        # Extract and preprocess input values
        present_price = data.get('Present_Price', 0)
        kms_driven = data.get('Kms_Driven', 0)
        owner = data.get('Owner', 0)
        car_age = data.get('Car_Age', 0)
        fuel_type_diesel = data.get('Fuel_Type_Diesel', 0)
        fuel_type_petrol = data.get('Fuel_Type_Petrol', 0)
        seller_type_individual = data.get('Seller_Type_Individual', 0)
        transmission_manual = data.get('Transmission_Manual', 0)

        # Convert the data into the format required by the model (a numpy array)
        input_features = np.array([[
            present_price,
            kms_driven,
            owner,
            car_age,
            fuel_type_diesel,
            fuel_type_petrol,
            seller_type_individual,
            transmission_manual
        ]])

        # Use the model to predict the car price
        prediction = model.predict(input_features)

        # Return the predicted price as a JSON response
        return jsonify({"Predicted_Price": round(prediction[0], 2)})

    except Exception as e:
        # If there is an error (like invalid input), return an error message
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Train the model (only once, initially)
    train_model()

    # Run the Flask app
    app.run(debug=True)
