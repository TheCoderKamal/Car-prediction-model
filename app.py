from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (this assumes the model was trained and saved in 'car_price_model.pkl')
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')  # Home route to display the HTML page
def home():
    return render_template('index.html')  # This will render the index.html file from the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request (user inputs in JSON format)
        data = request.get_json()

        # Extract values from the JSON data
        present_price = float(data.get('Present_Price', 0))
        kms_driven = int(data.get('Kms_Driven', 0))
        owner = int(data.get('Owner', 0))
        car_age = int(data.get('Car_Age', 0))
        fuel_type_diesel = int(data.get('Fuel_Type_Diesel', 0))
        fuel_type_petrol = int(data.get('Fuel_Type_Petrol', 0))
        seller_type_individual = int(data.get('Seller_Type_Individual', 0))
        transmission_manual = int(data.get('Transmission_Manual', 0))

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
    # Run the Flask app
    app.run(debug=True)
