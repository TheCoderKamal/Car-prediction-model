document.getElementById('car-form').addEventListener('submit', function (e) {
    e.preventDefault();

    // Collect values from the form
    const presentPrice = document.getElementById('present_price').value;
    const kmsDriven = document.getElementById('kms_driven').value;
    const owner = document.getElementById('owner').value;
    const carAge = document.getElementById('car_age').value;
    const fuelType = document.getElementById('fuel_type').value;
    const fuelTypeDiesel = fuelType === '1' ? 1 : 0;  // Diesel = 1, Petrol = 0
    const fuelTypePetrol = fuelType === '2' ? 1 : 0;  // Petrol = 1, Diesel = 0
    const sellerType = document.getElementById('seller_type').value === '1' ? 1 : 0; // 1 for Individual, 0 for Dealer
    const transmission = document.getElementById('transmission').value === '1' ? 1 : 0; // 1 for Manual, 0 for Automatic

    // Input validation
    if (!presentPrice || !kmsDriven || !owner || !carAge) {
        alert("Please fill in all fields!");
        return;
    }

    // Create JSON data
    const formData = {
        Present_Price: presentPrice,
        Kms_Driven: kmsDriven,
        Owner: owner,
        Car_Age: carAge,
        Fuel_Type_Diesel: fuelTypeDiesel,
        Fuel_Type_Petrol: fuelTypePetrol,
        Seller_Type_Individual: sellerType,
        Transmission_Manual: transmission
    };

    // Send POST request to Flask server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        if (data.Predicted_Price) {
            document.getElementById('result').classList.remove('hidden');
            document.getElementById('price-result').textContent = `₹${data.Predicted_Price}`;
        } else {
            document.getElementById('result').classList.remove('hidden');
            document.getElementById('price-result').textContent = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('price-result').textContent = `Error: Unable to fetch result`;
    });
});
