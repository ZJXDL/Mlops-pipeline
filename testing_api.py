import requests

# 1. Define the API endpoint URL
# Note: If your Docker container is mapped to a different port, change 8000.
url = "http://127.0.0.1:8000/predict"

# 2. Create the data payload (the exact same JSON structure you used in the UI)
sensor_data = {
    "engine_rpm": 6500,
    "engine_temp": 115,
    "vibration_level": 5.8
}

# 3. Send the POST request to the API
print("Sending data to the API...")
response = requests.post(url, json=sensor_data)

# 4. Check the results
if response.status_code == 200:
    print("Success! Here is the model's response:")
    # .json() converts the API's JSON response back into a Python dictionary
    prediction_result = response.json()
    print(prediction_result)
    
    # You can now use the specific values in your script
    if prediction_result["prediction"] == 1:
        print("WARNING: Maintenance required immediately!")
else:
    print(f"Failed with status code: {response.status_code}")
    print(response.text)