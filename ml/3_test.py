# Content moved from tests/test_serving.py

import requests
import json

def test_predict():
    # Start the Flask app in the background
    from multiprocessing import Process
    from serve_simple import app

    server = Process(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000})
    server.start()

    try:
        # Prepare sample input data
        sample_data = {
            "images": [[[0.0] * 28] * 28]  # A single 28x28 grayscale image with all zeros
        }

        # Send a POST request to the /predict endpoint
        response = requests.post(
            "http://0.0.0.0:5000/predict",
            data=json.dumps(sample_data),
            headers={"Content-Type": "application/json"},
        )

        # Assert the response is successful and contains predictions
        assert response.status_code == 200
        assert "predictions" in response.json()
        print("Test passed: /predict endpoint is working correctly.")
    finally:
        # Terminate the Flask app
        server.terminate()

if __name__ == "__main__":
    test_predict()
