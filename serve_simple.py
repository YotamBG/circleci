from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model_export')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()
        images = np.array(input_data['images'])

        # Perform prediction
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Return predictions as JSON
        return jsonify({"predictions": predicted_classes.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
