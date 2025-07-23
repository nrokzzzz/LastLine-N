from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("mailByNB.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Last Line - N"

@app.route('/predict-N', methods=['POST'])
def predict():
    data = request.get_json()

    # Check for 'body' in the input
    if 'body' not in data:
        return jsonify({"error": "Missing 'body' field in JSON"}), 400

    mail_body = data['body']

    # Get prediction and confidence
    prediction = model.predict([mail_body])[0]
    probabilities = model.predict_proba([mail_body])[0]
    class_index = np.argmax(probabilities)
    confidence = round(probabilities[class_index] * 100, 2)

    return jsonify({
        "prediction": prediction,
        "confidence": f"{confidence}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
