from flask import Flask, request, jsonify
import pickle
from joblib import load
import numpy as np
from deadLinePrediction import extract_deadline_date

app = Flask(__name__)

# Load models
with open("mailByNB.pkl", "rb") as f:
    model = pickle.load(f) 


# for modelNaive
vectorizer = load('vectorizer.pkl')

# spam filter
modelNaive = load('modelNaive.pkl')         

@app.route('/')
def home():
    return "Last Line - N"

@app.route('/predict-N', methods=['POST'])
def predict():
    data = request.get_json()

    if 'body' not in data:
        return jsonify({"error": "Missing 'body' field in JSON"}), 400

    mail_body = data['body']  
    
    #Use full classification model 
    final_prediction = model.predict([mail_body])[0]
    probabilities = model.predict_proba([mail_body])[0]
    class_index = np.argmax(probabilities)
    confidence = round(probabilities[class_index] * 100, 2)

    #Use Naive Bayes spam filter
    mail_text_vectorized = vectorizer.transform([mail_body])
    spam_prediction = modelNaive.predict(mail_text_vectorized)[0]

    if final_prediction == 'others':
        # other mail's block

        return jsonify({
            "prediction": final_prediction,
            "confidence": f"{confidence}%"
        })
    else:
        spam_prediction = modelNaive.predict(mail_text_vectorized)[0]
        if spam_prediction == 0:
            return jsonify({
                "prediction": final_prediction,
                "confidence": f"{confidence}%",
                "spam":"no_spam",
                "deadline":extract_deadline_date(mail_body)
            })
        else:
            return jsonify({
                "prediction": final_prediction,
                "confidence": f"{confidence}%",
                "spam":"spam",
                "deadline":extract_deadline_date(mail_body)
            })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
