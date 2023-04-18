from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return 'Hello, world!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the POST request
    review = request.form['review']

    # Use the machine learning model to predict the sentiment
    sentiment = model.predict([review])[0]

    # Return the sentiment as a JSON object
    response = {
        'sentiment': sentiment
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

