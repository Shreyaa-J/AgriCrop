"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
models = {
    "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("gradient_boosting_model.pkl", "rb")),
    "Bayes Net": pickle.load(open("bayes_net_model.pkl", "rb"))
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

        # Predict using all models
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict([data])[0]

        # Render results for all models
        return render_template('result.html', predictions=predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models and their accuracies
models = {
    "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("gradient_boosting_model.pkl", "rb")),
    "Bayes Net": pickle.load(open("bayes_net_model.pkl", "rb"))
}
accuracies = pickle.load(open("model_accuracies.pkl", "rb"))  # Load model accuracies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

        # Predict using all models
        predictions = {name: model.predict([data])[0] for name, model in models.items()}

        # Combine predictions with accuracies and sort by accuracy
        results = [{"model": name, "accuracy": accuracies[name], "prediction": predictions[name]} 
                   for name in models]
        results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

        # Pass predictions and results to result.html
        return render_template('result.html', predictions=predictions, results=results)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
