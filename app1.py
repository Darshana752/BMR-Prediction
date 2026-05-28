import pickle
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model once at startup (not on every request)
MODEL_PATH = os.path.join('model', 'predictor.pickle')


def load_model():
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)


try:
    model = load_model()
    print("✅ Model loaded successfully!")
    print(f"   Classes : {model.classes_.tolist()}")
    print(f"   K value : {model.n_neighbors}")
except FileNotFoundError:
    model = None
    print("❌ Model file not found at:", MODEL_PATH)


def prediction(height, weight):
    """Run KNN prediction and return BMR condition string."""
    if model is None:
        raise FileNotFoundError(
            "Model not loaded. Ensure 'model/predictor.pickle' exists.")
    pred_value = model.predict([[height, weight]])
    return str(pred_value[0])  # e.g. "normal", "overweight", "underweight"


@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    error = None

    if request.method == 'POST':
        try:
            height = request.form.get('height', '').strip()
            weight = request.form.get('weight', '').strip()

            # Validate inputs
            if not height or not weight:
                raise ValueError("Both height and weight are required.")

            height = float(height)
            weight = float(weight)

            # Basic range validation (matches notebook data)
            if not (50 <= height <= 300):
                raise ValueError("Height must be between 50 cm and 300 cm.")
            if not (10 <= weight <= 300):
                raise ValueError("Weight must be between 10 kg and 300 kg.")

            print("-" * 30)
            print(f"  Height : {height} cm")
            print(f"  Weight : {weight} kg")

            pred = prediction(height, weight)
            print(f"  Result : {pred}")
            print("-" * 30)

        except ValueError as e:
            error = str(e)
        except FileNotFoundError as e:
            error = str(e)
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    return render_template("index1.html", pred=pred, error=error)


if __name__ == '__main__':
    app.run(debug=True)
