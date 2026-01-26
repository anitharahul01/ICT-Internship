from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load YOUR trained model
print("🔄 Loading model...")
model = joblib.load("course_success_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")
print("✅ Model loaded! Ready for predictions.")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    ai_advice = None

    if request.method == "POST":
        try:
            # Get form inputs
            duration = float(request.form["duration"])
            lessons = int(request.form["lessons"])
            rating = float(request.form["rating"])
            price = request.form["price"]
            category = request.form["category"]

            # Create features EXACTLY like EDA + model.py
            is_free = 1 if price == "Free" else 0

            # Category encoding (MUST match your EDA LabelEncoder)
            category_map = {
                "General": 0,
                "Generative AI": 1,
                "AI Agents": 2,
                "Programming": 3,
                "AI/ML": 4,
                "Data Science": 5,
            }
            cat_encoded = category_map.get(category, 0)

            # Enrollments estimate (for derived features)
            enroll_estimate = 1000  # Default for prediction

            # Create ALL features from model
            input_data = pd.DataFrame(
                {
                    "duration_hours": [duration],
                    "lessons": [lessons],
                    "rating": [rating],
                    "enrollments": [enroll_estimate],
                    "category_encoded": [cat_encoded],
                    "is_free": [is_free],
                    "duration_per_lesson": [duration / (lessons + 1)],
                    "rating_enroll_ratio": [rating * np.log1p(enroll_estimate)],
                }
            )[
                model_features
            ]  # Exact feature order

            # Predict
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            prediction = "🎉 HIGH SUCCESS" if pred == 1 else "⚠️ LOW SUCCESS"
            probability = f"{prob:.1%} success chance"

            # GenAI Advisor
            success_tips = [
                "✅ Make it FREE - 2x success rate",
                "✅ Target 1-3 hour duration (sweet spot)",
                "✅ Aim for General/Data Science category",
                "✅ Create 10-20 lessons with quizzes",
                "✅ Optimize title with trending keywords",
            ]

            if pred == 0:
                ai_advice = "🚀 QUICK WINS TO BOOST SUCCESS:\n\n" + "\n".join(
                    success_tips[:4]
                )
            else:
                ai_advice = "🎯 EXCELLENT! To MAXIMIZE:\n\n" + "\n".join(
                    success_tips[2:]
                )

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template(
        "index.html", prediction=prediction, probability=probability, advice=ai_advice
    )


if __name__ == "__main__":
    # Development only
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    # Production (gunicorn)
    application = app


if __name__ == "__main__":
    print("🚀 Launching Course Success Predictor")
    print("🌐 Visit: http://localhost:5000")
    app.run(debug=True, port=5000)
