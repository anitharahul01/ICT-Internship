from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load
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

            # Create features
            is_free = 1 if price == "Free" else 0

            # Category encoding
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
            )[model_features]

            # Predict probability
            input_scaled = scaler.transform(input_data)
            prob = model.predict_proba(input_scaled)[0][
                1
            ]  # Probability of SUCCESS (class 1)

            # ✅ NEW: 3-TIER CLASSIFICATION (>0.85 = High, >0.65 = Medium, <0.65 = Low)
            if prob >= 0.85:
                prediction = "🎉 HIGH SUCCESS"
                level = "success-glow bg-success bg-opacity-10"
            elif prob >= 0.65:
                prediction = "⚡ MEDIUM SUCCESS"
                level = "info-glow bg-info bg-opacity-10"
            else:
                prediction = "⚠️ LOW SUCCESS"
                level = "warning-glow bg-warning bg-opacity-10"

            probability = f"{prob:.1%} success probability"

            # 🎯 ENHANCED AI ADVISOR (3 levels)
            if "HIGH" in prediction:
                ai_advice = """🎯 EXCELLENT COURSE! 🚀

✅ You're in TOP 15% of courses!

💎 TO DOMINATE MARKET:
• Add certificates/badges
• Live Q&A sessions
• Video testimonials
• Partner promotions

🎖️ Success Score: ELITE"""

            elif "MEDIUM" in prediction:
                ai_advice = """⚡ GOOD COURSE - MEDIUM POTENTIAL!

🔥 QUICK WINS (Boost to HIGH):
• Make FREE (if paid)
• Shorten to 1-3 hours
• Add 2-3 interactive quizzes
• Optimize thumbnail/title

🎯 Target: 85%+ success"""

            else:  # LOW
                ai_advice = """⚠️ LOW SUCCESS - NEEDS WORK!

🚀 CRITICAL FIXES:
• Make FREE (98% top courses)
• 1-3 hour duration only
• General/Data Science category
• 12-20 lessons max
• Rating target: 4.5+

📈 Follow = 3X success boost"""

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"
            probability = None
            ai_advice = None

    return render_template(
        "index.html", prediction=prediction, probability=probability, advice=ai_advice
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Launching Course Success Predictor")
    print(f"🌐 Visit: http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
