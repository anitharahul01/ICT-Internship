import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print("🚀 COMPREHENSIVE MODEL COMPARISON FOR COURSE SUCCESS PREDICTION")
print("=" * 70)

# 1. LOAD & PREPARE DATA
print("\n📊 1. LOADING DATA...")
df = pd.read_csv("processed_courses.csv")
print(f"✅ Loaded: {df.shape[0]:,} courses | Success rate: {df['success'].mean():.1%}")

# Features
features = [
    "duration_hours",
    "lessons",
    "rating",
    "enrollments",
    "category_encoded",
    "duration_per_lesson",
    "rating_enroll_ratio",
]
df["is_free"] = (df["price"] == "Free").astype(int)
features += ["is_free"]

X = df[features]
y = df["success"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")

# 2. DEFINE 8 MODELS WITH HYPERPARAMETERS
print("\n🤖 2. DEFINING 8 MODELS...")

models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]},
        "use_scale": True,
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5],
        },
        "use_scale": False,
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.2],
        },
        "use_scale": False,
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.2],
        },
        "use_scale": False,
    },
    "SVM": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "use_scale": True,
    },
    "KNN": {
        "model": KNeighborsClassifier(n_jobs=-1),
        "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        "use_scale": True,
    },
}

# 3. TRAIN & EVALUATE ALL MODELS
print("\n⚡ 3. TRAINING & COMPARING 8 MODELS...")
results = []

for name, config in models.items():
    print(f"\n🔄 Training {name}...")

    # Use scaled data
    X_tr = X_train_scaled if config["use_scale"] else X_train
    X_te = X_test_scaled if config["use_scale"] else X_test

    # Hyperparameter tuning
    grid = GridSearchCV(
        config["model"], config["params"], cv=5, scoring="f1", n_jobs=-1
    )
    grid.fit(X_tr, y_train)

    # Predictions
    y_pred = grid.predict(X_te)
    y_pred_proba = grid.predict_proba(X_te)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    results.append(
        {
            "Model": name,
            "F1-Score": f1,
            "AUC-ROC": auc,
            "Best_Params": grid.best_params_,
            "CV_Score": grid.best_score_,
            "Model": grid.best_estimator_,
        }
    )

    print(f"   ✅ F1: {f1:.3f} | AUC: {auc:.3f} | CV: {grid.best_score_:.3f}")

# 4. RESULTS DATAFRAME & BEST MODEL
print("\n🏆 4. FINAL RESULTS:")
results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
print(results_df.round(3))

# Best model
best_idx = results_df["F1-Score"].idxmax()
best_model = results[best_idx]["Model"]
best_name = results_df.loc[best_idx, "Model"]

print(f"\n🎉 CHAMPION: {best_name}")
print(f"   F1-Score: {results_df.loc[best_idx, 'F1-Score']:.3f}")
print(f"   AUC-ROC:  {results_df.loc[best_idx, 'AUC-ROC']:.3f}")

# 5. SAVE BEST MODEL
joblib.dump(best_model, "course_success_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "model_features.pkl")
joblib.dump(results_df, "model_comparison_results.pkl")

print("\n💾 SAVED FILES:")
print("   • course_success_model.pkl (BEST MODEL)")
print("   • scaler.pkl")
print("   • model_features.pkl")
print("   • model_comparison_results.pkl")

# 6. VISUALIZATIONS
plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# F1 Score Bar Chart
top5 = results_df.head(5)
axes[0, 0].barh(range(len(top5)), top5["F1-Score"], color="skyblue")
axes[0, 0].set_yticks(range(len(top5)))
axes[0, 0].set_yticklabels(top5["Model"])
axes[0, 0].set_title("🏆 Top 5 Models by F1-Score")
axes[0, 0].set_xlabel("F1-Score")

# AUC-ROC Bar Chart
axes[0, 1].barh(range(len(top5)), top5["AUC-ROC"], color="lightcoral")
axes[0, 1].set_yticks(range(len(top5)))
axes[0, 1].set_yticklabels(top5["Model"])
axes[0, 1].set_title("📈 Top 5 Models by AUC-ROC")
axes[0, 1].set_xlabel("AUC-ROC")

# Confusion Matrix for Best Model
cm = confusion_matrix(
    y_test,
    best_model.predict(X_test_scaled if models[best_name]["use_scale"] else X_test),
)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
axes[1, 0].set_title(f"🎯 {best_name} Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# Feature Importance (if tree-based)
if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.DataFrame(
        {"feature": features, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    sns.barplot(
        data=feat_imp.head(8),
        x="importance",
        y="feature",
        ax=axes[1, 1],
        palette="viridis",
    )
    axes[1, 1].set_title(f"🔍 {best_name} Feature Importance")

plt.tight_layout()
plt.savefig("model_comparison_results.png", dpi=300, bbox_inches="tight")
plt.show()

print("📊 Results saved: model_comparison_results.png")

print("\n🎉 MODEL TRAINING COMPLETE!")

print("   course_success_model.pkl")
print("   scaler.pkl")
print("   model_features.pkl")
print("   feature_importance.png")
