import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os

DATA_PATH = "D:\PROJECTS\Emotion Detector\Emotion_Detection\data\\features\\features.csv"
MODEL_DIR = "D:\PROJECTS\Emotion Detector\Emotion_Detection\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["file", "label"])
y = df["label"]

# 2. Train-Test split (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")

# 3. Define models to try


best_model = None
best_f1 = 0

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for SVM (best choice for audio features)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(class_weight='balanced', probability=True))
])

param_grid = {
    'model__C': [0.1, 1, 5, 10],
    'model__gamma': [0.01, 0.1, 1]
}

grid = GridSearchCV(pipe, param_grid,
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=2)

print("\n🚀 Running Hyperparameter Search (this may take a bit)...")
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\n🔥 Best Parameters Found:", grid.best_params_)


# Final evaluation on test set
test_preds = best_model.predict(X_test)

print("\n===============================")
print("🏁 Best Model Test Evaluation")
print("===============================")
print(classification_report(y_test, test_preds))
print("Accuracy:", accuracy_score(y_test, test_preds))
print("Macro F1:", f1_score(y_test, test_preds, average="macro"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_preds))

# 6. Save best model
MODEL_PATH = f"{MODEL_DIR}/emotion_model.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"\n🔥 Model saved to {MODEL_PATH}")
