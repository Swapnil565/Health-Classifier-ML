# Health-Classifier-ML
📌 README.md for ML-Classification-Models
md
Copy
Edit
# ML-Classification-Models

This repository contains various **classification models** trained on a dataset that predicts whether an individual is **healthy** or not based on different lifestyle attributes. The models are trained, saved, and can be reloaded on any machine using `pickle`.

## 📂 Project Structure
📦 ML-Classification-Models ┣ 📜 random_forest.pkl ┣ 📜 svm.pkl ┣ 📜 logistic_regression.pkl ┣ 📜 decision_tree.pkl ┣ 📜 naive_bayes.pkl ┣ 📜 knn.pkl ┣ 📜 gradient_boosting.pkl ┣ 📜 train_models.py ┣ 📜 load_models.py ┣ 📜 requirements.txt ┗ 📜 README.md

markdown
Copy
Edit

## 🚀 Features
- Trains **multiple classification models**: 
  - ✅ Random Forest
  - ✅ Support Vector Machine (SVM)
  - ✅ Logistic Regression
  - ✅ Decision Tree
  - ✅ Naive Bayes
  - ✅ K-Nearest Neighbors (KNN)
  - ✅ Gradient Boosting
- Saves trained models using **pickle**.
- Can be loaded and used on another device.
- Supports **decision boundary visualization** for `phy_fitness` and `mindfulness`.

## 📌 Setup Instructions
1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/ML-Classification-Models.git
cd ML-Classification-Models
2️⃣ Install Required Packages

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train and Save Models

bash
Copy
Edit
python train_models.py
4️⃣ Load and Use Saved Models

python
Copy
Edit
python load_models.py
📊 Decision Boundary Plot
A decision boundary plot is generated for phy_fitness vs mindfulness using trained models.

📂 Dataset
6000 entries of individuals with physical and lifestyle attributes.
Target label: is_healthy (1 = Healthy, 0 = Not Healthy).
📌 Using Models on Another Machine
To load a saved model:

python
Copy
Edit
import pickle

def load_model(model_name):
    with open(f"{model_name}.pkl", "rb") as file:
        return pickle.load(file)

# Example usage:
model = load_model("random_forest")
✨ Future Enhancements
Hyperparameter tuning for better performance.
Feature selection to improve model efficiency.
Deploying model via API using Flask/FastAPI.
